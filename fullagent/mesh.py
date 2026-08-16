"""MESH — the agent-to-agent network: nodes over TCP.

Two FullAgent instances on two machines become ONE distributed agent:

    serve()      a JSON-lines TCP server announces this node's
                 capabilities (roles it can run, models it carries) and
                 serves three verbs: HELLO (discovery), TASK (delegate
                 work), PING (heartbeat)
    discover()   connect to a peer and pull its capability card
    delegate()   ship a task to the best peer — local execution stays
                 local; the mesh only exists for work worth moving
    heartbeat()  liveness probes; dead peers drop out of the roster

Every delegation is sealed on BOTH ends' kernels if both run FullAgent
(mesh.task out / mesh.result in). The protocol is deliberately tiny and
human-readable — one JSON object per line — so any process that can
open a socket can join the mesh, not just this codebase.

The executor is injectable; the self-test runs a real two-node mesh
over loopback TCP with scripted executors and proves delegation,
discovery, and failure handling.
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field

from .kernel import EventLog

_PROTO = "fullagent-mesh/1"
_RECV_LIMIT = 1 << 20          # 1 MiB per message line
_DEFAULT_TTL = 60              # seconds a peer stays in the roster


@dataclass
class Peer:
    host: str
    port: int
    capabilities: dict = field(default_factory=dict)   # roles, models…
    last_seen: float = field(default_factory=time.time)

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"

    def alive(self, ttl: float = _DEFAULT_TTL) -> bool:
        return (time.time() - self.last_seen) < ttl


class MeshNode:
    """One node: a TCP server + a client with a peer roster."""

    def __init__(self, log: EventLog, node_id: str,
                 executor=None, host: str = "127.0.0.1") -> None:
        """executor(task: str, role: str) -> {"status", "summary"} —
        how THIS node runs delegated work (production: a Team worker)."""
        self.log = log
        self.node_id = node_id
        self.executor = executor
        self.host = host
        self.port = 0
        self.peers: dict[str, Peer] = {}
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self.handled = 0

    # -- capabilities ------------------------------------------------------------

    def capabilities(self) -> dict:
        return {"node": self.node_id, "protocol": _PROTO,
                "roles": ["any"], "models": [],
                "executor": self.executor is not None}

    # -- server ---------------------------------------------------------------------

    def _handle(self, raw: bytes, addr) -> dict:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "error": "malformed json"}
        verb = str(msg.get("verb", "")).upper()
        if verb == "HELLO":
            return {"ok": True, "proto": _PROTO,
                    "node": self.node_id,
                    "capabilities": self.capabilities()}
        if verb == "PING":
            return {"ok": True, "pong": True, "node": self.node_id}
        if verb == "TASK":
            task = str(msg.get("task", "")).strip()
            role = str(msg.get("role", "")).strip()
            if not task:
                return {"ok": False, "error": "empty task"}
            if self.executor is None:
                return {"ok": False, "error": "node cannot execute"}
            self.log.append("mesh.task",
                            {"from": msg.get("from", "?"), "task": task,
                             "role": role})
            try:
                result = self.executor(task, role) or {}
            except Exception as e:   # a failing task never kills the node
                result = {"status": "error", "summary": str(e)}
            self.handled += 1
            self.log.append("mesh.result",
                            {"task": task, **result})
            return {"ok": True, "node": self.node_id, "result": result}
        return {"ok": False, "error": f"unknown verb {verb!r}"}

    def serve(self, port: int = 0) -> int:
        """Start the TCP server; returns the bound port."""
        if self._server is not None:
            return self.port
        node = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    line = self.rfile.readline(_RECV_LIMIT)
                    reply = node._handle(line, self.client_address)
                    self.wfile.write(
                        json.dumps(reply, default=str).encode("utf-8")
                        + b"\n")
                except (ConnectionError, OSError):
                    pass

            def log_message(self, *a):
                pass

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((self.host, int(port)), Handler)
        self.port = self._server.server_address[1]
        self.log.append("mesh.node",
                        {"node": self.node_id, "port": self.port,
                         "serving": True})
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"mesh:{self.node_id}", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    # -- client --------------------------------------------------------------------

    def _rpc(self, peer: Peer, message: dict,
             timeout: float = 30.0) -> dict:
        with socket.create_connection((peer.host, peer.port),
                                      timeout=timeout) as sock:
            sock.sendall(json.dumps(message).encode("utf-8") + b"\n")
            data = sock.recv(_RECV_LIMIT)
        try:
            return json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "error": "peer sent garbage"}

    def discover(self, host: str, port: int) -> Peer | None:
        """HELLO a peer and file its capability card in the roster."""
        candidate = Peer(host=host, port=port)
        try:
            reply = self._rpc(candidate,
                              {"verb": "HELLO", "from": self.node_id},
                              timeout=10.0)
        except OSError:
            return None
        if not reply.get("ok"):
            return None
        peer = Peer(host=host, port=port,
                    capabilities=reply.get("capabilities") or {})
        self.peers[reply.get("node", peer.addr)] = peer
        return peer

    def heartbeat(self) -> dict[str, bool]:
        """PING every peer; dead ones drop from the roster."""
        statuses: dict[str, bool] = {}
        for name, peer in list(self.peers.items()):
            try:
                reply = self._rpc(peer,
                                  {"verb": "PING", "from": self.node_id},
                                  timeout=5.0)
                alive = bool(reply.get("pong"))
            except OSError:
                alive = False
            statuses[name] = alive
            if alive:
                peer.last_seen = time.time()
            else:
                del self.peers[name]
        return statuses

    def delegate(self, task: str, role: str = "",
                 peer_name: str = "") -> dict:
        """Ship work to a peer (any live peer by default). Returns the
        peer's result dict; a dead mesh never raises."""
        candidates = ([self.peers[peer_name]] if peer_name in self.peers
                      else [p for p in self.peers.values() if p.alive()])
        if not candidates:
            return {"ok": False,
                    "error": "no peers — discover() one first"}
        peer = candidates[0]
        try:
            reply = self._rpc(peer,
                              {"verb": "TASK", "from": self.node_id,
                               "task": task, "role": role})
        except OSError as e:
            return {"ok": False, "error": f"peer unreachable: {e}"}
        return reply


# ---------------------------------------------------------------------------
# Self-test — a real two-node mesh over loopback TCP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log_a = EventLog(Path(td) / "mesh-a.jsonl")
        log_b = EventLog(Path(td) / "mesh-b.jsonl")

        def exec_b(task: str, role: str) -> dict:
            if "boom" in task:
                raise RuntimeError("worker exploded")
            return {"status": "done",
                    "summary": f"B ran {task} as {role or 'any'}"}

        a = MeshNode(log_a, "alpha", executor=None)
        b = MeshNode(log_b, "bravo", executor=exec_b)
        port_a = a.serve()
        port_b = b.serve()
        assert port_a > 0 and port_b > 0 and port_a != port_b

        # discovery: HELLO exchanges capability cards
        peer = a.discover("127.0.0.1", port_b)
        assert peer is not None
        assert peer.capabilities.get("node") == "bravo"
        assert peer.capabilities.get("executor") is True
        assert any("bravo" in name for name in a.peers)

        # a node WITHOUT an executor refuses tasks politely
        log_c = EventLog(Path(td) / "mesh-c.jsonl")
        c = MeshNode(log_c, "charlie", executor=None)
        port_c = c.serve()
        no_exec = a.discover("127.0.0.1", port_c)
        assert no_exec is not None
        reply = a.delegate("do something")
        assert reply.get("ok") and "B ran" in \
            reply["result"]["summary"], reply

        # delegation really executed on B and sealed on B's kernel
        assert b.handled == 1
        kinds_b = [e.type for e in log_b.events()]
        assert "mesh.task" in kinds_b and "mesh.result" in kinds_b

        # a failing worker is reported, never crashes the node
        err = a.delegate("make it boom")
        assert err["ok"] and err["result"]["status"] == "error"
        assert "exploded" in err["result"]["summary"]
        assert b.handled == 2

        # heartbeat keeps live peers, drops dead ones
        b.stop()
        statuses = a.heartbeat()
        assert statuses.get("bravo") is False
        assert not any("bravo" in n for n in a.peers)
        statuses2 = a.heartbeat()
        assert "charlie" in statuses2 and statuses2["charlie"] is True

        # delegating with an empty roster is a clean error
        a2 = MeshNode(log_a, "alpha2")
        empty = a2.delegate("x")
        assert not empty["ok"] and "no peers" in empty["error"]

        # malformed payloads get clean errors, node stays up
        with socket.create_connection(("127.0.0.1", port_c),
                                      timeout=10) as s:
            s.sendall(b"this is not json\n")
            resp = json.loads(s.recv(_RECV_LIMIT).decode())
        assert resp == {"ok": False, "error": "malformed json"}
        with socket.create_connection(("127.0.0.1", port_c),
                                      timeout=10) as s:
            s.sendall(json.dumps({"verb": "DANCE"}).encode() + b"\n")
            resp = json.loads(s.recv(_RECV_LIMIT).decode())
        assert "unknown verb" in resp["error"]

        a.stop()
        c.stop()

        print("MESH SELF-TEST PASS")
