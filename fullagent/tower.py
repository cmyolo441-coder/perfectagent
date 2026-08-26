"""TOWER — the web control tower: mission control in a browser tab.

A real HTTP server (stdlib only — http.server + threading, zero new
dependencies) that streams the whole agent's world to a dark, animated
single-page dashboard:

    /api/state      live snapshot: cost, tokens, tool calls, errors,
                    goal, crew roster, budget, branches, brain stats —
                    one JSON, one poll
    /api/events     the event river: every new kernel event since a seq
                    (the page polls and paints them as they seal)
    /api/timeline   the scrubber strip (Theater frames)
    /api/command    POST {"text": "..."} — a REAL agent turn launched in
                    the background; its events light up the river live.
                    {"sleep": true} runs the Brain's consolidation pass.

The page is fully self-contained (no CDN, no build step): vanilla JS,
CSS grid, dark mission-control theme. Launch with /tower [port] from
the TUI — the browser opens itself.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .kernel import EventLog, fold
from .theater import Theater, _summary

_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>FullAgent Control Tower</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#21262d;--fg:#c9d1d9;--dim:#8b949e;
--acc:#58a6ff;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--pink:#bc8cff}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,
Consolas,monospace;padding:18px}
h1{font-size:17px;color:var(--acc);letter-spacing:.5px}
h1 .pulse{display:inline-block;width:9px;height:9px;border-radius:50%;
background:var(--ok);margin-right:8px;animation:p 1.6s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:10px 12px}
.card .v{font-size:21px;color:var(--fg)}
.card .l{font-size:11px;color:var(--dim);text-transform:uppercase;
letter-spacing:.6px}
.cols{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:12px;min-height:220px;overflow:auto;max-height:46vh}
.panel h2{font-size:12px;color:var(--dim);text-transform:uppercase;
letter-spacing:.6px;margin-bottom:8px}
#river div{padding:2px 6px;border-left:2px solid var(--line);margin:3px 0;
animation:in .4s ease}
@keyframes in{from{background:#1c2432}to{background:transparent}}
#river .seq{color:var(--dim);margin-right:8px}
#river .t{color:var(--acc)}
#river .u{color:var(--fg)}
#river .err{border-left-color:var(--bad)}
#river .ok{border-left-color:var(--ok)}
#tl{display:flex;gap:3px;flex-wrap:wrap;margin-top:8px}
#tl i{width:9px;height:16px;border-radius:2px;background:var(--acc);
cursor:pointer;opacity:.75}
#tl i:hover{opacity:1;outline:1px solid var(--pink)}
form{display:flex;gap:8px;margin-top:12px}
input{flex:1;background:var(--card);border:1px solid var(--line);
border-radius:7px;color:var(--fg);padding:9px 12px;font:inherit}
button{background:#1f6feb;border:none;border-radius:7px;color:#fff;
padding:9px 16px;font:inherit;cursor:pointer}
button:hover{background:#388bfd}
.muted{color:var(--dim)} .ok{color:var(--ok)} .bad{color:var(--bad)}
.pink{color:var(--pink)}
</style></head><body>
<h1><span class="pulse"></span>FULLAGENT CONTROL TOWER
<span class="muted" id="branch"></span></h1>
<div class="grid" id="cards"></div>
<div class="cols">
 <div class="panel"><h2>Event river — live</h2><div id="river"></div></div>
  <div><div class="panel"><h2>Crew / brain</h2><div id="crew"
  class="muted">—</div></div>
 <div class="panel" style="margin-top:12px"><h2>Timeline scrubber</h2>
  <div id="tl"></div><div id="frame" class="muted" style="margin-top:8px">
  click a bar to inspect the frame</div></div></div>
</div>
<form id="cmd"><input id="txt" placeholder="command the agent… (real turn)">
<button>SEND</button></form>
<script>
let lastSeq=-1;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;',
'>':'&gt;','"':'&quot;'}[c]));
async function pollState(){try{const r=await fetch('/api/state');
const s=await r.json();const c=s.cards||{};
$('cards').innerHTML=[['cost',s.cost],['tokens',s.tokens],
['tool calls',s.tool_calls],['errors',s.errors],['messages',s.messages],
['files',s.files],['crew agents',s.crew_agents],['branches',s.branches]]
.map(([l,v])=>`<div class="card"><div class="v">${esc(v)}</div>
<div class="l">${l}</div></div>`).join('');
$('branch').textContent=' · '+s.branch;
$('crew').innerHTML=(s.crew||[]).map(a=>`${a.icon} <b>${esc(a.nickname)}
</b> <span class="muted">(${a.role})</span> ${a.state==='done'?
'<span class=ok>✓</span>':a.state==='error'?'<span class=bad>✗</span>':
a.state}</>`).join('<br>')||'crew is idle';
if(s.brain)$('crew').innerHTML+=`<br><span class="pink">brain:</span>
${s.brain}`;}catch(e){}}
async function pollEvents(){try{const r=await fetch(
'/api/events?since='+lastSeq);const evs=await r.json();
for(const e of evs){lastSeq=Math.max(lastSeq,e.seq);
const d=document.createElement('div');
d.className=e.type.includes('error')||e.type.includes('fail')?'err':
(e.type.includes('done')||e.type.includes('pass'))?'ok':'';
d.innerHTML=`<span class="seq">${e.seq}</span><span class="t">
${esc(e.type)}</span> <span class="u">${esc(e.summary||'')}</span>`;
$('river').prepend(d);}
if(evs.length)drawTimeline();}catch(e){}}
async function drawTimeline(){try{const r=await fetch('/api/timeline');
const f=await r.json();$('tl').innerHTML=f.frames.slice(-120).map(x=>
`<i title="${x.seq} ${esc(x.type)}" data-s="${x.seq}"></i>`).join('');
[...$('tl').children].forEach(i=>i.onclick=async()=>{const r=await fetch(
'/api/frame?seq='+i.dataset.s);const fr=await r.json();
$('frame').innerHTML=`<b class="pink">seq ${fr.seq}</b> ${esc(fr.type)}
<br><span class="muted">${esc(fr.summary||'')}</span><br>msgs
${fr.messages} · tools ${fr.tool_calls} · cost $${fr.cost}`;});}
catch(e){}}
$('cmd').onsubmit=async ev=>{ev.preventDefault();const t=$('txt').value.
trim();if(!t)return;$('txt').value='';await fetch('/api/command',
{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:t})});};
pollState();pollEvents();drawTimeline();
setInterval(pollState,2000);setInterval(pollEvents,1200);
</script></body></html>"""


class Tower:
    """The mission-control web server over a live agent."""

    def __init__(self, agent, host: str = "127.0.0.1") -> None:
        self.agent = agent
        self.log = agent.log
        self.host = host
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.url = ""

    # -- state assembly ----------------------------------------------------------

    def state(self) -> dict:
        st = fold(self.log)
        ag = self.agent
        crew = []
        crew_obj = getattr(ag, "crew", None)
        if crew_obj is not None:
            crew = [a.to_dict() for a in crew_obj.list()[:12]]
            for a, d in zip(crew_obj.list()[:12], crew):
                d["icon"] = a.icon
        brain = getattr(ag, "brain", None)
        brain_txt = ""
        if brain is not None:
            bstats = brain.stats()
            brain_txt = (f"{bstats['total']} memories · "
                         f"{bstats['alive']} alive · retention "
                         f"{bstats['avg_retention']}")
        return {"branch": self.log.branch,
                "cost": f"${st.cost_usd:.4f}",
                "tokens": f"{(st.tokens_in + st.tokens_out):,}",
                "tool_calls": st.tool_calls, "errors": st.tool_errors,
                "messages": len(st.messages),
                "files": len(st.files_touched),
                "crew_agents": len(crew),
                "branches": len(self.log.branches()),
                "crew": crew, "brain": brain_txt}

    # -- command execution -----------------------------------------------------------

    def command(self, payload: dict) -> dict:
        """Run a REAL agent turn in the background; events light the
        river. Sleep command runs the Brain's consolidation."""
        ag = self.agent
        if payload.get("sleep"):
            brain = getattr(ag, "brain", None)
            if brain is None:
                return {"ok": False, "error": "no brain attached"}
            stats = brain.sleep()
            return {"ok": True, "slept": stats}
        text = str(payload.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "empty command"}

        def _run() -> None:
            try:
                ag.run_turn(text, on_token=lambda t: None,
                            on_reasoning=lambda r: None,
                            on_tool_call=lambda e: None,
                            on_tool_update=lambda e: None,
                            on_status=lambda s: None,
                            approve=lambda tool, args: False)
            except Exception:      # the river shows the sealed error
                pass

        threading.Thread(target=_run, name="tower:turn",
                         daemon=True).start()
        return {"ok": True, "launched": text[:100]}

    # -- HTTP plumbing ----------------------------------------------------------------

    def _handler(self) -> type:
        tower = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):     # silence per-request noise
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    body = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/state":
                    self._json(tower.state())
                elif self.path.startswith("/api/events"):
                    since = -1
                    if "since=" in self.path:
                        try:
                            since = int(self.path.split("since=")[1]
                                        .split("&")[0])
                        except ValueError:
                            pass
                    evs = [{"seq": e.seq, "type": e.type,
                            "actor": e.actor, "summary": _summary(e)}
                           for e in tower.log.events() if e.seq > since]
                    self._json({"events": evs[-200:]})
                elif self.path.startswith("/api/timeline"):
                    self._json({"frames":
                                Theater(tower.log).frames()[-120:]})
                elif self.path.startswith("/api/frame"):
                    try:
                        seq = int(self.path.split("seq=")[1].split("&")[0])
                    except (ValueError, IndexError):
                        self._json({"error": "bad seq"}, 400)
                        return
                    f = Theater(tower.log).frame(seq)
                    if f is None:
                        self._json({"error": "no such frame"}, 404)
                        return
                    self._json({"seq": f.seq, "type": f.type,
                                "summary": f.summary,
                                "messages": len(f.state.get("messages",
                                                            [])),
                                "tool_calls": f.state.get("tool_calls", 0),
                                "cost": f.state.get("cost_usd", 0.0)})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path != "/api/command":
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    self._json({"error": "bad json"}, 400)
                    return
                if not isinstance(payload, dict):
                    # command() does payload.get(...) — an array/string
                    # body would AttributeError inside the handler thread
                    # and close the connection with NO response at all
                    self._json({"error": "payload must be a JSON object"},
                               400)
                    return
                self._json(tower.command(payload))

        return Handler

    # -- lifecycle ------------------------------------------------------------------------

    def start(self, port: int = 7860, open_browser: bool = True) -> str:
        """Start the server on a daemon thread; returns the URL."""
        if self.server is not None:
            return self.url
        self.server = ThreadingHTTPServer((self.host, int(port)),
                                          self._handler())
        self.url = f"http://{self.host}:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="tower:http", daemon=True)
        self.thread.start()
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return self.url

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            # shutdown() only stops the serve loop — without server_close()
            # the listening socket leaks until GC and an immediate restart
            # on the same port can fail to bind
            self.server.server_close()
            self.server = None
            self.thread = None


# ---------------------------------------------------------------------------
# Self-test — real HTTP round-trips against a duck-typed agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import urllib.request
    from pathlib import Path
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "tower.jsonl")
        log.append("user.message", {"text": "hello tower"})
        log.append("tool.call", {"name": "read_file",
                                 "args": {"path": "x.py"}})
        log.append("assistant.message", {"text": "done reading"})
        fake = SimpleNamespace(log=log, crew=None, brain=None)
        tower = Tower(fake)
        url = tower.start(port=0, open_browser=False)
        assert url.startswith("http://")

        def get(path: str):
            with urllib.request.urlopen(url + path, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))

        # state snapshot is real fold data
        state = get("/api/state")
        assert state["tool_calls"] == 1 and state["branch"] == "main"
        assert state["cost"].startswith("$")

        # event river: since=-1 returns everything; since=head only new
        evs = get("/api/events?since=-1")["events"]
        assert [e["seq"] for e in evs] == [0, 1, 2], evs
        assert evs[0]["type"] == "user.message"
        after2 = get(f"/api/events?since={evs[-1]['seq']}")["events"]
        assert after2 == []
        log.append("fact.learned", {"fact": "tower works"})
        after3 = get(f"/api/events?since={evs[-1]['seq']}")["events"]
        assert len(after3) == 1 and after3[0]["type"] == "fact.learned"

        # timeline + frame inspection
        tl = get("/api/timeline")["frames"]
        assert len(tl) == 4 and tl[0]["type"] == "user.message"
        frame = get("/api/frame?seq=1")
        assert frame["tool_calls"] == 1 and "read_file" in \
            frame["summary"]

        # the page serves
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            page = r.read().decode("utf-8")
        assert "CONTROL TOWER" in page and "api/events" in page

        # bad paths are clean 404s
        import urllib.error
        try:
            urllib.request.urlopen(url + "/nope", timeout=10)
            raise AssertionError("404 expected")
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # command endpoint: empty text rejected cleanly
        req = urllib.request.Request(
            url + "/api/command", data=b'{"text": ""}',
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert not json.loads(r.read())["ok"]

        tower.stop()
        print("TOWER SELF-TEST PASS")
