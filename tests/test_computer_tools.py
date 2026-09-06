from __future__ import annotations
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fullagent.computer.state import Board, Settings, ComputerError, Cancelled, BudgetExceeded, load_checkpoint, safe_text
from fullagent.computer.tools import WorkspaceTools, digest
from fullagent.computer.research import Research
from tests.support import plan


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'project';self.root.mkdir()
        self.settings=Settings(command_timeout=3)
        self.cancelled=threading.Event()
        def control():
            if self.cancelled.is_set():raise Cancelled('test cancellation')
        self.control=control
        self.board=Board(self.root,self.base/'state','test project',self.settings)
        self.tools=WorkspaceTools(self.root,self.board,Research(self.board,control),control,lambda k,d:True)
        self.tools.set_plan(plan());self.tools.approved_plan=True

    def test_atomic_write_backup_and_stale_hash_conflict(self):
        p=self.root/'a1.py';p.write_text('before\n');h=digest(p.read_bytes())
        result=self.tools.write_file('a1','a1.py','after\n',h)
        self.assertTrue(result['ok']);self.assertEqual(p.read_text(),'after\n')
        self.assertEqual((self.board.path/'backups'/h).read_text(),'before\n')
        with self.assertRaises(ComputerError):self.tools.write_file('a1','a1.py','wrong',h)
        self.assertEqual(p.read_text(),'after\n')

    def test_read_reports_full_hash_and_page_boundaries(self):
        p=self.root/'a1.py';p.write_text('one\ntwo\nthree\n')
        r=self.tools.read_file('a1.py',offset=2,limit=1)
        self.assertEqual(r['sha256'],digest(p.read_bytes()))
        self.assertEqual(r['content'],'2: two');self.assertTrue(r['truncated'])

    def test_optimistic_write_race_has_one_winner(self):
        p=self.root/'a1.py';p.write_text('base');h=digest(p.read_bytes())
        outcomes=[]
        barrier=threading.Barrier(2)
        def worker(value):
            barrier.wait()
            try:self.tools.write_file('a1','a1.py',value,h);outcomes.append('won')
            except ComputerError:outcomes.append('conflict')
        threads=[threading.Thread(target=worker,args=(v,)) for v in ('first','second')]
        for t in threads:t.start()
        for t in threads:t.join(5)
        self.assertCountEqual(outcomes,['won','conflict'])
        self.assertEqual(len(self.board.data['files']),1)

    def test_exact_edit_requires_one_match_and_new_hash(self):
        p=self.root/'a1.py';p.write_text('hello hello')
        with self.assertRaises(ComputerError):self.tools.edit_file('a1','a1.py','hello','bye',digest(p.read_bytes()))
        p.write_text('hello once')
        self.tools.edit_file('a1','a1.py','hello','bye',digest(p.read_bytes()))
        self.assertEqual(p.read_text(),'bye once')

    def test_scope_ownership_and_phase_permissions_enforced(self):
        with self.assertRaises(ComputerError):self.tools.write_file('a2','a1.py','bad','MISSING')
        with self.assertRaises(ComputerError):self.tools.execute('a1','write_file',{'path':'a1.py','content':'bad','expected_sha256':'MISSING'},False,False)
        self.tools.approved_plan=False
        with self.assertRaises(ComputerError):self.tools.write_file('a1','a1.py','bad','MISSING')
        self.assertFalse((self.root/'a1.py').exists())

    def test_secrets_traversal_devices_and_absolute_paths_blocked(self):
        for name in ('../outside','/tmp/x','C:/outside','a/../../b','.env','.env.local','.git/config','.ssh/id_rsa','secrets.json','cert.pem','NUL','a/CON.txt','a\\b'):
            with self.subTest(name=name):
                with self.assertRaises(ComputerError):self.tools.resolve(name)

    @unittest.skipIf(os.name=='nt','symlink capability is policy-dependent on Windows')
    def test_symlink_and_hardlink_escape_blocked(self):
        outside=self.base/'outside';outside.write_text('do not touch')
        (self.root/'a1.py').symlink_to(outside)
        with self.assertRaises(ComputerError):self.tools.read_file('a1.py')
        (self.root/'a1.py').unlink()
        os.link(outside,self.root/'a1.py')
        with self.assertRaises(ComputerError):self.tools.read_file('a1.py')

    def test_binary_and_oversized_files_blocked(self):
        p=self.root/'a1.py';p.write_bytes(b'a\0b')
        with self.assertRaises(ComputerError):self.tools.read_file('a1.py')
        p.write_bytes(b'x'*(self.settings.max_file_bytes+1))
        with self.assertRaises(ComputerError):self.tools.read_file('a1.py')

    def test_list_and_literal_search_exclude_secret_and_cache_files(self):
        (self.root/'a1.py').write_text('needle\n')
        (self.root/'.env').write_text('needle SECRET')
        (self.root/'node_modules').mkdir();(self.root/'node_modules'/'x.py').write_text('needle')
        self.assertEqual(self.tools.list_files()['files'],['a1.py'])
        self.assertEqual(len(self.tools.search_code('needle')['matches']),1)

    def test_command_streams_actual_output_and_exit(self):
        r=self.tools.run_command('a5',[sys.executable,'-u','-c','print("ACTUAL OUTPUT")'])
        self.assertTrue(r['ok']);self.assertEqual(r['exit_code'],0)
        self.assertIn('ACTUAL OUTPUT',r['stdout'])
        self.assertIn('output',(self.board.path/'events.jsonl').read_text())

    def test_command_denial_does_not_execute(self):
        self.tools.approve=lambda k,d:False
        with self.assertRaises(ComputerError):self.tools.run_command('a5',[sys.executable,'-c','open("pwned","w").write("x")'])
        self.assertFalse((self.root/'pwned').exists())

    def test_command_failure_is_not_success(self):
        r=self.tools.run_command('a5',[sys.executable,'-c','raise SystemExit(7)'])
        self.assertFalse(r['ok']);self.assertEqual(r['exit_code'],7)

    def test_command_timeout_is_bounded(self):
        start=time.monotonic()
        r=self.tools.run_command('a5',[sys.executable,'-c','import time;time.sleep(30)'],timeout=1)
        self.assertTrue(r['timed_out']);self.assertFalse(r['ok'])
        self.assertLess(time.monotonic()-start,5)

    def test_large_command_output_is_bounded(self):
        r=self.tools.run_command('a5',[sys.executable,'-c','print("x"*250000)'])
        self.assertTrue(r['ok']);self.assertTrue(r['output_truncated'])
        self.assertLessEqual(len(r['stdout']),self.settings.max_result_chars)
        self.assertLess((self.board.path/r['log']).stat().st_size,1_010_000)

    def test_command_gets_no_ambient_api_key(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'VERY_PRIVATE_API_KEY':'test-only-value'}):
            r=self.tools.run_command('a5',[sys.executable,'-c','import os; assert "VERY_PRIVATE_API_KEY" not in os.environ; print("clean")'])
        self.assertTrue(r['ok']);self.assertIn('clean',r['stdout'])

    def test_always_grant_is_exact_command_and_cwd(self):
        decisions=[]
        self.tools.approve=lambda k,d:decisions.append(d['argv']) or 'always'
        cmd=[sys.executable,'-c','print("one")']
        self.tools.run_command('a5',cmd);self.tools.run_command('a5',cmd)
        self.tools.run_command('a5',[sys.executable,'-c','print("two")'])
        self.assertEqual(len(decisions),2)

    def test_cancel_kills_command_and_returns_promptly(self):
        errors=[]
        def run():
            try:self.tools.run_command('a5',[sys.executable,'-u','-c','import time; print("ready",flush=True); time.sleep(30)'])
            except Cancelled:errors.append('cancelled')
        t=threading.Thread(target=run);t.start()
        end=time.monotonic()+3
        while time.monotonic()<end and not any(e['kind']=='output' for e in self.board.recent):time.sleep(.02)
        self.cancelled.set();t.join(5)
        self.assertFalse(t.is_alive());self.assertEqual(errors,['cancelled'])

    def test_peer_handoff_is_shared_not_executed(self):
        self.tools.share_note('a1','Need a2 to document interface','a2')
        notes=self.tools.read_board()['notes']
        self.assertEqual(notes[-1]['to'],'a2')
        self.assertEqual(list(self.root.iterdir()),[])


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.b=Board(self.path,self.path/'state','state tests',Settings(token_budget=1000))

    def test_atomic_reservations_prevent_concurrent_oversubscription(self):
        wins=[]
        def reserve():
            try:wins.append(self.b.reserve(600))
            except BudgetExceeded:pass
        threads=[threading.Thread(target=reserve) for _ in range(8)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(wins),1);self.assertEqual(self.b.data['reserved_tokens'],600)
        self.b.settle(wins[0],'a1',{'prompt_tokens':20,'completion_tokens':10})
        self.assertEqual(self.b.data['charged_tokens'],30)

    def test_unknown_usage_is_estimated_not_invented(self):
        ticket=self.b.reserve(500);self.b.settle(ticket,'a1',None)
        self.assertEqual(self.b.data['reported_tokens'],0)
        self.assertEqual(self.b.data['estimated_tokens'],500)
        self.assertEqual(self.b.data['reserved_tokens'],0)

    def test_crash_inflight_reservation_is_not_refunded(self):
        self.b.reserve(600)
        prior=load_checkpoint(self.path/'state',self.b.data['id'])
        resumed=Board(self.path,self.path/'state','state tests',Settings(token_budget=1000),previous=prior)
        self.assertEqual(resumed.data['charged_tokens'],600)
        self.assertEqual(resumed.data['reserved_tokens'],0)

    def test_settings_reject_truthy_false_strings_and_unbounded_workers(self):
        for args in ({'network':'false'},{'max_parallel':9},{'token_budget':-1},{'work_steps':True}):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):Settings(**args)

    def test_invalid_checkpoint_id_rejected(self):
        with self.assertRaises(ValueError):load_checkpoint(self.path/'state','../../etc/passwd')

    def test_logs_strip_terminal_escapes_and_obvious_secrets(self):
        raw='\x1b[2Jhello bearer '+('a'*35)+' password=abcdefghijk'
        clean=safe_text(raw)
        self.assertNotIn('\x1b',clean);self.assertNotIn('a'*35,clean);self.assertNotIn('abcdefghijk',clean)
        self.assertIn('[REDACTED]',clean)

    def test_dashboard_ring_buffer_is_bounded(self):
        for i in range(100):self.b.event('activity',message=str(i))
        self.assertEqual(len(self.b.recent),80)
