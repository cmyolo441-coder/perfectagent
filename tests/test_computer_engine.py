from __future__ import annotations
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from fullagent.computer.engine import Computer, compact_messages, validate_plan, validate_report
from fullagent.computer.state import Settings, ComputerError, AGENT_IDS, load_checkpoint
from tests.support import Driver, plan


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.store = base/'project', base/'state'
        self.root.mkdir()
        self.engines = []
        self.settings = Settings(plan_rounds=2, repair_rounds=1, token_budget=800000, request_timeout=5, work_steps=6, review_steps=3)

    def tearDown(self):
        for engine in self.engines:
            if engine.running:
                engine.cancel()
            engine.join(6)

    def engine(self, driver=None, approve=None, settings=None, listener=None):
        driver = driver or Driver()
        e = Computer(self.root, self.store, settings or self.settings, driver.factory,
                     approve or (lambda kind, details: True), listener)
        self.engines.append(e)
        return e

    def complete(self, e):
        e.start('Create the eight-part fixture project')
        self.assertTrue(e.join(20), 'engine did not finish')
        return e.snapshot()

    def test_eight_real_threads_overlap_and_write_verified_files(self):
        driver = Driver(barrier=True)
        e = self.engine(driver)
        d = self.complete(e)
        self.assertEqual(driver.peak, 8)
        self.assertEqual(d['status'], 'completed', d.get('error'))
        self.assertEqual(len(list(self.root.glob('a*.py'))), 8)
        self.assertTrue(all(c['ok'] for c in d['checks']))
        self.assertGreater(d['reported_tokens'], 0)
        self.assertEqual(d['reserved_tokens'], 0)
        self.assertEqual(d['estimated_tokens'], 0)
        self.assertTrue((e.board.path/'report.md').is_file())
        self.assertEqual(len(d['files']), 8)
        self.assertEqual(set(d['tasks']), set(AGENT_IDS))

    def test_independent_threads_are_bounded_by_configured_cap(self):
        driver = Driver()
        d = self.complete(self.engine(driver, settings=replace(self.settings, max_parallel=2)))
        self.assertEqual(d['status'], 'completed', d.get('error'))
        self.assertLessEqual(driver.peak, 2)
        self.assertEqual(driver.peak, 2)

    def test_actual_failing_check_drives_repair_then_reverification(self):
        driver = Driver(bug=True)
        e = self.engine(driver)
        d = self.complete(e)
        self.assertEqual(d['status'], 'completed', d.get('error'))
        self.assertEqual(d['repair_round'], 1)
        log = (e.board.path/'events.jsonl').read_text()
        self.assertIn('FAIL', log)
        self.assertIn('repair.started', log)
        self.assertIn('FIXED', (self.root/'a3.py').read_text())
        backups = list((e.board.path/'backups').iterdir())
        self.assertTrue(any('BUG' in p.read_text() for p in backups))

    def test_review_blocker_never_becomes_success(self):
        d = self.complete(self.engine(Driver(review_failure=True), settings=replace(self.settings, repair_rounds=0)))
        self.assertEqual(d['status'], 'needs_attention')
        self.assertTrue(all(c['ok'] for c in d['checks']))

    def test_denied_plan_means_no_project_writes(self):
        d = self.complete(self.engine(approve=lambda kind, details: False))
        self.assertEqual(d['status'], 'planned')
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(d['checks'], [])

    def test_plan_only_does_not_request_implementation_approval(self):
        approvals=[]
        d=self.complete(self.engine(settings=replace(self.settings, plan_only=True),
                                   approve=lambda k,d: approvals.append(k)))
        self.assertEqual(d['status'], 'planned')
        self.assertEqual(approvals, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_denied_command_is_failed_check_not_fake_success(self):
        d=self.complete(self.engine(approve=lambda kind, details: kind=='plan', settings=replace(self.settings, repair_rounds=0)))
        self.assertEqual(d['status'], 'needs_attention')
        self.assertFalse(d['checks'][0]['ok'])
        self.assertIn('denied', d['checks'][0]['error'])

    def test_bad_plan_is_repaired_before_execution(self):
        driver=Driver(malformed_plan=True)
        d=self.complete(self.engine(driver))
        self.assertEqual(d['status'], 'completed', d.get('error'))
        self.assertEqual(driver.plans, 2)

    def test_provider_failure_is_visible_and_checkpointed(self):
        e=self.engine(Driver(failure=('a4','research')))
        d=self.complete(e)
        self.assertEqual(d['status'], 'needs_attention')
        self.assertEqual(d['reports']['research']['a4']['status'], 'error')
        self.assertFalse(list(self.root.iterdir()))
        stored=load_checkpoint(self.store,d['id'])
        self.assertEqual(stored['status'],'needs_attention')
        self.assertGreater(stored['estimated_tokens'], 0)

    def test_missing_client_credentials_fail_closed(self):
        def missing(_):
            raise ComputerError('No test credentials')
        e=Computer(self.root,self.store,self.settings,missing)
        self.engines.append(e)
        d=self.complete(e)
        self.assertEqual(d['status'],'error')
        self.assertIn('credentials',d['error'])
        self.assertEqual(d['requests'],0)

    def test_budget_reservations_stop_before_sending_request(self):
        driver=Driver()
        d=self.complete(self.engine(driver, settings=replace(self.settings, token_budget=1000)))
        self.assertEqual(d['status'],'budget_exhausted')
        self.assertEqual(driver.calls,[])
        self.assertEqual(d['reserved_tokens'],0)

    def test_pause_then_cancel_stops_scheduling_and_drains(self):
        driver=Driver()
        driver.release=threading.Event()
        e=self.engine(driver)
        e.start('Test paused project')
        self.assertTrue(driver.reached.wait(3))
        e.pause()
        driver.release.set()
        time.sleep(.2)
        self.assertEqual(e.snapshot()['status'],'paused')
        self.assertFalse(list(self.root.iterdir()))
        e.cancel()
        self.assertTrue(e.join(5))
        self.assertEqual(e.snapshot()['status'],'cancelled')
        self.assertEqual(driver.active,0)

    def test_cancelled_mission_resumes_without_blind_writes(self):
        driver=Driver()
        holder={}
        cancelled=[False]
        def listener(ev):
            if ev['kind']=='file.written' and not cancelled[0]:
                cancelled[0]=True
                holder['e'].cancel()
        e=self.engine(driver,listener=listener)
        holder['e']=e
        d=self.complete(e)
        self.assertEqual(d['status'],'cancelled',d.get('error'))
        mid=d['id']
        e.start(resume_id=mid)
        self.assertTrue(e.join(20))
        d=e.snapshot()
        self.assertEqual(d['status'],'completed',d.get('error'))
        self.assertEqual(d['id'],mid)
        self.assertEqual(len(d['files']),8)

    def test_workspace_lease_blocks_concurrent_runs(self):
        driver=Driver()
        driver.release=threading.Event()
        e=self.engine(driver)
        e.start('Hold workspace for test')
        self.assertTrue(driver.reached.wait(3))
        other=self.engine()
        with self.assertRaises(ComputerError):
            other.start('Conflicting workspace mission')
        e.cancel()
        self.assertTrue(e.join(5))
        self.assertEqual(self.complete(other)['status'],'completed')

    def test_dependency_tasks_wait_for_upstream_finish(self):
        d=self.complete(self.engine(Driver(deps=True)))
        self.assertEqual(d['status'],'completed',d.get('error'))


class ContractTests(unittest.TestCase):
    def test_valid_plan_has_exactly_eight_owners(self):
        self.assertEqual(len(validate_plan(plan())['tasks']),8)

    def test_cycle_rejected(self):
        p=plan(); p['tasks'][0]['depends_on']=['a2'];p['tasks'][1]['depends_on']=['a1']
        with self.assertRaises(ComputerError): validate_plan(p)

    def test_file_ownership_overlap_rejected(self):
        p=plan(); p['tasks'][0]['files']=['src/'];p['tasks'][1]['files']=['src/api.py']
        with self.assertRaises(ComputerError): validate_plan(p)

    def test_duplicate_agent_rejected(self):
        p=plan();p['tasks'][1]['id']='a1'
        with self.assertRaises(ComputerError): validate_plan(p)

    def test_root_secret_and_traversal_scopes_rejected(self):
        for path in ('.','../escape','/tmp/file','.env','src/../../x','.git/hooks/'):
            with self.subTest(path=path):
                p=plan();p['tasks'][0]['files']=[path]
                with self.assertRaises(ComputerError): validate_plan(p)

    def test_checkless_plan_rejected(self):
        p=plan();p['checks']=[]
        with self.assertRaises(ComputerError):validate_plan(p)

    def test_unfinished_status_is_not_silently_done(self):
        with self.assertRaises(ComputerError):validate_report('Everything done, trust me')
        r=validate_report(json.dumps({'status':'done','summary':'Finished','issues':['Still a blocker']}))
        self.assertEqual(r['status'],'blocked')

    def test_legacy_crew_explicit_error_is_not_marked_done(self):
        from types import SimpleNamespace
        from fullagent.crew import Crew
        from fullagent.kernel import EventLog
        from tests.support import answer
        with tempfile.TemporaryDirectory() as td:
            model=SimpleNamespace(id='fixture',provider='fixture',supports_tools=True)
            crew=Crew(EventLog(Path(td)/'crew.jsonl'),None,model,None,
                      chat=lambda *a:answer('STATUS: ERROR\\nSUMMARY: intentional fixture failure'))
            agent=crew.spawn('fixture task')
            crew.wait([agent.id],timeout=3)
            self.assertEqual(agent.state,'error')
            crew.close(agent.id)

    def test_compaction_preserves_complete_tool_protocol(self):
        messages=[{'role':'system','content':'system'}, {'role':'user','content':'goal'}]
        for i in range(30):
            messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':str(i),'function':{'name':'read','arguments':'{}'}}]},
                             {'role':'tool','tool_call_id':str(i),'content':'x'*1000}])
        compact_messages(messages,7000)
        self.assertLess(len(json.dumps(messages)),7200)
        calls=set()
        for m in messages:
            if m['role']=='assistant': calls.update(t['id'] for t in m.get('tool_calls',[]))
            if m['role']=='tool': self.assertIn(m['tool_call_id'],calls)
