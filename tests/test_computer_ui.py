from __future__ import annotations
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fullagent.computer.engine import Computer
from fullagent.computer.state import Settings
from fullagent.computer.view import dashboard_lines,cells,fragments
from fullagent.computer.bridge import ComputerBridge

ROOT=Path(__file__).resolve().parents[1]


class FakeUI:
    def __init__(self):
        from fullagent.config import Config
        self.cfg=Config();self._busy=False
        self.events=[];self._approve_request=None;self._approve_result=False
        self.app=SimpleNamespace(invalidate=lambda:None,output=SimpleNamespace(get_size=lambda:SimpleNamespace(columns=80,rows=24)))
        self.agent=SimpleNamespace(log=SimpleNamespace(append=lambda *a,**k:None))
    def print_info(self,text,color=None):self.events.append(text)
    def _emit_user(self,text):self.events.append(text)


class UITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.computer=Computer(self.root,self.root/'state',Settings(),lambda i:None)

    def test_idle_dashboard_shows_eight_agents_and_no_fake_work(self):
        snap=self.computer.snapshot()
        text='\n'.join(line for _,line in dashboard_lines(snap,120,21))
        for i in range(1,9):self.assertIn(f'a{i}',text)
        self.assertIn('0 reported',text)
        self.assertIn('0/0 passed',text)
        self.assertIn('Enter a project goal',snap['goal'])

    def test_responsive_dashboard_has_no_display_cell_overflow(self):
        snap=self.computer.snapshot()
        snap['agents']['a1']['activity']='測試 🚀 very long activity '*50
        snap['recent']=[{'kind':'output','agent':'a1','message':'x'*300}]
        for width,height in ((120,24),(80,19),(60,18),(40,14),(20,6)):
            with self.subTest(width=width,height=height):
                rendered=dashboard_lines(snap,width,height)
                self.assertLessEqual(len(rendered),height)
                self.assertTrue(all(cells(text)<=width for _,text in rendered))

    def test_fragments_treat_source_markup_and_escapes_as_text(self):
        snap=self.computer.snapshot();snap['agents']['a1']['activity']='[red]raw <script> text\x1b[2J'
        text=''.join(t for _,t in fragments(snap,120,21))
        self.assertIn('[red]raw <script>',text);self.assertNotIn('\x1b',text)

    def test_bridge_on_off_and_help_work_without_api_calls(self):
        ui=FakeUI();bridge=ComputerBridge(ui)
        bridge.on(str(self.root));self.assertTrue(bridge.enabled)
        self.assertFalse(bridge.running);self.assertTrue(bridge.dashboard())
        bridge.command('help');self.assertIn('NOT a VM',ui.events[-1])
        bridge.off();self.assertFalse(bridge.enabled)
        self.assertEqual(bridge.dashboard(),[])

    def test_bridge_rejects_start_without_on(self):
        from fullagent.computer.state import ComputerError
        bridge=ComputerBridge(FakeUI())
        with self.assertRaises(ComputerError):bridge.submit('project')

    def test_real_tui_source_has_dispatch_layout_and_cancel_hooks(self):
        source=(ROOT/'fullagent/tui.py').read_text()
        ast.parse(source)
        for fragment in ('("/on",','("/off",','("/computer",','self.computer_mode.submit(text)',
                         'FormattedTextControl(self.computer_mode.dashboard)','self.computer_mode.cancel()',
                         'self._computer_approval_kind == "command"'):
            self.assertIn(fragment,source)

    def test_actual_approval_handler_never_enables_global_autoapprove(self):
        import threading
        tree=ast.parse((ROOT/'fullagent/tui.py').read_text())
        ui_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UI')
        method=next(n for n in ui_class.body if isinstance(n,ast.FunctionDef) and n.name=='_answer_approve')
        module=ast.Module(body=[method],type_ignores=[])
        namespace={}
        exec(compile(ast.fix_missing_locations(module),'actual-ui-approval-method','exec'),namespace)
        handler=namespace['_answer_approve']
        for kind,expected in (('command','always'),('plan',True)):
            ui=FakeUI();done=threading.Event()
            ui._approve_request=(None,{},done);ui._computer_approval_kind=kind
            handler(ui,'a')
            self.assertEqual(ui._approve_result,expected)
            self.assertTrue(done.is_set())
            self.assertFalse(ui.cfg.auto_approve)
        ui._approve_request=None
        handler(ui,'y')  # late keypress after cancellation is a safe no-op

    @unittest.skipUnless(importlib.util.find_spec('prompt_toolkit') and importlib.util.find_spec('rich'),
                         'Full TTY dependencies unavailable; pure dashboard and bridge tests still run')
    def test_actual_prompt_toolkit_ui_constructs_and_handles_on(self):
        from unittest.mock import patch
        from fullagent.agent import Agent
        from fullagent.config import Config
        from fullagent.tui import UI
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.application.current import create_app_session
        with patch('fullagent.client.prewarm_connection'),create_app_session(input=DummyInput(),output=DummyOutput()):
            ui=UI(Config(),Agent(Config()))
            ui._route_slash('/on',str(self.root))
            self.assertTrue(ui.computer_mode.enabled)
            self.assertTrue(ui.computer_mode.dashboard())
            ui._route_slash('/off','')
            self.assertFalse(ui.computer_mode.enabled)


class EntryPointTests(unittest.TestCase):
    def call(self,*args):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ,FULLAGENT_HOME=td)
            return subprocess.run([sys.executable,str(ROOT/'main.py'),*args],capture_output=True,text=True,timeout=12,env=env)

    def test_cli_help_never_loads_ui_or_model(self):
        r=self.call('computer','--help');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('Eight-agent',r.stdout)

    def test_version_and_top_level_help(self):
        import fullagent
        r=self.call('--version');self.assertIn(fullagent.__version__,r.stdout)
        r=self.call('--help');self.assertEqual(r.returncode,0);self.assertIn('computer --help',r.stdout)

    def test_doctor_reports_capability_without_model_calls(self):
        r=self.call('computer','doctor');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('No model/network benchmark',r.stdout)

    def test_headless_invalid_arguments_fail_closed(self):
        r=self.call('computer','run','--tokens','-1','--goal','test')
        self.assertEqual(r.returncode,2)
