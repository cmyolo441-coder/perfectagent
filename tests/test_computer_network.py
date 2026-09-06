from __future__ import annotations
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fullagent.config import Provider, Model, Effort
from fullagent.computer.state import Board, Settings, ComputerError
from fullagent.computer.research import Research, validate_public_url, TextExtractor, PublicHTTP
from fullagent.computer.transport import APIClient, TransportError


class HTTPFixture:
    """Local deterministic HTTP fixture. Not an external model/source test."""
    def __init__(self, responses):
        self.responses=list(responses);self.received=[]
    def __enter__(self):
        fixture=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                fixture.received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                status,ctype,body=fixture.responses.pop(0)
                self.send_response(status);self.send_header('Content-Type',ctype)
                self.send_header('Content-Length',str(len(body)));self.end_headers()
                try:self.wfile.write(body);self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/v1'
        return self
    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(2)


def client(url):
    return APIClient(Provider('fixture','Local test fixture',url,'','#fff'),
                     Model('fixture-model','fixture','Fixture',supports_tools=True,context_window=16384),
                     Effort('low','Low','#fff',1024,0.0,None,'test effort'),timeout=5)


def sse(events):
    return ''.join('data: '+(e if isinstance(e,str) else json.dumps(e))+'\n\n' for e in events).encode()


class TransportTests(unittest.TestCase):
    messages=[{'role':'system','content':'Test only'},{'role':'user','content':'hello'}]
    def test_real_http_json_response_and_usage(self):
        body=json.dumps({'choices':[{'message':{'content':'hello','reasoning_content':'PRIVATE_REASONING'},'finish_reason':'stop'}],
                         'usage':{'prompt_tokens':12,'completion_tokens':3}}).encode()
        with HTTPFixture([(200,'application/json',body)]) as server:
            c=client(server.url);self.addCleanup(c.close)
            r=c.chat(self.messages,[],1024,lambda:None)
            self.assertEqual(r.content,'hello');self.assertEqual(r.reasoning,'')
            self.assertEqual(r.usage['prompt_tokens'],12)
            self.assertLessEqual(server.received[0]['max_tokens'],1024)

    def test_real_sse_accumulates_tools_and_drops_reasoning(self):
        events=[{'choices':[{'delta':{'reasoning_content':'PRIVATE_REASONING','content':'working'}}]},
                {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'c1','function':{'name':'read_file','arguments':'{"path":'}}]}}]},
                {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"x.py"}'}}]},'finish_reason':'tool_calls'}]},
                {'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':8}},'[DONE]']
        with HTTPFixture([(200,'text/event-stream',sse(events))]) as server:
            c=client(server.url);self.addCleanup(c.close);events_seen=[]
            r=c.chat(self.messages,[],1024,lambda:None,lambda text,n:events_seen.append(text))
            self.assertEqual(r.reasoning,'');self.assertEqual(r.content,'working')
            self.assertEqual(json.loads(r.tool_calls[0]['function']['arguments']),{'path':'x.py'})
            self.assertEqual(r.usage['completion_tokens'],8)
            self.assertTrue(events_seen)

    def test_incomplete_stream_never_executes_partial_tool_call(self):
        events=[{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'x','function':{'name':'write_file','arguments':'{"path":'}}]}}]}]
        with HTTPFixture([(200,'text/event-stream',sse(events))]) as server:
            c=client(server.url);self.addCleanup(c.close)
            with self.assertRaises(TransportError):c.chat(self.messages,[],1024,lambda:None)

    def test_rate_limit_is_retryable_and_not_hidden(self):
        with HTTPFixture([(429,'application/json',b'{"error":"rate limit"}')]) as server:
            c=client(server.url);self.addCleanup(c.close)
            with self.assertRaises(TransportError) as caught:c.chat(self.messages,[],1024,lambda:None)
            self.assertTrue(caught.exception.retryable)
            self.assertIn('429',str(caught.exception))
            self.assertEqual(len(server.received),1)

    def test_stream_option_fallback_requires_a_separate_attempt(self):
        response=json.dumps({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}]}).encode()
        with HTTPFixture([(400,'application/json',b'{"error":"unsupported stream_options"}'),(200,'application/json',response)]) as server:
            c=client(server.url);self.addCleanup(c.close)
            with self.assertRaises(TransportError):c.chat(self.messages,[],1024,lambda:None)
            r=c.chat(self.messages,[],1024,lambda:None)
            self.assertEqual(r.content,'ok')
            self.assertIn('stream_options',server.received[0]);self.assertNotIn('stream_options',server.received[1])

    def test_bounded_response_buffer_rejects_oversized_data(self):
        with HTTPFixture([(200,'application/json',b'x'*200000)]) as server:
            c=client(server.url);self.addCleanup(c.close)
            with self.assertRaises(TransportError):c.chat(self.messages,[],512,lambda:None)

    def test_empty_remote_key_and_toolless_models_fail_closed(self):
        p=Provider('remote','Remote','https://example.com/v1','','#fff')
        m=Model('m','remote','M')
        e=Effort('low','Low','#fff',512,0,None,'')
        with self.assertRaises(ComputerError):APIClient(p,m,e)
        with self.assertRaises(ComputerError):APIClient(Provider('l','L','http://localhost','','#fff'),Model('m','l','M',supports_tools=False),e)


class FakeSources:
    def __init__(self):self.calls=[]
    def get(self,url,params=None):
        self.calls.append((url,params))
        if 'api.github' in url:
            return json.dumps({'items':[{'full_name':'example/project','html_url':'https://github.com/example/project','description':'fixture repository'}]}),url,'application/json'
        if 'gitlab.com/api' in url:
            return json.dumps([{'path_with_namespace':'example/repo','web_url':'https://gitlab.com/example/repo','description':'fixture'}]),url,'application/json'
        if 'registry.npmjs' in url:
            return json.dumps({'objects':[{'package':{'name':'fixture-pkg','version':'1.0.0','description':'fixture package'}}]}),url,'application/json'
        if 'wikipedia' in url:
            return json.dumps({'query':{'search':[{'title':'Software testing','snippet':'<b>Test</b> fixture'}]}}),url,'application/json'
        if 'googleapis' in url:
            return json.dumps({'items':[{'title':'Fixture result','link':'https://example.org/fixture','snippet':'fixture'}]}),url,'application/json'
        if 'duckduckgo' in url:
            return '<a class="result__a" href="https://example.org/doc">Fixture document</a><a class="result__snippet">fixture excerpt</a>',url,'text/html'
        return '<h1>Fixture content</h1><script>do not show</script><p>facts</p>',url,'text/html'


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.board=Board(root,root/'state','research fixtures',Settings())
        self.http=FakeSources();self.research=Research(self.board,lambda:None,self.http)

    def test_all_six_real_adapter_shapes_have_provenance(self):
        with patch.dict(os.environ,{'GOOGLE_CSE_API_KEY':'test-only-key','GOOGLE_CSE_ID':'test-engine'}):
            r=self.research.search('software testing',['web','github','gitlab','npm','wiki','google'])
        self.assertEqual(r['errors'],{});self.assertEqual(len(r['results']),6)
        self.assertTrue(all(h.get('retrieved_at') and h.get('url') for h in r['results']))
        self.assertEqual(len(self.board.data['sources']),6)
        self.assertNotIn('test-only-key',json.dumps(self.board.snapshot()))

    def test_search_cache_avoids_repeating_network_requests(self):
        self.research.search('testing',['github']);self.research.search('testing',['github'])
        self.assertEqual(len(self.http.calls),1)

    def test_google_missing_configuration_is_explicit(self):
        with patch.dict(os.environ,{'GOOGLE_CSE_API_KEY':'','GOOGLE_CSE_ID':''}):
            r=self.research.search('testing',['google'])
        self.assertFalse(r['results']);self.assertIn('google',r['errors'])
        self.assertEqual(len(self.http.calls),0)

    def test_network_disabled_never_calls_source_adapter(self):
        self.board.settings=Settings(network=False)
        r=self.research.search('testing',['github'])
        self.assertFalse(r['results']);self.assertIn('network',r['errors'])
        self.assertFalse(self.http.calls)

    def test_source_failure_is_not_replaced_by_invented_results(self):
        def failure(*a):raise ComputerError('offline fixture')
        self.http.get=failure
        r=self.research.search('testing',['github'])
        self.assertFalse(r['results']);self.assertIn('offline fixture',r['errors']['github'])
        self.assertFalse(self.board.data['sources'])

    def test_fetch_records_hash_and_untrusted_source_notice(self):
        r=self.research.fetch('https://example.org/doc')
        self.assertEqual(len(r['sha256']),64);self.assertIn('UNTRUSTED',r['notice'])
        self.assertNotIn('do not show',r['excerpt'])
        self.assertEqual(self.board.data['sources'][0]['kind'],'page')

    def test_local_metadata_reserved_and_mixed_dns_hosts_blocked(self):
        for addresses in (['127.0.0.1'],['169.254.169.254'],['10.2.3.4'],['::1'],['93.184.216.34','127.0.0.1']):
            values=[(2,1,6,'',(ip,443)) for ip in addresses]
            with self.subTest(addresses=addresses),patch('socket.getaddrinfo',return_value=values):
                with self.assertRaises(ComputerError):validate_public_url('https://example.org/path')

    def test_public_https_url_allowed_and_userinfo_blocked(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443))]):
            self.assertEqual(validate_public_url('https://example.org/doc#part'),'https://example.org/doc')
            with self.assertRaises(ComputerError):validate_public_url('https://user:pass@example.org/doc')
            with self.assertRaises(ComputerError):validate_public_url('https://example.org:9000/doc')

    def test_secret_queries_and_terminal_escapes_blocked(self):
        for text in ('API key sk-'+'x'*30,'hello\x1b[2Jworld','x\ny'):
            with self.subTest(text=text):
                with self.assertRaises(ComputerError):self.research.search(text,['github'])
