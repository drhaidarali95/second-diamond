import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('boardroom',Path(__file__).resolve().parents[1]/'boardroom/server.py')
boardroom=importlib.util.module_from_spec(spec)
spec.loader.exec_module(boardroom)

class BoardroomTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.old=boardroom.DB
        boardroom.DB=Path(self.tmp.name)/'state.sqlite3'
    def tearDown(self):
        boardroom.DB=self.old
        self.tmp.cleanup()
    def test_reply_is_persisted_and_context_reused(self):
        calls=[]
        def infer(path,payload):
            calls.append(payload)
            return {'message':{'content':'Verified test reply'}}
        boardroom.converse('s1','atlas','First question','offline-test',infer)
        boardroom.converse('s1','atlas','Second question','offline-test',infer)
        self.assertEqual(calls[1]['messages'][1]['content'],'First question')
        with boardroom.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0],4)
    def test_invalid_agent_and_missing_model_fail(self):
        with self.assertRaises(ValueError): boardroom.converse('s1','unknown','Hello','test')
        with self.assertRaises(ValueError): boardroom.converse('s1','atlas','Hello','')
    def test_failed_inference_is_not_saved_as_reply(self):
        def fail(*args): raise RuntimeError('simulated inference failure')
        with self.assertRaises(RuntimeError): boardroom.converse('s1','atlas','Hello','test',fail)
        with boardroom.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0],0)
