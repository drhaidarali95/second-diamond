import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import company

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=company.database(Path(self.tmp.name)/'state.db')
        self.config,self.agents=company.load()
    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()
    def test_roster(self):
        self.assertEqual(len(self.agents),25)
        self.assertEqual(self.agents['keen']['reports_to'],'founder')
    def test_demo_persistence(self):
        rid,outputs=company.run(self.db,self.config,self.agents,'Validate company workflow')
        self.assertEqual([o['owner'] for o in outputs],self.config['pilot_order'])
        self.assertTrue(all('DEMO FIXTURE' in o['output'] for o in outputs))
        self.assertEqual(self.db.execute('SELECT status FROM runs WHERE id=?',(rid,)).fetchone()[0],'awaiting_founder_review')
        self.assertEqual(self.db.execute("SELECT count(*) FROM tasks WHERE status='completed'").fetchone()[0],5)
    def test_missing_credentials_creates_no_run(self):
        with self.assertRaises(ValueError): company.run(self.db,self.config,self.agents,'Goal',live=True)
        self.assertEqual(self.db.execute('SELECT count(*) FROM runs').fetchone()[0],0)
    def test_live_adapter_handoffs_offline(self):
        calls=[]
        def provider(agent,context,*args):
            calls.append((agent['id'],context))
            return 'Proposed output from '+agent['id']
        company.run(self.db,self.config,self.agents,'Goal',True,'test-model','not-a-real-key',provider)
        self.assertEqual(len(calls),5)
        self.assertIn('Proposed output from nova',calls[1][1])
    def test_provider_failure_blocks_downstream(self):
        def fail(*args): raise RuntimeError('simulated provider failure')
        with self.assertRaises(RuntimeError):
            company.run(self.db,self.config,self.agents,'Goal',True,'test-model','not-a-real-key',fail)
        self.assertEqual(self.db.execute("SELECT count(*) FROM tasks WHERE status='blocked'").fetchone()[0],4)
        self.assertEqual(self.db.execute('SELECT status FROM runs').fetchone()[0],'failed')

if __name__=='__main__': unittest.main()
