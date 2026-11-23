"""Synthetic unit fixtures only; never research experiment results."""
import unittest,tempfile,csv,json,sys
from pathlib import Path
import attribute_pilot as p
from attribute_labels import merge

class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
        self.rows=[{'image_id':str(i),'split':'dev' if i<50 else 'eval','sha256':str(i),'color':'','sleeve_length':'','pattern':'','reviewer':'','notes':''} for i in range(150)]
        self.man=self.d/'manifest.csv';self.lab=self.d/'labels.csv';self.pred=self.d/'predictions.json';self.write(self.man,self.rows)
    def write(self,path,rows):
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(rows)
    def predictions(self,split='dev'):
        items=[]
        for r in self.rows:
            if r['split']!=split:continue
            attrs={}
            for g,values in p.VALUES.items():
                scores={v:(.5 if j==0 else .1) for j,v in enumerate(values)}
                attrs[g]={'prediction':values[0],'score_margin':.4,'cosine_scores':scores}
            items.append({'image_id':r['image_id'],'split':split,'attributes':attrs})
        doc={'protocol_sha256':p.protocol_digest(),'manifest_sha256':p.manifest_digest(self.rows),'script_sha256':p.digest(Path(p.__file__)),'predictions':items}
        self.pred.write_text(json.dumps(doc));return doc
    def test_empty_labels_not_zero_accuracy(self):
        self.write(self.lab,self.rows);self.predictions();out=self.d/'m.json';p.evaluate(self.lab,self.pred,out,'dev',self.man)
        r=json.loads(out.read_text());self.assertEqual(r['status'],'pending_labels');self.assertIsNone(r['attributes']['color']['accuracy'])
    def test_metrics_exclusions_and_absent_classes(self):
        self.rows[0].update(color='black',sleeve_length='sleeveless',pattern='plain',reviewer='fixture')
        self.rows[1].update(color='white',sleeve_length='uncertain',pattern='not_visible',reviewer='fixture')
        self.write(self.lab,self.rows);self.predictions();out=self.d/'m.json';p.evaluate(self.lab,self.pred,out,'dev',self.man)
        r=json.loads(out.read_text());a=r['attributes']['color'];self.assertEqual(a['accuracy'],.5);self.assertAlmostEqual(a['macro_f1'],1/3)
        self.assertEqual(a['per_class']['white']['support'],1);self.assertIsNone(a['per_class']['red']['recall']);self.assertEqual(r['attributes']['sleeve_length']['known'],1)
        self.assertEqual(r['attributes']['sleeve_length']['excluded']['uncertain'],1)
        with out.with_name('m_errors.csv').open(encoding='utf-8-sig') as f:self.assertEqual(len(list(csv.DictReader(f))),1)
    def test_invalid_score_and_missing_prediction_rejected(self):
        self.write(self.lab,self.rows);doc=self.predictions();doc['predictions'][0]['attributes']['color']['cosine_scores']['red']=float('nan');self.pred.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):p.evaluate(self.lab,self.pred,self.d/'m.json','dev',self.man)
        doc=self.predictions();doc['predictions'].pop();self.pred.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):p.evaluate(self.lab,self.pred,self.d/'m.json','dev',self.man)
    def test_manifest_and_reviewer_protection(self):
        self.rows[0]['color']='black';self.write(self.lab,self.rows);self.predictions()
        with self.assertRaises(ValueError):p.evaluate(self.lab,self.pred,self.d/'m.json','dev',self.man)
        self.rows[0]['reviewer']='fixture';self.rows[0]['split']='eval';self.write(self.lab,self.rows)
        with self.assertRaises(ValueError):p.evaluate(self.lab,self.pred,self.d/'m.json','dev',self.man)
    def test_eval_freeze_required_and_mutations_detected(self):
        self.write(self.lab,self.rows)
        with self.assertRaises(ValueError):p.freeze(self.man,self.lab,self.d/'lock.json','fixture')
        for r in self.rows[:50]:r.update(color='unknown',sleeve_length='uncertain',pattern='not_visible',reviewer='fixture')
        self.write(self.lab,self.rows);lock=self.d/'lock.json';p.freeze(self.man,self.lab,lock,'fixture');p.verify_lock(lock,self.rows)
        with self.assertRaises(ValueError):p.verify_lock(None,self.rows)
        self.rows[0]['sha256']='changed'
        with self.assertRaises(ValueError):p.verify_lock(lock,self.rows)
    def test_merge_and_disagreement_preservation(self):
        first=[dict(r) for r in self.rows[:50]];first[0].update(color='black',reviewer='A',notes='comma, newline\nquote"')
        second=[dict(r) for r in self.rows[50:]];second[0].update(color='white',reviewer='B')
        a=self.d/'a.csv';b=self.d/'b.csv';self.write(a,first);self.write(b,second);out=self.d/'merged.csv';merge(self.man,[a,b],out)
        r=p.read(out);self.assertEqual(r[0]['notes'],first[0]['notes']);self.assertEqual(r[50]['reviewer'],'B')
        first[0]['color']='white';self.write(b,first)
        with self.assertRaises(ValueError):merge(self.man,[a,b],self.d/'conflicting.csv')
        self.assertFalse((self.d/'conflicting.csv').exists());self.assertTrue((self.d/'conflicting.conflicts.json').exists())

if __name__=='__main__':unittest.main()
