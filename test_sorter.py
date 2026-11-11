import csv
import json
import tempfile
import unittest
from pathlib import Path
from sorter import validate, safety_match, process, demo_classifier

VALID = {'category': 'taste', 'sentiment': 'negative', 'urgent': False, 'summary': 'Too salty'}

class SorterTests(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(validate(json.dumps(VALID))[1])
    def test_missing_extra_wrong_fields(self):
        for item in [{**VALID, 'extra': 1}, {k:v for k,v in VALID.items() if k!='urgent'}, {**VALID, 'mood':'bad'}]:
            self.assertIsNotNone(validate(item)[1])
    def test_bad_values_and_types(self):
        for item in [{**VALID, 'category':'service'}, {**VALID, 'sentiment':'angry'},
                     {**VALID, 'urgent':'true'}, {**VALID, 'summary':' '.join(['word']*13)}]:
            self.assertIsNotNone(validate(item)[1])
    def test_safety_multilingual(self):
        for text in ['insect in sambar', 'hair in rice', 'poochi in rice', 'mudi in rice',
                     'சாம்பாரில் பூச்சி', 'expired milk', 'fell sick']:
            self.assertTrue(safety_match(text), text)
        self.assertFalse(safety_match('very salty and angry'))
    def test_retry_override_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td)/'feedback.csv', Path(td)/'output'
            with src.open('w', newline='', encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['id','date','meal','comment'])
                w.writerows([['1','2026-10-01','Lunch','insect in rice'],
                             ['2','2026-10-01','Lunch','too salty'],
                             ['3','2026-10-01','Lunch',' '],
                             ['4','2026-10-01','Lunch','mudi in curry']])
            calls={}
            def fake(comment, model):
                calls[comment]=calls.get(comment,0)+1
                if comment=='too salty' and calls[comment]==1: return '{bad'
                if comment=='mudi in curry': return '{bad'
                return json.dumps(VALID)
            self.assertEqual(process(src,out,fake,'test'),(4,2,2))
            with (out/'classified_feedback.csv').open(encoding='utf-8-sig') as f:
                data=list(csv.DictReader(f))
            self.assertEqual(data[0]['urgent'],'true')
            self.assertEqual(calls['too salty'],2)
            self.assertEqual(data[2]['status'],'needs_manual_review')
            self.assertEqual(data[3]['urgent'],'true')
            self.assertEqual(calls['mudi in curry'],2)
            self.assertNotIn(' ',calls)
    def test_sample_demo(self):
        root=Path(__file__).parent
        with tempfile.TemporaryDirectory() as td:
            total, urgent, review=process(root/'feedback.csv',Path(td),demo_classifier,'demo')
            self.assertEqual(total,24)
            self.assertGreaterEqual(urgent,7)
            self.assertEqual(review,1)
            self.assertEqual(len(list(Path(td).iterdir())),3)

if __name__=='__main__': unittest.main()
