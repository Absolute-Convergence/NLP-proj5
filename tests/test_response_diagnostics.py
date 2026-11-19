"""Content diagnostics neither guess missing answers nor hide formatting failures."""
import json
import unittest
from notebook_helpers import load_definitions


class ResponseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.functions=load_definitions('response_diagnostics',{'json':json})
        self.target={'recoverable':True,'operations':['parse_number_words']}

    def diagnose(self,text,target=None):
        return self.functions['diagnose_predictions']([text],[{'id':'case','target':target or self.target}])[0]

    def test_prose_and_fences_do_not_erase_complete_answer_content(self):
        answer=json.dumps(self.target)
        for text in (answer,'Explanation '+answer+' trailing text','```json\n'+answer+'\n```'):
            with self.subTest(text=text):
                row=self.diagnose(text)
                self.assertEqual(row['Recoverability content'],'correct')
                self.assertEqual(row['Tool-order content'],'correct')
                self.assertEqual(row['Full-plan content'],'correct')
                self.assertEqual(row['Strict format'],text==answer)
                self.assertEqual(row['Strict tool call'],text==answer)

    def test_incomplete_and_ambiguous_json_are_unscorable_without_field_salvage(self):
        texts=[None,'not JSON','{"recoverable":true,"operations":["parse_number_words"',
               '{"outer":'+json.dumps(self.target),json.dumps(self.target)+' '+json.dumps(self.target),
               '{"recoverable":false,"recoverable":true,"operations":[]}',
               '{"recoverable":true,"operations":NaN}']
        for text in texts:
            with self.subTest(text=text):
                row=self.diagnose(text)
                self.assertEqual(row['Recoverability content'],'unscorable')
                self.assertEqual(row['Tool-order content'],'unscorable')
                self.assertEqual(row['Full-plan content'],'unscorable')
                self.assertFalse(row['Strict format'])

    def test_wrong_content_unknown_tools_and_extra_fields_are_not_conflated(self):
        wrong=self.diagnose('```json\n{"recoverable":false,"operations":["unknown"]}\n```')
        self.assertEqual(wrong['Recoverability content'],'wrong')
        self.assertEqual(wrong['Tool-order content'],'wrong')
        self.assertEqual(wrong['Full-plan content'],'wrong')
        extra=self.diagnose(json.dumps({**self.target,'explanation':'extra field'}))
        self.assertEqual(extra['Full-plan content'],'correct')
        self.assertFalse(extra['Strict format'])
        self.assertFalse(extra['Strict tool call'])

    def test_readable_fields_are_scored_independently_without_type_coercion(self):
        row=self.diagnose('{"recoverable":true,"operations":"parse_number_words"}')
        self.assertEqual(row['Recoverability content'],'correct')
        self.assertEqual(row['Tool-order content'],'unscorable')
        self.assertEqual(row['Full-plan content'],'unscorable')
        row=self.diagnose('{"recoverable":"true","operations":["parse_number_words"]}')
        self.assertEqual(row['Recoverability content'],'unscorable')
        self.assertEqual(row['Tool-order content'],'correct')
        self.assertEqual(row['Full-plan content'],'unscorable')
        negative=self.diagnose('bad',{'recoverable':False,'operations':[]})
        self.assertEqual(negative['Tool-order content'],'not applicable')

    def test_mixed_readability_keeps_unknown_cases_separate_from_wrong_answers(self):
        records=[{'id':str(i),'target':self.target if i!=2 else {'recoverable':False,'operations':[]}}
                 for i in range(5)]
        incomplete='Explanation ```json {"recoverable":true,"operations":["parse_'
        complete='Explanation ```json {"recoverable":true,"operations":[]} ```'
        rows=self.functions['diagnose_predictions']([incomplete,complete,complete,incomplete,incomplete],records)
        self.assertEqual([sum(r['Recoverability content']==state for r in rows)
                          for state in ('correct','wrong','unscorable')],[1,1,3])
        self.assertEqual([sum(r['Full-plan content']==state for r in rows)
                          for state in ('correct','wrong','unscorable')],[0,2,3])
        self.assertEqual(sum(r['Strict format'] for r in rows),0)


if __name__=='__main__':
    unittest.main()
