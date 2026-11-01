from copy import deepcopy
import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book
import test_book_srl_corpus as corpus_tests


class NaturalQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=book.Library()
        cls.data=book.read_json(book.DEFAULT_LIBRARY/'questions.json')
        cls.questions=book.QuestionLibrary(book.DEFAULT_LIBRARY,cls.library,data=cls.data)

    def answers(self,text):
        return self.questions.by_text[' '.join(text.casefold().split())]

    def test_every_source_and_compatible_binding_has_an_explicit_disposition(self):
        records={r['sentence_id'] for r in self.data['questions']}
        excluded={r['sentence_id'] for r in self.data['excluded_sentences']}
        self.assertFalse(records & excluded)
        self.assertEqual(records | excluded,
                         {t['id'] for u in self.library.units.values() for t in u['templates']})
        for unit in self.library.units:
            for template,_,text,_,_ in self.library.variants(unit):
                if template['id'] in records:
                    self.assertTrue(self.answers(text),template['id'])

    def test_original_and_variant_questions_use_the_actual_subject_and_object(self):
        self.assertEqual(self.answers("I'm reading an interesting book.")['What are you reading?'],
                         'An interesting book.')
        self.assertEqual(self.answers("She's writing an interesting article.")['What is she writing?'],
                         'An interesting article.')
        self.assertNotIn('What are you reading?',self.answers("She's writing an interesting article."))

    def test_negation_intention_and_reported_questions_preserve_their_meaning(self):
        self.assertEqual(self.answers('Mark is not very practical.')['Is Mark very practical?'],'No.')
        pairs=self.answers('I want to give him a book as a present.')
        self.assertEqual(pairs['What do you want to give him?'],'A book.')
        self.assertNotIn('What did you give him?',pairs)
        pairs=self.answers('She asks, “Have you brought the book?”')
        self.assertEqual(pairs['What does she ask?'],'"Have you brought the book?"')
        self.assertNotIn('Have you brought the book?',pairs)
        self.assertNotIn('have you seen them?',self.questions.by_text)

    def test_identical_questions_in_one_context_cannot_have_conflicting_answers(self):
        data=deepcopy(self.data)
        data['questions'][0]['questions'].append(dict(data['questions'][0]['questions'][0],answer='Different.'))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            book.QuestionLibrary(book.DEFAULT_LIBRARY,self.library,data=data)

    def test_conditionals_negation_and_shared_comparisons_have_supported_answers(self):
        pairs=self.answers("If Tom weren't nervous he wouldn't have shouted at his child.")
        self.assertEqual(pairs['Why did Tom shout at his child?'], 'Because he was nervous.')
        self.assertNotIn('Were Tom nervous he would not have shouted at his child if?',pairs)
        pairs=self.answers("I don't know anything about gardening.")
        self.assertEqual(pairs['What do you know about gardening?'],'Nothing.')
        self.assertNotIn('What do you know?',pairs)
        pairs=self.answers('Yes, he is, but he is taller and seems much stronger than Paul.')
        self.assertEqual(pairs['Who is taller than Paul?'],'He is.')
        self.assertEqual(pairs['Who seems stronger than Paul?'],'He does.')

    def test_missing_sources_or_variants_are_not_silently_omitted(self):
        data=deepcopy(self.data)
        data['questions'].pop()
        with self.assertRaisesRegex(ValueError,'every source'):
            book.QuestionLibrary(book.DEFAULT_LIBRARY,self.library,data=data)
        data=deepcopy(self.data)
        record=next(r for r in data['questions'] if r.get('variants'))
        record['variants'].pop()
        with self.assertRaisesRegex(ValueError,'bindings'):
            book.QuestionLibrary(book.DEFAULT_LIBRARY,self.library,data=data)

    def test_changed_source_text_requires_review(self):
        data=deepcopy(self.data)
        data['questions'][0]['text']='A different statement.'
        with self.assertRaisesRegex(ValueError,'source text changed'):
            book.QuestionLibrary(book.DEFAULT_LIBRARY,self.library,data=data)

    def test_natural_output_has_plain_questions_and_original_text_evidence(self):
        text="I'm reading an interesting book."
        pairs=[dict(question=q,answer=a) for q,a in self.answers(text).items()]
        story=book.render_questions(text,pairs)
        self.assertEqual(book.validate_story(story),len(pairs))
        self.assertTrue(story.startswith('1 User: '+text+'\n'))
        self.assertIn('What are you reading?\tAn interesting book.\t1\n',story)
        self.assertNotIn('fragment?',story)
        self.assertNotIn('Assistant:',story)


class NaturalQuestionBuildTests(unittest.TestCase):
    def test_one_command_keeps_question_variants_in_the_srl_family_split(self):
        fixture=corpus_tests.UnifiedBuildTests()
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            fixture.fixture(directory)
            _,output=fixture.run_cli(directory)
            first={p.relative_to(directory/'out'):p.read_bytes()
                   for p in (directory/'out').rglob('*.txt')}
            fixture.run_cli(directory)
            self.assertEqual(first,{p.relative_to(directory/'out'):p.read_bytes()
                                    for p in (directory/'out').rglob('*.txt')})
            membership={}
            for split in ('train','valid','test'):
                for story in (directory/'out'/f'{split}.txt').read_text(encoding='utf-8').strip().split('\n\n'):
                    if story: membership[story.splitlines()[0]]=split
                path=directory/'out'/'questions'/f'{split}.txt'
                for story in path.read_text(encoding='utf-8').strip().split('\n\n'):
                    if not story: continue
                    self.assertEqual(membership[story.splitlines()[0]],split)
                    self.assertEqual(book.validate_story(story),2)
            self.assertEqual(membership['1 User: Mary reads.'],membership['1 User: Nora reads.'])
            self.assertIn('Natural questions: 84 source examples',output)

    def test_invalid_questions_do_not_overwrite_either_dataset(self):
        fixture=corpus_tests.UnifiedBuildTests()
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            fixture.fixture(directory)
            fixture.run_cli(directory)
            before={p.relative_to(directory/'out'):p.read_bytes()
                    for p in (directory/'out').rglob('*.txt')}
            data=book.read_json(directory/'questions.json')
            data['questions'][0]['questions']=[]
            fixture.write_json(directory/'questions.json',data)
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                fixture.run_cli(directory)
            self.assertEqual(before,{p.relative_to(directory/'out'):p.read_bytes()
                                     for p in (directory/'out').rglob('*.txt')})


if __name__=='__main__':
    unittest.main()
