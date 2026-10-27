from copy import deepcopy
import contextlib
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import build_book_srl as generator
from book_srl_corpus import CorpusLibrary, CORPUS_NAME, synchronize


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = CorpusLibrary(generator.DEFAULT_LIBRARY, generator.DEFAULT_BOOK)

    def test_source_inventory_keeps_tables_mixed_prose_tasks_and_solutions(self):
        records = self.library.records
        self.assertEqual({r['unit'] for r in records}, set(range(1,85)))
        self.assertEqual({r['section'] for r in records}, {'theory','task','solution'})
        self.assertGreater(len(records), 9000)
        texts = {r['text'] for r in records}
        self.assertTrue('I have an apple.' in texts)
        self.assertTrue('The dress is made of very good silk. I like hot coffee.' in texts)
        self.assertTrue(any('She likes interviewing people.' in text for text in texts))
        self.assertFalse(any('<br>' in text for text in texts))
        self.assertTrue(any('.......' in text for text in texts))
        lines = generator.DEFAULT_BOOK.read_text(encoding='utf-8').splitlines()
        for r in records:
            self.assertIn(r['raw'], lines[r['line']-1])
            self.assertTrue(lines[r['line']-1][r['column']-1:].startswith(r['raw']))
            self.assertNotEqual(r['source'], 'answer-keys')

    def test_printed_line_wraps_preserve_whole_sentences_and_original_locations(self):
        examples = {e.text for e in self.library.generate(8)}
        self.assertIn("I'm going to skate a lot in winter.", examples)
        self.assertIn('It was a cold morning. I met him on a hot summer day.', examples)
        self.assertIn('The dinner she made was tasty.', examples)
        wrapped = next(r for r in self.library.by_unit[8] if r['text']=='The dinner she made was tasty.')
        self.assertIn('<br>', wrapped['raw'])
        self.assertNotIn('<br>', wrapped['text'])

    def test_reviewed_units_cover_every_passage_and_link_completed_tasks(self):
        self.library.require_complete(range(1,13))
        for unit in range(1,13):
            count = self.library.coverage([unit])[unit]
            self.assertEqual(count['checked'],count['passages'])
            self.assertEqual(count['pending'],0)
            self.assertGreater(count['sentences'],40)
            for r in self.library.by_unit[unit]:
                if r['review']['kind']=='exercise_gap':
                    for target in r['review']['solutions']:
                        self.assertEqual(self.library.by_id[target]['review']['state'],'checked')
                        self.assertEqual(self.library.by_id[target]['item'],r['item'])

    def test_incomplete_book_fails_before_replacing_output(self):
        library = deepcopy(self.library)
        library.by_unit[1][0]['review'] = {'state':'pending'}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = out/'train.txt'
            original.write_text('keep existing data',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                generator.build(library,out,[1])
            self.assertEqual(original.read_text(encoding='utf-8'),'keep existing data')
            self.assertFalse((out/'valid.txt').exists())

    def test_pending_passages_have_no_invented_labels(self):
        pending = [r for r in self.library.records if r['review']['state']=='pending']
        for record in pending:
            self.assertEqual(record['review'],{'state':'pending'})
        record = deepcopy(self.library.records[0])
        record['review'] = {'state':'pending'}
        record['review']['variants'] = []
        with self.assertRaisesRegex(ValueError,'pending text'):
            self.library.validate_review(record)

    def test_original_nested_predicates_are_manually_scoped(self):
        example = next(e for e in self.library.generate(1)
                       if e.text=='I want to give him a book as a present.')
        want,give = example.sentences[0]['predicates']
        self.assertEqual(want['roles']['action'],'want')
        self.assertEqual((give['roles']['action'],give['roles']['object'],give['roles']['recipient']),('give','book','him'))
        self.assertEqual((give['form'],give['status'],give['parent']),('infinitive','intended',1))
        relative = next(e for e in self.library.generate(1)
                        if e.text=='The text he is translating is rather difficult.')
        state,process = relative.sentences[0]['predicates']
        self.assertEqual(state['roles']['subject'],'text')
        self.assertEqual((process['roles']['actor'],process['roles']['object'],process['aspect']),('he','text','continuous'))

    def test_ellipsis_and_vocative_do_not_invent_finite_predicates(self):
        example = next(e for e in self.library.generate(3) if e.text.startswith('Waiter!'))
        self.assertEqual(example.sentences[0]['predicates'],[])
        self.assertEqual(example.sentences[0]['features']['clause_type'],'vocative')
        generator.validate_story(generator.render(example))
        example = next(e for e in self.library.generate(2) if e.text.startswith('What do you call'))
        self.assertEqual(example.sentences[1]['predicates'],[])
        self.assertEqual(example.sentences[1]['features']['answer'],'a ceiling')

    def test_manual_record_must_cover_the_entire_source_passage(self):
        record = deepcopy(next(r for r in self.library.by_unit[1]
                               if len(r['review'].get('variants',[[]])[0])>1))
        record['review']['variants'][0].pop()
        with self.assertRaisesRegex(ValueError,'entire source passage'):
            self.library.validate_review(record)

    def test_wrong_exercise_solution_link_is_rejected(self):
        record = deepcopy(next(r for r in self.library.by_unit[1]
                               if r['review']['kind']=='exercise_gap'))
        target = next(r for r in self.library.by_unit[1] if r['section']=='solution'
                      and r['item'] != record['item'])
        record['review']['solutions'] = [target['id']]
        with self.assertRaisesRegex(ValueError,'different exercise item'):
            self.library.validate_review(record)

    def test_mixed_passages_preserve_sentences_after_word_lists_and_gap_prompts(self):
        examples = {e.text:e for e in self.library.generate(4)}
        self.assertIn('He belongs to the nobility. He is afraid of the police.', examples)
        self.assertIn('Do you think the English are fond of their weather?', examples)
        self.assertIn('The lion is the king of all animals.', examples)
        self.assertFalse(any('etc.' in text or '..........' in text for text in examples))
        multi = {e.text for e in self.library.generate(4, 'multi')}
        self.assertIn('He belongs to the nobility. He is afraid of the police.', multi)
        self.assertNotIn('The lion is the king of all animals.', multi)

    def test_mixed_review_cannot_drop_source_text_or_sentence_annotations(self):
        original = next(r for r in self.library.by_unit[4] if r['review']['kind']=='mixed')
        record = deepcopy(original)
        record['review']['parts'][0]['text'] = 'shortened word list '
        with self.assertRaisesRegex(ValueError, 'entire source passage'):
            self.library.validate_review(record)
        record = deepcopy(original)
        record['review']['parts'][1]['sentences'].pop()
        with self.assertRaisesRegex(ValueError, 'each sentence part'):
            self.library.validate_review(record)
        record = deepcopy(original)
        record['review']['parts'][0].pop('rule')
        with self.assertRaisesRegex(ValueError, 'fragment needs a reason'):
            self.library.validate_review(record)

    def test_short_modal_answer_uses_its_question_as_context(self):
        example = next(e for e in self.library.generate(4) if e.text.startswith('Which of you'))
        answer = example.sentences[1]['predicates'][0]
        self.assertEqual(answer['text'], 'can')
        self.assertEqual(answer['roles']['action'], 'play')
        self.assertEqual(answer['roles']['object'], 'horn')
        self.assertEqual(answer['context'], [1])
        generator.validate_story(generator.render(example))

    def test_narrative_quotations_resolve_the_addressee_in_each_dialogue(self):
        example = next(e for e in self.library.generate(10)
                       if e.text.startswith('I remember it very well.'))
        self.assertEqual(len(example.sentences),28)
        sale_question = example.sentences[16]['predicates']
        self.assertEqual(sale_question[0]['roles']['actor'],'General Miles')
        self.assertEqual(sale_question[1]['roles']['subject'],'you')
        self.assertEqual(sale_question[1]['roles']['actor'],'user')
        self.assertEqual(sale_question[1]['status'],'questioned')
        search_question = example.sentences[23]['predicates']
        self.assertEqual(search_question[1]['roles']['actor'],'old man')
        self.assertEqual(search_question[1]['context'],[22])
        self.assertNotIn('(After W. S. Maugham)',example.text)
        generator.validate_story(generator.render(example))

    def test_passive_and_perfect_in_narratives_keep_their_roles(self):
        example = next(e for e in self.library.generate(10)
                       if e.text.startswith('Opposite the Houses of Parliament'))
        crowned,buried = example.sentences[1]['predicates']
        for event in (crowned,buried):
            self.assertEqual(event['voice'],'passive')
            self.assertEqual(event['aspect'],'perfect')
            self.assertEqual(event['roles']['actor'],'unknown')
            self.assertEqual(event['roles']['object'],'British monarchs')
            self.assertEqual(event['roles']['location'],'Westminster Abbey')
            self.assertEqual(event['context'],[1])

    def test_pluralization_preserves_both_source_and_solution_in_one_family(self):
        examples = {e.text:e for e in self.library.generate(11)}
        original,plural = examples['I saw a cat playing with a mouse.'], examples['I saw cats playing with mice.']
        self.assertEqual(original.family,plural.family)
        play = original.sentences[0]['predicates'][1]
        self.assertEqual(play['roles']['actor'],'cat')
        self.assertEqual(play['roles']['companion'],'mouse')
        self.assertEqual(play['parent'],1)
        play = plural.sentences[0]['predicates'][1]
        self.assertEqual(play['roles']['actor'],'cats')
        self.assertEqual(play['roles']['companion'],'mice')
        self.assertNotIn('those shelves',examples)

    def test_elliptical_meal_request_does_not_invent_a_spoken_verb(self):
        example = next(e for e in self.library.generate(12)
                       if e.text=='Two coffees and a tea, please.')
        self.assertEqual(example.sentences[0]['predicates'],[])
        self.assertEqual(example.sentences[0]['features']['clause_type'],'ellipsis')
        generator.validate_story(generator.render(example))

    def test_complete_questions_with_answer_spaces_are_annotated_and_grouped_with_answers(self):
        cases = [(5, 'Where is Belgium situated?', 'Belgium is situated in Europe.'),
                 (6, 'Where do trains arrive?', 'Trains arrive at Victoria Station.')]
        for unit, question, answer in cases:
            examples = {e.text:e for e in self.library.generate(unit)}
            self.assertIn(question, examples)
            self.assertIn(answer, examples)
            self.assertEqual(examples[question].family, examples[answer].family)
            self.assertEqual(examples[question].sentences[0]['predicates'][0]['status'], 'questioned')

    def test_mixed_annotations_support_trailing_lists_and_parenthetical_sentences(self):
        examples = {e.text:e for e in self.library.generate(5)}
        self.assertIn('He lives in Leeds. He settled down in Colorado.', examples)
        self.assertTrue(any(text.startswith('Do you know Doctor Brown?') for text in examples))
        self.assertFalse(any('the painter Turner' in text for text in examples))
        example = next(e for e in self.library.generate(6) if 'are not proper names' in e.text)
        self.assertEqual(example.text, '(“white” and “royal” are not proper names).')
        self.assertEqual(example.sentences[0]['predicates'][0]['polarity'], 'negative')

    def test_sync_preserves_review_without_labelling_new_content(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            book=folder/'book.md'
            original=generator.DEFAULT_BOOK.read_text(encoding='utf-8')
            book.write_text(original,encoding='utf-8')
            corpus=folder/CORPUS_NAME
            corpus.write_bytes((generator.DEFAULT_LIBRARY/CORPUS_NAME).read_bytes())
            # New lines change locations but not the IDs of existing passages.
            changed=original.replace("I'm reading an interesting book.\n", "I'm reading an interesting book.\n\nA new example has appeared.\n",1)
            book.write_text(changed,encoding='utf-8')
            result=synchronize(book,corpus)
            new=next(r for r in result['records'] if r['text']=='A new example has appeared.')
            self.assertEqual(new['review'],{'state':'pending'})
            old=next(r for r in result['records'] if r['text']=="I'm reading an interesting book.")
            self.assertEqual(old['review']['state'],'checked')
            CorpusLibrary(folder,book)
            before=corpus.read_bytes()
            book.write_text(changed.replace('I want to give him a book as a present.',
                                           'I want to give him a pen as a present.',1),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed or disappeared'):
                synchronize(book,corpus)
            self.assertEqual(corpus.read_bytes(),before)

    def test_manual_dataset_is_deterministic_and_has_no_duplicate_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            stats=generator.build(self.library,a,range(1,13),per_unit=True)
            self.assertEqual(stats,generator.build(self.library,b,range(1,13),per_unit=True))
            seen=set()
            sentences=0
            for split in ('train','valid','test'):
                first=(a/f'{split}.txt').read_bytes()
                self.assertEqual(first,(b/f'{split}.txt').read_bytes())
                for story in first.decode('utf-8').strip().split('\n\n'):
                    generator.validate_story(story)
                    text=story.splitlines()[0]
                    self.assertNotIn(text,seen)
                    seen.add(text)
                    sentences += sum(bool(re.match(r'^\d+ sentence \d+\?\t',line))
                                     for line in story.splitlines())
            self.assertGreater(stats['unique_examples'],150)
            self.assertEqual(stats['sentences'],sentences)

    def test_coverage_and_validation_are_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'output'
            for flag in ('--coverage','--validate-only'):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(generator.main([flag,'--units','1-3','--out',str(out)]),0)
                self.assertTrue(output.getvalue())
                self.assertFalse(out.exists())

    def test_default_build_uses_all_ready_units_and_reports_incomplete_coverage(self):
        coverage = self.library.coverage()
        ready = [u for u,c in coverage.items() if c['pending']==0]
        pending = [u for u,c in coverage.items() if c['pending']]
        with tempfile.TemporaryDirectory() as folder:
            default, explicit = Path(folder)/'default', Path(folder)/'explicit'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(generator.main(['--out',str(default),'--per-unit']),0)
            self.assertIn(f'{len(ready)}/84 units fully reviewed',output.getvalue())
            self.assertIn(f'Book dataset: {len(ready)}/84 book units;',output.getvalue())
            self.assertIn('annotated sentences in those stories',output.getvalue())
            self.assertNotIn('template families',output.getvalue())
            self.assertIn(f'Selected units: {generator.format_units(ready)}',output.getvalue())
            if pending:
                self.assertIn(f'units {generator.format_units(pending)} still have',output.getvalue())
            generator.build(self.library,explicit,ready,per_unit=True)
            for path in default.rglob('*.txt'):
                self.assertEqual(path.read_bytes(),(explicit/path.relative_to(default)).read_bytes())
            self.assertEqual({p.name for p in (default/'units').glob('*.txt')},
                             {f'unit_{u:02d}.txt' for u in ready})

    def test_default_validation_reports_ready_selection_without_writing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'unused'
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(generator.main(['--validate-only','--out',str(out)]),0)
            self.assertIn('units fully reviewed',output.getvalue())
            self.assertIn('Validated',output.getvalue())
            self.assertFalse(out.exists())

    def test_explicit_subset_summary_uses_whole_book_denominator(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(generator.main(['--units','2','--validate-only']),0)
        self.assertIn('Validated book dataset: 1/84 book units;',output.getvalue())
        self.assertNotIn('1/1',output.getvalue())

    def test_explicit_incomplete_selection_still_fails_without_overwriting(self):
        library=deepcopy(self.library)
        library.by_unit[1][0]['review']={'state':'pending'}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            target=out/'train.txt'
            target.write_text('preserve existing training data',encoding='utf-8')
            with patch('book_srl_corpus.CorpusLibrary',return_value=library), \
                 contextlib.redirect_stderr(io.StringIO()) as error, \
                 self.assertRaises(SystemExit) as exit:
                generator.main(['--units','1-84','--out',str(out)])
            self.assertEqual(exit.exception.code,2)
            self.assertIn('Manual annotation is incomplete',error.getvalue())
            self.assertIn('--coverage',error.getvalue())
            self.assertNotIn('Pending by unit:',error.getvalue())
            self.assertEqual(target.read_text(encoding='utf-8'),'preserve existing training data')
            self.assertFalse((out/'valid.txt').exists())

    def test_default_selection_is_dynamic_and_can_have_gaps(self):
        library=deepcopy(self.library)
        for unit in library.units:
            if unit not in (1,3):
                library.by_unit[unit][0]['review']={'state':'pending'}
        with patch('book_srl_corpus.CorpusLibrary',return_value=library), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(generator.main(['--validate-only']),0)
        self.assertIn('Selected units: 1,3.',output.getvalue())
        self.assertIn('units 2,4-84 still have',output.getvalue())

    def test_no_ready_units_fails_without_creating_outputs(self):
        library=deepcopy(self.library)
        for unit in library.units:
            library.by_unit[unit][0]['review']={'state':'pending'}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'unused'
            with patch('book_srl_corpus.CorpusLibrary',return_value=library), \
                 contextlib.redirect_stderr(io.StringIO()) as error, \
                 self.assertRaises(SystemExit) as exit:
                generator.main(['--out',str(out)])
            self.assertEqual(exit.exception.code,2)
            self.assertIn('No book units are fully reviewed yet',error.getvalue())
            self.assertFalse(out.exists())

    def test_list_and_coverage_still_show_all_units_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'unused'
            for flag in ('--list','--coverage'):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(generator.main([flag,'--out',str(out)]),0)
                text=output.getvalue()
                self.assertIn('01:',text)
                self.assertIn('84:',text)
                self.assertNotIn('Selected units:',text)
                self.assertFalse(out.exists())


if __name__=='__main__':
    unittest.main()
