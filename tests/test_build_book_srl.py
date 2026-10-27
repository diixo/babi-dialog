from copy import deepcopy
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import book_srl_annotations as book


class BookSRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = book.Library()

    def example(self, unit, text):
        return next(e for e in self.library.generate(unit) if e.text == text)

    def test_every_lesson_has_annotated_generator_and_valid_examples(self):
        generators = self.library.generators()
        self.assertEqual(set(generators), set(range(1, 85)))
        for unit, generate in generators.items():
            with self.subTest(unit=unit):
                self.assertEqual(generate.__name__, f'generate_unit_{unit:02d}')
                self.assertIn(self.library.units[unit]['title'], generate.__doc__)
                examples = list(generate())
                self.assertTrue(examples)
                for example in examples:
                    self.assertEqual(example.unit, unit)
                    self.assertIn(f'unit-{example.source_unit}', self.library.anchors)
                    self.assertGreater(book.validate_story(book.render(example)), 0)

    def test_irregular_forms_and_article_agreement(self):
        self.example(1, 'Mary bought an apple. The apple is on the table.')
        self.example(11, 'The children are here.')
        self.example(11, 'The sheep are here.')
        past = self.example(44, 'Mary wrote a letter last week.')
        event = past.sentences[0]['predicates'][0]
        self.assertEqual((event['roles']['action'], event['tense']), ('write', 'past'))
        perfect = self.example(45, 'John has just written the book.')
        self.assertEqual(perfect.sentences[0]['predicates'][0]['aspect'], 'perfect')

    def test_verb_noun_compatibility_excludes_invalid_products(self):
        template = {'id': 'compatibility', 'slots': {
            'v': ['verb:read', 'verb:eat'], 'n': ['noun:book', 'noun:rice']},
            'compatible': [['v', 'n']]}
        combinations = {(b['v']['lemma'], b['n']['singular'])
                        for _, b in self.library.bindings_for(template)}
        self.assertEqual(combinations, {('read', 'book'), ('eat', 'rice')})

    def test_passive_subject_is_not_the_agent(self):
        example = self.example(60, 'The car was repaired by Mary.')
        event = example.sentences[0]['predicates'][0]
        self.assertEqual(event['roles'], {
            'subject': 'car', 'actor': 'mary', 'action': 'repair', 'object': 'car'})
        self.assertEqual(event['voice'], 'passive')
        omitted = self.example(60, 'The letter was written yesterday.')
        self.assertEqual(omitted.sentences[0]['predicates'][0]['roles']['actor'], 'unknown')
        recipient = self.example(61, 'Mary was given a book by John.')
        roles = recipient.sentences[0]['predicates'][0]['roles']
        self.assertEqual((roles['subject'], roles['actor'], roles['recipient']), ('mary','john','mary'))

    def test_want_keeps_intended_action_separate_from_main_predicate(self):
        example = self.example(77, 'Mary wants to read the book.')
        first, second = example.sentences[0]['predicates']
        self.assertEqual(first['roles']['action'], 'want')
        self.assertEqual((second['roles']['action'], second['roles']['actor']), ('read', 'mary'))
        self.assertEqual((second['form'], second['status']), ('infinitive', 'intended'))
        self.assertEqual((second['parent'], second['relation']), (1, 'complement'))
        controlled = self.example(78, 'Mary wants John to read the book.')
        self.assertEqual(controlled.sentences[0]['predicates'][1]['roles']['actor'], 'john')

    def test_reported_and_counterfactual_actions_are_not_asserted(self):
        reported = self.example(58, 'Mary said that John had written the letter.')
        first, second = reported.sentences[0]['predicates']
        self.assertEqual((first['status'], second['status']), ('asserted', 'reported'))
        wish = self.example(64, 'I wish I had read that book.')
        self.assertEqual(wish.sentences[0]['predicates'][1]['status'], 'counterfactual')

    def test_absence_of_obligation_does_not_negate_action(self):
        example = self.example(68, 'Mary does not have to work tomorrow.')
        event = example.sentences[0]['predicates'][0]
        self.assertEqual(event['polarity'], 'positive')
        self.assertEqual(event['roles']['negation_scope'], 'obligation')
        prohibition = self.example(68, 'Mary must not open this box.')
        self.assertEqual(prohibition.sentences[0]['predicates'][0]['polarity'], 'negative')
        unnecessary = self.example(72, 'Mary need not have bought bread.')
        self.assertEqual(unnecessary.sentences[0]['predicates'][0]['status'], 'asserted')

    def test_multi_sentence_anaphora_has_prior_sentence_evidence(self):
        example = self.example(16, 'Mary owns a car. She likes her car.')
        event = example.sentences[1]['predicates'][0]
        self.assertEqual((event['roles']['subject'], event['roles']['actor']), ('she', 'mary'))
        self.assertEqual(event['context'], [1])
        rows = book.render(example).strip().splitlines()
        prior = next(int(row.split(' ', 1)[0]) for row in rows if ' sentence 1?\t' in row)
        current = next(int(row.split(' ', 1)[0]) for row in rows if ' sentence 2?\t' in row)
        actor_rows = [row for row in rows if ' actor of predicate ' in row]
        refs = list(map(int, actor_rows[1].split('\t')[2].split()))
        self.assertIn(prior, refs)
        self.assertIn(current, refs)

    def test_ellipsis_recovers_predicate_from_context(self):
        example = self.example(84, 'Mary does not read novels. Neither do I.')
        event = example.sentences[1]['predicates'][0]
        self.assertEqual(event['text'], 'do')
        self.assertEqual(event['roles']['action'], 'read')
        self.assertEqual(event['roles']['actor'], 'user')
        self.assertEqual(event['roles']['object'], 'novels')
        self.assertEqual(event['polarity'], 'negative')
        self.assertEqual(event['context'], [1])

    def test_distinct_predicates_keep_their_own_times_and_actors(self):
        example = self.example(51, 'Mary was reading a book when John arrived.')
        first, second = example.sentences[0]['predicates']
        self.assertEqual((first['roles']['actor'], first['aspect']), ('mary', 'continuous'))
        self.assertEqual((second['roles']['actor'], second['aspect']), ('john', 'simple'))
        self.assertEqual(second['relation'], 'during')
        story = book.render(example)
        self.assertIn('actor of predicate 1 of sentence 1?\tmary\t', story)
        self.assertIn('actor of predicate 2 of sentence 1?\tjohn\t', story)
        self.assertIn('parent of predicate 2 of sentence 1?\tpredicate 1 of sentence 1\t', story)
        questions = [row.partition(' ')[2].split('\t')[0] for row in story.strip().splitlines()[1:]]
        self.assertEqual(len(questions), len(set(questions)))

    def test_review_inherits_templates_and_origins(self):
        original = {e.text: e for e in self.library.generate(1)}
        review = {e.text: e for e in self.library.generate(10)}
        for text, example in original.items():
            self.assertEqual(review[text].source_unit, 1)
            self.assertEqual(review[text].family, example.family)
            self.assertEqual(book.render(review[text]), book.render(example))

    def test_reproducible_disjoint_splits_hold_out_whole_template_families(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            stats = book.build(self.library, a, per_unit=True)
            self.assertEqual(stats, book.build(self.library, b, per_unit=True))
            for path in a.rglob('*.txt'):
                self.assertEqual(path.read_bytes(), (b/path.relative_to(a)).read_bytes())
            inputs = {}
            for split in ('train', 'valid', 'test'):
                stories = (a/f'{split}.txt').read_text(encoding='utf-8').strip().split('\n\n')
                self.assertTrue(stories)
                for story in stories:
                    text = story.splitlines()[0][8:]
                    self.assertNotIn(text, inputs)
                    inputs[text] = split
            families = {}
            for unit in self.library.units:
                for example in self.library.generate(unit):
                    families.setdefault(example.family, set()).add(inputs[example.text])
            self.assertTrue(all(len(splits) == 1 for splits in families.values()))
            self.assertEqual(len(list((a/'units').glob('unit_*.txt'))), 84)
            self.assertEqual(stats['unique_examples'], len(inputs))

    def test_mode_filtering_and_empty_filtered_units(self):
        for mode in ('single', 'multi'):
            for unit in self.library.units:
                for example in self.library.generate(unit, mode):
                    self.assertEqual(len(example.sentences) == 1, mode == 'single')
            stats = book.build(self.library, None, mode=mode)
            self.assertGreater(stats['unique_examples'], 0)
        self.assertEqual(book.build(self.library, None, [84], 'single')['units'], {84: 0})

    def test_invalid_library_does_not_overwrite_datasets(self):
        broken = deepcopy(self.library)
        broken.units[84]['templates'][0]['sentences'][0]['text'] = '{missing.field}.'
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for split in ('train', 'valid', 'test'):
                (out/f'{split}.txt').write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unknown placeholder'):
                book.build(broken, out)
            for path in out.iterdir():
                self.assertEqual(path.read_text(encoding='utf-8'), 'keep')

    def test_conflicting_duplicate_annotations_are_rejected(self):
        broken = deepcopy(self.library)
        duplicate = deepcopy(broken.units[1]['templates'][1])
        duplicate['id'] = 'conflicting_teacher'
        duplicate['sentences'][0]['predicates'][0]['roles']['actor'] = 'wrong'
        broken.units[1]['templates'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'Conflicting annotations'):
            book.build(broken, None, [1])

    def test_duplicate_inputs_join_families_in_one_split(self):
        library = deepcopy(self.library)
        duplicate = deepcopy(library.units[1]['templates'][1])
        duplicate['id'] = 'another_teacher_family'
        library.units[1]['templates'].append(duplicate)
        before = book.build(self.library, None, [1])
        after = book.build(library, None, [1])
        self.assertEqual(before['unique_examples'], after['unique_examples'])
        self.assertEqual(before['families'], after['families'])

    def test_bad_context_parent_predicate_and_evidence_are_rejected(self):
        example = self.example(16, 'Mary owns a car. She likes her car.')
        for field, value in [('context', [2]), ('parent', 1), ('text', 'invisible')]:
            sentences = deepcopy(example.sentences)
            sentences[1]['predicates'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                book.validate_example(replace(example, sentences=sentences))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            book.validate_story('1 User: Mary works.\n2 actor?\tmary\t3\n')

    def test_json_duplicates_and_format_expressions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'duplicate.json'
            path.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                book.read_json(path)
        for placeholder in ('{p.__class__}', '{p.name!r}', '{p.name:20}'):
            with self.subTest(placeholder=placeholder), self.assertRaises(ValueError):
                book.substitute(placeholder, {'p': {'name': 'Mary'}})

    def test_cli_lists_and_validates_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'unused'
            for flag in ('--list', '--validate-only'):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(book.main([flag, '--source', 'templates', '--units', '1,16,39-44', '--out', str(out)]), 0)
                self.assertTrue(output.getvalue())
                self.assertFalse(out.exists())
        self.assertEqual(book.parse_units('1,16,39-41'), [1,16,39,40,41])
        for value in ('0','85','3-1','1,','a','-2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                book.parse_units(value)


if __name__ == '__main__':
    unittest.main()
