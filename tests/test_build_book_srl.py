from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book


class BookSRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = book.AnnotationLibrary()

    def example(self, unit, text):
        return next(e for e in self.library.generate(unit) if e.text == text)

    def groups(self, example):
        groups = []
        for row in book.render_annotation(example).strip().splitlines()[1:]:
            number, _, qa = row.partition(' ')
            question, answer, evidence = qa.split('\t')
            if question == 'fragment?':
                groups.append({'line': int(number)})
            self.assertNotIn(question, groups[-1])
            groups[-1][question] = (answer, list(map(int, evidence.split())))
        return groups

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
                    self.assertGreater(book.validate_story(book.render_annotation(example)), 0)
                    groups = self.groups(example)
                    self.assertEqual(len(groups), sum(len(s['predicates']) for s in example.sentences))
                    cursor = 0
                    for group in groups:
                        fragment, refs = group['fragment?']
                        self.assertIn(fragment, example.text)
                        cursor = example.text.index(fragment, cursor) + len(fragment)
                        self.assertEqual(refs, [1])
                        self.assertIn(group['type?'][0], book.UTTERANCE_TYPES)
                        self.assertEqual(group['type?'][1], [1, group['line']])
                        self.assertTrue({'predicate?', 'verb?', 'action?'} <= group.keys())
                        for question, value in group.items():
                            if question == 'line':
                                continue
                            answer, refs = value
                            self.assertRegex(question, r'^[a-z][a-z ]*\?$')
                            self.assertNotIn(question, {'sentence?', 'sentence count?', 'predicate count?',
                                'subject?', 'tense?', 'aspect?', 'voice?', 'beneficiary?', 'status?',
                                'form?', 'parent?', 'relation?', 'article?', 'question type?'})
                            if question != 'fragment?':
                                self.assertIn(group['line'], refs)
                                self.assertTrue(set(refs) <= {1, group['line']})
                            if answer == 'none':
                                self.assertIn(question, {'predicate?', 'verb?'})

    def test_irregular_forms_and_article_agreement(self):
        self.example(1, 'Mary bought an apple. The apple is on the table.')
        self.example(11, 'The children are here.')
        self.example(11, 'The sheep are here.')
        past = self.example(44, 'Mary wrote a letter last week.')
        event = past.sentences[0]['predicates'][0]
        self.assertEqual((event['roles']['action'], event['tense']), ('write', 'past'))
        perfect = self.example(45, 'John has just written the book.')
        self.assertEqual(perfect.sentences[0]['predicates'][0]['aspect'], 'perfect')

    def test_new_named_people_keep_pronouns_bound_across_sentences(self):
        for name, pronoun, possessive in [('Daniel', 'He', 'his'), ('Emily', 'She', 'her')]:
            example = self.example(16, f'{name} owns a car. {pronoun} likes {possessive} car.')
            for sentence in example.sentences:
                self.assertEqual(sentence['predicates'][0]['roles']['person'], name)

    def test_annotation_status_is_optional_but_deprecated_values_are_rejected(self):
        example = next(e for unit in self.library.units for e in self.library.generate(unit)
                       if len(e.sentences) == 1 and len(e.sentences[0]['predicates']) == 1
                       and 'status' not in e.sentences[0]['predicates'][0])
        rendered = book.render_annotation(example)
        self.assertNotIn('status?', rendered)
        self.assertGreater(book.validate_story(rendered), 0)
        for field, value in [('status', 'questioned'), ('status', None), ('tense', None)]:
            sentences = deepcopy(example.sentences)
            sentences[0]['predicates'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'invalid '+field):
                book.validate_annotation_example(replace(example, sentences=sentences))

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
            'subject': 'car', 'actor': 'person', 'person': 'Mary', 'action': 'repair', 'object': 'car'})
        self.assertEqual(event['voice'], 'passive')
        omitted = self.example(60, 'The letter was written yesterday.')
        self.assertEqual(omitted.sentences[0]['predicates'][0]['roles']['actor'], 'unknown')
        recipient = self.example(61, 'Mary was given a book by John.')
        roles = recipient.sentences[0]['predicates'][0]['roles']
        self.assertEqual((roles['subject'], roles['person'], roles['recipient']), ('Mary','John','Mary'))

    def test_want_keeps_intended_action_separate_from_main_predicate(self):
        example = self.example(77, 'Mary wants to read the book.')
        first, second = example.sentences[0]['predicates']
        self.assertEqual(first['roles']['action'], 'want')
        self.assertEqual((second['roles']['action'], second['roles']['person']), ('read', 'Mary'))
        self.assertEqual((second['form'], second['status']), ('infinitive', 'intended'))
        self.assertEqual((second['parent'], second['relation']), (1, 'complement'))
        controlled = self.example(78, 'Mary wants John to read the book.')
        self.assertEqual(controlled.sentences[0]['predicates'][1]['roles']['person'], 'John')

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

    def test_multi_sentence_anaphora_has_full_input_and_fragment_evidence(self):
        example = self.example(16, 'Mary owns a car. She likes her car.')
        event = example.sentences[1]['predicates'][0]
        self.assertEqual((event['roles']['subject'], event['roles']['person']), ('she', 'Mary'))
        self.assertEqual(event['context'], [1])
        first, second = self.groups(example)
        self.assertEqual(first['fragment?'][0], 'Mary owns a car')
        self.assertEqual(second['fragment?'][0], 'She likes her car')
        self.assertEqual(first['person?'], ('Mary', [first['line']]))
        self.assertEqual(second['person?'], ('Mary', [1, second['line']]))

    def test_ellipsis_recovers_predicate_from_context(self):
        example = self.example(84, 'Mary does not read novels. Neither do I.')
        event = example.sentences[1]['predicates'][0]
        self.assertEqual(event['text'], 'do')
        self.assertEqual(event['roles']['action'], 'read')
        self.assertEqual(event['roles']['person'], 'user')
        self.assertEqual(event['roles']['object'], 'novels')
        self.assertEqual(event['polarity'], 'negative')
        self.assertEqual(event['context'], [1])
        second = self.groups(example)[1]
        self.assertEqual(second['predicate?'], ('do', [second['line']]))
        self.assertEqual(second['verb?'], ('do', [second['line']]))
        self.assertEqual(second['action?'], ('read', [1, second['line']]))
        self.assertEqual(second['object?'], ('novels', [1, second['line']]))

    def test_distinct_predicates_keep_their_own_times_and_actors(self):
        example = self.example(51, 'Mary was reading a book when John arrived.')
        first, second = example.sentences[0]['predicates']
        self.assertEqual((first['roles']['person'], first['aspect']), ('Mary', 'continuous'))
        self.assertEqual((second['roles']['person'], second['aspect']), ('John', 'simple'))
        self.assertEqual(second['relation'], 'during')
        first, second = self.groups(example)
        self.assertEqual(first['fragment?'][0], 'Mary was reading a book')
        self.assertEqual(second['fragment?'][0], 'when John arrived')
        self.assertEqual(first['person?'], ('Mary', [first['line']]))
        self.assertEqual(second['person?'], ('John', [second['line']]))
        self.assertEqual(first['when?'], ('when John arrived', [1, first['line']]))

    def test_surface_verbs_and_full_predicates_match_book_format(self):
        cases = [
            (30, "Doesn't Mary read books?", 0, "Doesn't read", 'read', 'read'),
            (45, 'Mary has just written the book.', 0, 'has written', 'written', 'write'),
            (48, 'Mary has written three letters today. She has been writing letters all morning.',
             1, 'has been writing', 'writing', 'write'),
            (60, 'The car was repaired by Mary.', 0, 'was repaired', 'repaired', 'repair'),
            (61, 'The child was looked after by Mary.', 0, 'was looked after', 'looked after', 'look after'),
            (76, 'Mary looks forward to meeting John.', 0, 'looks forward to', 'looks forward to', 'look forward to'),
            (77, 'Mary decided not to leave.', 1, 'not to leave', 'leave', 'leave'),
        ]
        for unit, text, index, predicate, verb, action in cases:
            with self.subTest(text=text):
                group = self.groups(self.example(unit, text))[index]
                self.assertEqual(tuple(group[key][0] for key in ('predicate?', 'verb?', 'action?')),
                                 (predicate, verb, action))

    def test_embedded_events_are_separate_and_indirect_reports_are_not_questions(self):
        for unit, text, fragments, kind in [
                (35, 'Where do you think Mary lives?', ['Where do you think', 'Mary lives'], 'question'),
                (57, 'Mary asked where John lived.', ['Mary asked', 'where John lived'], 'non_question'),
                (78, 'Mary wants John to read the book.', ['Mary wants', 'John to read the book'], 'non_question')]:
            with self.subTest(text=text):
                groups = self.groups(self.example(unit, text))
                self.assertEqual([g['fragment?'][0] for g in groups], fragments)
                self.assertNotIn('object?', groups[0])
                self.assertEqual([g['type?'][0] for g in groups], [kind, kind])
        groups = self.groups(self.example(35, 'Where do you think Mary lives?'))
        self.assertEqual(groups[1]['where?'], ('unknown', [1, groups[1]['line']]))
        groups = self.groups(self.example(63, 'If Mary had time, she would read this book.'))
        self.assertEqual(groups[1]['condition?'], ('If Mary had time', [1, groups[1]['line']]))

    def test_participial_and_gerund_fragments_follow_source_order(self):
        for unit, text, expected in [
                (75, 'Reading books is useful.', ['Reading books', 'is useful']),
                (80, 'The boy reading the book is John.', ['The boy reading the book', 'is John']),
                (80, 'Having finished the report, Mary went home.', ['Having finished the report', 'Mary went home'])]:
            with self.subTest(text=text):
                self.assertEqual([g['fragment?'][0] for g in self.groups(self.example(unit, text))], expected)

    def test_repeated_fragments_in_combined_examples_keep_distinct_actors_and_references(self):
        left = self.example(77, 'Mary wants to read the book.')
        right = self.example(77, 'John wants to read the book.')
        example = replace(left, sentences=left.sentences + right.sentences)
        groups = self.groups(example)
        self.assertEqual([groups[i]['fragment?'][0] for i in (1,3)], ['to read the book','to read the book'])
        for sentence, group in zip(example.sentences, (groups[1],groups[3])):
            self.assertEqual(group['person?'],
                             (sentence['predicates'][1]['roles']['person'], [1,group['line']]))
        self.assertNotEqual(groups[1]['person?'][0], groups[3]['person?'][0])

    def test_missing_or_invalid_spans_and_type_are_rejected(self):
        example = self.example(77, 'Mary wants to read the book.')
        for mutation in [lambda s: s.pop('fragment_spans'),
                         lambda s: s['verb_spans'].__setitem__(0, [[0,4]]),
                         lambda s: s['fragment_spans'].__setitem__(0, [[0,1]]),
                         lambda s: s['predicates'][0].pop('type'),
                         lambda s: s.update(embedded_roles={'0':{'object':[0]}})]:
            sentences = deepcopy(example.sentences)
            mutation(sentences[0])
            with self.assertRaises(ValueError):
                book.validate_annotation_example(replace(example, sentences=sentences))

    def test_review_inherits_templates_and_origins(self):
        original = {e.text: e for e in self.library.generate(1)}
        review = {e.text: e for e in self.library.generate(10)}
        for text, example in original.items():
            self.assertEqual(review[text].source_unit, 1)
            self.assertEqual(review[text].family, example.family)
            self.assertEqual(book.render_annotation(review[text]), book.render_annotation(example))

    def test_reproducible_disjoint_splits_hold_out_whole_template_families(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            stats = book.build_annotations(self.library, a)
            self.assertEqual(stats, book.build_annotations(self.library, b))
            for path in a.rglob('*.txt'):
                self.assertEqual(path.read_bytes(), (b/path.relative_to(a)).read_bytes())
            inputs = {}
            for split in ('train', 'valid', 'test'):
                stories = (a/f'{split}.txt').read_text(encoding='utf-8').strip().split('\n\n')
                self.assertTrue(stories)
                for story in stories:
                    text = story.splitlines()[0].partition(' ')[2]
                    self.assertNotIn(text, inputs)
                    inputs[text] = split
            families = {}
            for unit in self.library.units:
                for example in self.library.generate(unit):
                    families.setdefault(example.family, set()).add(inputs[example.text])
            self.assertTrue(all(len(splits) == 1 for splits in families.values()))
            self.assertEqual({p.name for p in a.iterdir()}, {'train.txt', 'valid.txt', 'test.txt'})
            self.assertEqual(set(stats['units']), set(range(1, 85)))
            self.assertTrue(all(stats['units'].values()))
            self.assertEqual(stats['unique_examples'], 20337)
            self.assertEqual(stats['families'], 197)
            self.assertEqual(stats['sentences'], 38509)
            self.assertEqual(stats['unique_examples'], len(inputs))

    def test_generation_always_includes_single_and_multi_sentence_examples(self):
        examples = [example for unit in self.library.units
                    for example in self.library.generate(unit)]
        self.assertTrue(any(len(example.sentences) == 1 for example in examples))
        self.assertTrue(any(len(example.sentences) > 1 for example in examples))
        self.assertGreater(book.build_annotations(self.library)['units'][84], 0)

    def test_invalid_library_does_not_overwrite_datasets(self):
        broken = deepcopy(self.library)
        broken.units[84]['templates'][0]['sentences'][0]['text'] = '{missing.field}.'
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for split in ('train', 'valid', 'test'):
                (out/f'{split}.txt').write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unknown placeholder'):
                book.build_annotations(broken, out)
            for path in out.iterdir():
                self.assertEqual(path.read_text(encoding='utf-8'), 'keep')

    def test_conflicting_duplicate_annotations_are_rejected(self):
        broken = deepcopy(self.library)
        duplicate = deepcopy(broken.units[1]['templates'][1])
        duplicate['id'] = 'conflicting_teacher'
        duplicate['sentences'][0]['predicates'][0]['roles']['person'] = 'wrong'
        broken.units[1]['templates'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'Conflicting annotations'):
            book.build_annotations(broken)

    def test_duplicate_inputs_join_families_in_one_split(self):
        library = deepcopy(self.library)
        duplicate = deepcopy(library.units[1]['templates'][1])
        duplicate['id'] = 'another_teacher_family'
        library.units[1]['templates'].append(duplicate)
        before = book.build_annotations(self.library)
        after = book.build_annotations(library)
        self.assertEqual(before['unique_examples'], after['unique_examples'])
        self.assertEqual(before['families'], after['families'])

    def test_bad_context_parent_predicate_and_evidence_are_rejected(self):
        example = self.example(16, 'Mary owns a car. She likes her car.')
        for field, value in [('context', [2]), ('parent', 1), ('text', 'invisible')]:
            sentences = deepcopy(example.sentences)
            sentences[1]['predicates'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                book.validate_annotation_example(replace(example, sentences=sentences))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            book.validate_story('1 Mary works.\n2 actor?\tmary\t3\n')

    def test_json_duplicates_and_format_expressions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'duplicate.json'
            path.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                book.read_json(path)
        for placeholder in ('{p.__class__}', '{p.name!r}', '{p.name:20}'):
            with self.subTest(placeholder=placeholder), self.assertRaises(ValueError):
                book.substitute(placeholder, {'p': {'name': 'Mary'}})



if __name__ == '__main__':
    unittest.main()
