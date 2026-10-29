from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book
from book_srl_annotations import substitute
from book_srl_corpus import CorpusLibrary
from book_srl_predicates import expand_predicates, predicate_from_spans, validate_predicates


def span(text, piece, start=0):
    begin = text.index(piece, start)
    return [begin, begin + len(piece)]


class PredicateSpanTests(unittest.TestCase):
    def test_copulas_with_identical_spelling_have_distinct_anchors(self):
        text = 'Yes, he is, but he is taller and seems much stronger than Paul.'
        first = span(text, 'is')
        second = span(text, 'is', first[1])
        anchors = [[first], [second], [span(text, 'seems')]]
        facts = [{'predicate': 'is'}, {'predicate': 'is'}, {'predicate': 'seems'}]
        validate_predicates(text, facts, anchors)
        self.assertNotEqual(anchors[0], anchors[1])

    def test_question_joins_discontinuous_verbal_group(self):
        text = 'Has all the necessary information been collected?'
        selected = [span(text, 'Has'), span(text, 'been collected')]
        self.assertEqual(predicate_from_spans(text, selected), 'Has been collected')
        validate_predicates(text, [{'predicate': 'Has been collected'}], [selected])

    def test_contractions_keep_exact_surface_characters(self):
        for text, predicate in [
                ("I'm reading a book.", "'m reading"),
                ("She's been sleeping.", "'s been sleeping"),
                ("We'd been discussing it.", "'d been discussing"),
                ("He didn't turn it off.", "didn't turn off"),
                ('She’s been sleeping.', '’s been sleeping')]:
            with self.subTest(text=text):
                if 'turn it off' in text:
                    selected = [span(text, "didn't turn"), span(text, 'off')]
                else:
                    selected = [span(text, predicate)]
                validate_predicates(text, [{'predicate': predicate}], [selected])

    def test_shared_auxiliary_is_allowed_between_facts(self):
        text = 'She was sitting and watching him.'
        shared = span(text, 'was')
        anchors = [[shared, span(text, 'sitting')], [shared, span(text, 'watching')]]
        facts = [{'predicate': 'was sitting'}, {'predicate': 'was watching'}]
        validate_predicates(text, facts, anchors)

    def test_phrasal_verb_particle_does_not_absorb_object(self):
        text = 'Did he turn it off?'
        selected = [span(text, 'Did'), span(text, 'turn'), span(text, 'off')]
        self.assertEqual(predicate_from_spans(text, selected), 'Did turn off')
        validate_predicates(text, [{'predicate': 'Did turn off', 'action': 'turn off',
                                    'object': 'it'}], [selected])

    def test_surface_predicate_is_independent_of_normalized_action(self):
        for text, fact, piece in [
                ('There is a book.', {'predicate': 'is', 'action': 'exist'}, 'is'),
                ('So were we.', {'predicate': 'were', 'action': 'unknown'}, 'were')]:
            validate_predicates(text, [fact], [[span(text, piece)]])

    def test_omitted_copula_has_no_invented_word(self):
        text = 'The longer the flight, the more expensive the ticket.'
        facts = [{'predicate': 'none', 'action': 'be'}, {'predicate': 'none', 'action': 'be'}]
        validate_predicates(text, facts, [[], []])
        self.assertEqual(predicate_from_spans(text, []), 'none')
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_predicates(text, [{'predicate': 'is'}], [[]])

    def test_source_matching_is_case_sensitive(self):
        text = 'Have you seen them?'
        selected = [span(text, 'Have'), span(text, 'seen')]
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_predicates(text, [{'predicate': 'have seen'}], [selected])

    def test_invalid_offsets_and_span_shapes_are_rejected(self):
        text = 'He is here.'
        bad = [None, {}, [[3]], [[3, 5, 8]], [(3, 5)], [[True, 5]],
               [[3.0, 5]], [[-1, 5]], [[3, 3]], [[5, 3]], [[3, 99]],
               [[3, 5], [3, 5]], [[3, 5], [4, 5]], [[6, 10], [3, 5]],
               [[2, 5]], [[3, 6]], [[2, 3]]]
        for selected in bad:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                predicate_from_spans(text, selected)

    def test_invalid_fact_alignment_and_missing_predicates_fail(self):
        text = 'He is here.'
        for facts, anchors in [
                ([], []), (None, []), ([{}], [[[3, 5]]]),
                ([{'predicate': None}], [[[3, 5]]]),
                ([{'predicate': ''}], [[[3, 5]]]),
                ([{'predicate': 'is'}], None),
                ([{'predicate': 'is'}], []),
                ([{'predicate': 'is'}], [[[3, 5]], []])]:
            with self.subTest(facts=facts, anchors=anchors), self.assertRaises(ValueError):
                validate_predicates(text, facts, anchors)

    def test_errors_identify_record_and_fact(self):
        with self.assertRaisesRegex(ValueError, 'u24-e11-b07b: fact 1'):
            validate_predicates('He is here.', [{'predicate': 'was'}], [[[3, 5]]],
                                label='u24-e11-b07b')


class PredicateTemplateTests(unittest.TestCase):
    def test_substitutions_move_repeated_copulas_without_matching_by_text(self):
        template = '{subject.word} {subject.aux}, but {subject.lower} {subject.aux} taller.'
        first = span(template, '{subject.aux}')
        second = span(template, '{subject.aux}', first[1])
        for subject in [
                {'word': 'He', 'lower': 'he', 'aux': 'is'},
                {'word': 'They', 'lower': 'they', 'aux': 'are'}]:
            with self.subTest(subject=subject):
                text, anchors, predicates = expand_predicates(
                    template, {'subject': subject}, [[first], [second]])
                self.assertEqual(text, substitute(template, {'subject': subject}))
                self.assertEqual(predicates, [subject['aux'], subject['aux']])
                self.assertLess(anchors[0][0][1], anchors[1][0][0])
                validate_predicates(text, [{'predicate': p} for p in predicates], anchors)

    def test_separate_subject_and_contracted_auxiliary_follow_agreement(self):
        template = '{subject.word}{subject.aux} {verb.participle} a book.'
        anchors = [[span(template, '{subject.aux}'), span(template, '{verb.participle}')]]
        for subject, expected in [
                ({'word': 'I', 'aux': "'m"}, "'m reading"),
                ({'word': 'She', 'aux': "'s"}, "'s reading"),
                ({'word': 'They', 'aux': "'re"}, "'re reading")]:
            text, expanded, predicates = expand_predicates(
                template, {'subject': subject, 'verb': {'participle': 'reading'}}, anchors)
            self.assertEqual(predicates, [expected])
            self.assertEqual(text[expanded[0][0][0]:expanded[0][0][1]], subject['aux'])

    def test_question_particle_and_multiword_slots_preserve_exact_pieces(self):
        template = '{aux.word} {subject.word} {verb.word} it {verb.particle}?'
        anchors = [[span(template, '{aux.word}'), span(template, '{verb.word}'),
                    span(template, '{verb.particle}')]]
        bindings = {'aux': {'word': "Couldn't"}, 'subject': {'word': 'the new manager'},
                    'verb': {'word': 'have turned', 'particle': 'off'}}
        text, expanded, predicates = expand_predicates(template, bindings, anchors)
        self.assertEqual(text, "Couldn't the new manager have turned it off?")
        self.assertEqual(predicates, ["Couldn't have turned off"])
        self.assertEqual([text[a:b] for a, b in expanded[0]], ["Couldn't", 'have turned', 'off'])

    def test_shared_auxiliary_and_ellipsis_survive_expansion(self):
        template = '{subject.word} was sitting and {verb.word}; the longer the better.'
        auxiliary = span(template, 'was')
        anchors = [[auxiliary, span(template, 'sitting')],
                   [auxiliary, span(template, '{verb.word}')], []]
        text, expanded, predicates = expand_predicates(
            template, {'subject': {'word': 'Margaret'}, 'verb': {'word': 'reading'}}, anchors)
        self.assertEqual(predicates, ['was sitting', 'was reading', 'none'])
        self.assertEqual(expanded[0][0], expanded[1][0])
        self.assertEqual(expanded[2], [])
        validate_predicates(text, [{'predicate': p} for p in predicates], expanded)

    def test_span_may_cover_literal_and_whole_placeholder_together(self):
        template = 'She has been {verb.word}.'
        selected = [span(template, 'has been {verb.word}')]
        text, expanded, predicates = expand_predicates(
            template, {'verb': {'word': 'sleeping'}}, [selected])
        self.assertEqual(text, 'She has been sleeping.')
        self.assertEqual(predicates, ['has been sleeping'])
        self.assertEqual(expanded, [[[4, 21]]])

    def test_partial_placeholder_boundaries_are_rejected(self):
        template = '{subject.word} is here.'
        for selected in [[[1, 13]], [[0, 7]], [[2, 5]]]:
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'splits'):
                expand_predicates(template, {'subject': {'word': "She's"}}, [selected])

    def test_padded_slot_predicate_cannot_pass_validation(self):
        template = 'He {verb.word} here.'
        anchors = [[span(template, '{verb.word}')]]
        for value in [' ', ' is', 'is ', '\tis']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                expand_predicates(template, {'verb': {'word': value}}, anchors)

    def test_explicit_optional_auxiliary_and_joiner_may_disappear(self):
        template = '{choice.aux1} {choice.actor1} or {choice.aux2}{choice.actor2} want it?'
        anchors = [[span(template, '{choice.aux1}'), span(template, '{choice.aux2}'),
                    span(template, 'want')]]
        for choice, expected in [
                ({'aux1': 'Do', 'actor1': 'you', 'aux2': 'does', 'actor2': ' your mother'},
                 'Do does want'),
                ({'aux1': 'Does', 'actor1': 'he', 'aux2': '', 'actor2': 'his friend'},
                 'Does want')]:
            text, expanded, predicates = expand_predicates(template, {'choice': choice}, anchors)
            self.assertEqual(predicates, [expected])
            self.assertTrue(all(a < b for a, b in expanded[0]))
            validate_predicates(text, [{'predicate': expected}], expanded)

    def test_explicit_optional_single_predicate_may_be_omitted(self):
        template = 'The longer the flight{optional.verb}, the better.'
        text, expanded, predicates = expand_predicates(
            template, {'optional': {'verb': ''}}, [[span(template, '{optional.verb}')]])
        self.assertEqual(predicates, ['none'])
        self.assertEqual(expanded, [[]])
        validate_predicates(text, [{'predicate': 'none'}], expanded)

    def test_empty_expansion_of_multiple_placeholders_in_one_span_is_rejected(self):
        template = 'He {verb.aux}{verb.main} here.'
        with self.assertRaisesRegex(ValueError, 'empty'):
            expand_predicates(template, {'verb': {'aux': '', 'main': ''}},
                              [[span(template, '{verb.aux}{verb.main}')]])

    def test_unsupported_template_syntax_is_rejected(self):
        for template in ['{v!r}', '{v:>3}', '{v.__class__}', '{missing}', '{v']:
            with self.subTest(template=template), self.assertRaises(ValueError):
                expand_predicates(template, {'v': 'is'}, [[]])

    def test_escaped_literal_braces_move_offsets_consistently(self):
        template = '{{note}} He is here.'
        text, anchors, predicates = expand_predicates(template, {}, [[span(template, 'is')]])
        self.assertEqual(text, '{note} He is here.')
        self.assertEqual(predicates, ['is'])
        self.assertEqual(anchors, [[[10, 12]]])


class CorpusPredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = book.Library()
        cls.corpus = CorpusLibrary(book.DEFAULT_LIBRARY)
        cls.examples = {text: facts for unit in cls.library.units
                        for _, text, facts in cls.library.generate(unit)}

    def predicates(self, identifier):
        return [fact['predicate'] for fact in self.corpus.by_id[identifier]['facts']]

    def test_original_repeated_is_points_to_each_actual_occurrence(self):
        record = self.corpus.by_id['u24-e11-b07b']
        self.assertEqual(self.predicates(record['id']), ['is', 'is', 'seems'])
        self.assertEqual(record['predicate_spans'], [[[8, 10]], [[19, 21]], [[33, 38]]])
        self.assertEqual([f['action'] for f in record['facts']], ['be', 'be', 'seem'])
        self.assertEqual(record['facts'][0]['complement'], 'unknown')

    def test_contracted_is_has_had_and_would_keep_surface_and_meaning_separate(self):
        examples = [
            ("I'm reading an interesting book.", 0, "'m reading", 'read', 'continuous'),
            ("She's reading an interesting book.", 0, "'s reading", 'read', 'continuous'),
            ("She's been sleeping for two hours already.", 0, "'s been sleeping", 'sleep',
             'perfect_continuous'),
            ("When Kate came we'd been discussing the news for half an hour already.", 1,
             "'d been discussing", 'discuss', 'perfect_continuous'),
            ("I did it and I'd do it again.", 1, "'d do", 'do', 'simple')]
        for text, index, predicate, action, aspect in examples:
            with self.subTest(text=text):
                fact = self.examples[text][index]
                self.assertEqual((fact['predicate'], fact['action'], fact['aspect']),
                                 (predicate, action, aspect))

    def test_questions_and_passive_groups_keep_discontinuous_auxiliaries(self):
        for text, predicate, action, voice in [
                ('Have you seen them?', 'Have seen', 'see', 'active'),
                ('Has all the necessary information been collected?',
                 'Has been collected', 'collect', 'passive')]:
            with self.subTest(text=text):
                fact = self.examples[text][0]
                self.assertEqual((fact['predicate'], fact['action'], fact['voice'], fact['status']),
                                 (predicate, action, voice, 'questioned'))

    def test_separated_phrasal_particle_excludes_intervening_object(self):
        record = self.corpus.by_id['u51-e2-02']
        fact = record['facts'][1]
        self.assertEqual((fact['predicate'], fact['action'], fact['object']),
                         ('turned off', 'turn off', 'alarm-clock'))
        pieces = [record['text'][a:b] for a, b in record['predicate_spans'][1]]
        self.assertEqual(' '.join(pieces), 'turned off')
        self.assertNotIn('it', pieces)

    def test_retaliation_and_return_phone_call_keep_back_in_action(self):
        for identifier, index, predicate, action in [
                ('u65-e3-example-task', 1, 'hit back', 'hit back'),
                ('u65-e3-example', 1, "wouldn't have hit back", 'hit back'),
                ('u80-e3-example-task', 0, "hadn't phoned back", 'phone back'),
                ('u80-e3-example', 0, 'Not having phoned back', 'phone back')]:
            with self.subTest(identifier=identifier):
                fact = self.corpus.by_id[identifier]['facts'][index]
                self.assertEqual((fact['predicate'], fact['action']), (predicate, action))
        self.assertEqual(self.corpus.by_id['u65-e3-example-task']['facts'][0]['action'], 'hit')

    def test_zero_ellipsis_and_retained_infinitive_marker_are_distinct(self):
        self.assertEqual(self.predicates('u24-t06'), ['none', 'is', 'to look'])
        self.assertEqual(self.predicates('u30-e8-task05'), ['is', 'none'])
        self.assertEqual(self.predicates('u26-e3-06'), ["doesn't treat", "'d like", 'to'])
        self.assertEqual(self.corpus.by_id['u26-e3-06']['facts'][2]['action'], 'treat')
        self.assertEqual(self.corpus.by_id['u24-t06']['predicate_spans'][0], [])

    def test_shared_auxiliary_and_infinitive_marker_retain_their_source_positions(self):
        self.assertEqual(self.predicates('u42-t06'), ["'ll arrive", "'ll leave", "'ll take"])
        record = self.corpus.by_id['u42-t06']
        self.assertEqual(record['predicate_spans'][1][0], record['predicate_spans'][2][0])
        self.assertEqual(self.predicates('u17-ex2-07'), ['tried', 'to concentrate', 'to find'])
        self.assertEqual(self.predicates('u06-ffbd68f0e402d0c1-1-v1-s1'),
                         ['used to sit', 'used to read'])
        self.assertEqual(self.predicates('u37-t09'), ['is', 'is'])

    def test_fact_order_does_not_confuse_main_and_relative_copulas(self):
        record = self.corpus.by_id['u01-33d261b01c56e2c9-1-v1-s1']
        self.assertEqual(self.predicates(record['id']), ['is', 'is translating'])
        self.assertGreater(record['predicate_spans'][0][0][0], record['predicate_spans'][1][0][0])
        record = self.corpus.by_id['u06-028021fafb2e5031-1-v1-s1']
        self.assertEqual(record['facts'][0]['complement'], 'the Tower of London')
        self.assertGreater(record['predicate_spans'][0][0][0], record['predicate_spans'][1][0][0])

    def test_possessive_apostrophe_and_title_period_do_not_supply_or_cut_off_verbs(self):
        self.assertEqual(self.predicates('u01-09ad56c74a720264-1-v1-s1'), ['is'])
        self.assertEqual(self.predicates('u32-e7-a08'), ['does look'])
        self.assertEqual(self.predicates('u33-t23'), ['does phone'])
        self.assertEqual(self.predicates('u32-e1-03'), ['does work'])
        self.assertEqual(self.predicates('u35-e1-a07-task'), ['Does have'])

    def test_negation_stays_with_the_correct_predicate(self):
        self.assertEqual(self.predicates('u03-4b5f58037fcf704c-1-v1-s1'), ['Be', 'not to break'])
        self.assertEqual(self.predicates('u15-ex2-09'), ['was', 'was', 'decided', 'not to eat'])
        self.assertEqual(self.predicates('u28-e4-01'), ['can never do'])
        self.assertEqual(self.predicates('u26-e3-01a'), ['are not', 'shout'])

    def test_collapsed_intention_keeps_existing_action_and_target_infinitive(self):
        fact = self.corpus.by_id['u01-a459b6b146feb900-1-v1-s1']['facts'][0]
        self.assertEqual((fact['predicate'], fact['action'], fact['intention'], fact['operation']),
                         ('to give', 'give', 'give', 'add'))
        self.assertEqual(len(self.corpus.by_id['u01-a459b6b146feb900-1-v1-s1']['facts']), 1)

    def test_slot_variants_bind_auxiliary_contraction_and_actor_together(self):
        for text, actor, predicate, action in [
                ('Mary has been learning English for two years.', 'Mary', 'has been learning', 'learn'),
                ('I have been learning English for two years.', 'user', 'have been learning', 'learn'),
                ('He has been learning English for two years.', 'he', 'has been learning', 'learn'),
                ("I'm reading an interesting book.", 'user', "'m reading", 'read'),
                ("She's reading an interesting book.", 'she', "'s reading", 'read')]:
            with self.subTest(text=text):
                fact = self.examples[text][0]
                self.assertEqual((fact['actor'], fact['predicate'], fact['action']),
                                 (actor, predicate, action))

    def test_optional_alternative_auxiliary_does_not_leave_doubled_spaces(self):
        for subject, actor, predicate in [
                ('Do you or does your mother', 'addressee or your mother', 'Do or does want'),
                ('Does he or do his parents', 'he or his parents', 'Does or do want'),
                ('Does he or his friend', 'he or his friend', 'Does want')]:
            text = subject + ' want to move to another city?'
            with self.subTest(text=text):
                facts = self.examples[text]
                self.assertEqual((facts[0]['actor'], facts[0]['predicate']), (actor, predicate))
                self.assertEqual(facts[1]['predicate'], 'to move')
                self.assertNotIn('  ', text)
                self.assertNotIn('  ', facts[0]['predicate'])

    def test_generated_qa_exposes_predicate_and_action_but_not_service_offsets(self):
        text = 'Yes, he is, but he is taller and seems much stronger than Paul.'
        rendered = book.render(text, self.examples[text])
        self.assertIn('predicate 1?\tis\t1\n', rendered)
        self.assertIn('predicate 2?\tis\t1\n', rendered)
        self.assertIn('predicate 3?\tseems\t1\n', rendered)
        self.assertIn('action 1?\tbe\t1\n', rendered)
        self.assertNotIn('spans', rendered)
        self.assertNotIn('offset', rendered)

    def test_corrupted_template_predicate_fails_before_overwriting_outputs(self):
        library = deepcopy(self.library)
        library.units[24]['templates'][0]['facts'][0]['predicate'] = 'not in the text'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'train.txt').write_text('existing output', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Predicate'):
                book.build(library, out)
            self.assertEqual((out / 'train.txt').read_text(encoding='utf-8'), 'existing output')
            self.assertFalse((out / 'valid.txt').exists())


if __name__ == '__main__':
    unittest.main()
