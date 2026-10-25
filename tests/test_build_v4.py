import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_v4 as generator


def user_blocks(story):
    blocks = []
    for row in story.strip().splitlines():
        number, _, body = row.partition(' ')
        if body.startswith('User: '):
            blocks.append({'line': int(number), 'user': body[6:], 'qa': {}})
        elif body.startswith('Assistant: '):
            blocks[-1]['assistant'] = body[11:]
        else:
            label, answer, refs = body.split('\t')
            blocks[-1]['qa'][label] = (answer, [int(ref) for ref in refs.split()])
    return blocks


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.action = generator.ACTIONS[0]

    def render_turns(self, turns, task='history'):
        with patch.object(generator, 'make_dialogue', return_value=list(turns)):
            return generator.render(random.Random(42), task, self.action)

    def uncertain_turns(self):
        return [
            ('User', 'I want to drink something.', None),
            ('Assistant', 'What would you like to drink?', None),
            ('User', 'Not sure.', None),
            ('Assistant', 'Could you choose one option?', None),
            ('User', 'No idea.', None),
            ('Assistant', 'What would you like to drink?', None),
        ]

    def test_repeated_uncertainty_keeps_fact_pending_and_marks_loop(self):
        story = self.render_turns(self.uncertain_turns())
        blocks = user_blocks(story)
        self.assertEqual([b['qa']['next action?'][0] for b in blocks],
                         ['clarify', 'clarify', 'clarify'])
        last = blocks[-1]
        self.assertEqual(last['qa']['missing?'][0], 'object')
        self.assertEqual(last['qa']['declined?'][0], 'none')
        self.assertEqual(last['qa']['loop?'][0], 'yes')
        self.assertEqual(last['qa']['assistant response?'][0], last['assistant'])
        self.assertEqual(last['qa']['ask about object?'][0], last['assistant'])
        self.assertIn('?', last['assistant'])
        self.assertTrue({b['line'] for b in blocks[1:]} <=
                        set(last['qa']['next action?'][1]))
        generator.validate_story(story)

    def test_decisions_do_not_depend_on_hidden_task_name(self):
        history = self.render_turns(self.uncertain_turns(), 'history')
        loop = self.render_turns(self.uncertain_turns(), 'loop')
        self.assertEqual(history, loop)

    def test_concrete_choice_resolves_repeated_uncertainty(self):
        turns = self.uncertain_turns() + [('User', 'Tea.', 'tea')]
        story = self.render_turns(turns)
        final = user_blocks(story)[-1]['qa']
        self.assertEqual(final['current object?'][0], 'tea')
        self.assertEqual(final['next action?'][0], 'advance')
        self.assertEqual(final['ask about object?'][0], 'none')
        generator.validate_story(story)

    def test_refusal_stops_immediately_and_preserves_unknown_object(self):
        turns = self.uncertain_turns()[:2] + [
            ('User', "I don't want to answer.", 'declined'),
            ('Assistant', 'What would you like to drink?', None),
        ]
        story = self.render_turns(turns, 'decline')
        final = user_blocks(story)[-1]
        self.assertEqual(final['qa']['next action?'][0], 'stop_clarify')
        self.assertEqual(final['qa']['current object?'][0], 'unknown')
        self.assertNotIn('?', final['assistant'])
        generator.validate_story(story)

    def test_confirmation_resolves_context_without_inventing_literal_object(self):
        turns = [
            ('User', 'I want to drink something.', None),
            ('Assistant', 'Would you like coffee?', None),
            ('User', 'Yes.', 'coffee'),
        ]
        story = self.render_turns(turns, 'yes_context')
        blocks = user_blocks(story)
        self.assertEqual(blocks[0]['qa']['ask about object?'][0], 'Would you like coffee?')
        self.assertEqual(blocks[-1]['qa']['object?'][0], 'unknown')
        self.assertEqual(blocks[-1]['qa']['current object?'][0], 'coffee')
        generator.validate_story(story)

    def test_long_loops_continue_until_answer_or_refusal(self):
        lengths = set()
        endings = set()
        for seed in range(50):
            story = generator.render(random.Random(seed), 'loop', self.action)
            blocks = user_blocks(story)
            lengths.add(len(blocks) - 2)
            self.assertEqual(blocks[2]['qa']['loop?'][0], 'yes')
            for block in blocks[:-1]:
                self.assertEqual(block['qa']['next action?'][0], 'clarify')
                self.assertIn('?', block['qa']['assistant response?'][0])
            final = blocks[-1]['qa']
            endings.add(final['next action?'][0])
            if blocks[-1]['user'] in generator.DECLINED:
                self.assertEqual(final['next action?'][0], 'stop_clarify')
                self.assertEqual(final['missing?'][0], 'object')
                self.assertEqual(final['declined?'][0], 'object')
                self.assertNotIn('?', final['assistant response?'][0])
            else:
                self.assertEqual(final['next action?'][0], 'advance')
                self.assertEqual(final['missing?'][0], 'none')
            generator.validate_story(story)
        self.assertEqual(lengths, {2, 3, 4, 5})
        self.assertEqual(endings, {'advance', 'stop_clarify'})

    def test_emotion_uncertainty_does_not_stop_clarification(self):
        long_history_seen = False
        for seed in range(100):
            story = generator.render_emotion(random.Random(seed), self.action)
            uncertain = 0
            for block in user_blocks(story):
                if block['user'] in generator.UNKNOWN:
                    uncertain += 1
                    self.assertEqual(block['qa']['next action?'][0], 'clarify')
                    self.assertEqual(block['qa']['missing?'][0], 'emotion')
                    self.assertIn('?', block['qa']['assistant response?'][0])
            long_history_seen |= uncertain >= 3
            generator.validate_story(story)
        self.assertTrue(long_history_seen)

    def test_roles_cover_known_and_unknown_objects_with_more_distinct_inputs(self):
        rng = random.Random(42)
        inputs = set()
        known = unknown = 0
        for i in range(5000):
            story = generator.render(rng, 'roles', rng.choice(generator.ACTIONS))
            inputs.add(generator.dialogue_key(story))
            block = user_blocks(story)[0]
            value = block['qa']['object?'][0]
            if value == 'unknown':
                unknown += 1
            else:
                known += 1
                self.assertIn(value.casefold(), block['user'].casefold())
            generator.validate_story(story)
        self.assertGreater(len(inputs), 2500)
        self.assertGreater(known, 1000)
        self.assertGreater(unknown, 1000)

    def test_validator_rejects_old_contradiction_and_bad_references(self):
        contradiction = (
            '1 User: Not sure.\n'
            '2 next action?\tstop_clarify\t1\n'
            '3 ask about object?\tnone\t1\n'
            '4 assistant response?\tWhat would you like to drink?\t1 2 3\n\n'
        )
        with self.assertRaisesRegex(ValueError, 'stop_clarify'):
            generator.validate_story(contradiction)
        with self.assertRaisesRegex(ValueError, 'references'):
            generator.validate_story('1 User: Tea.\n2 object?\ttea\t2\n\n')


class FactCollectionTests(unittest.TestCase):
    def test_all_required_facts_are_answered_or_explicitly_declined(self):
        coverage = set()
        for render in (generator.render_weather, generator.render_tired, generator.render_sleepy):
            for seed in range(80):
                story = render(random.Random(seed))
                blocks = user_blocks(story)
                previous_key = None
                uncertain = 0
                for index, block in enumerate(blocks):
                    qa = block['qa']
                    current = {label[len('current '):-1]: value[0]
                               for label, value in qa.items() if label.startswith('current ')}
                    missing = {key for key, value in current.items() if value == 'unknown'}
                    reported_missing = set() if qa['missing?'][0] == 'none' else set(qa['missing?'][0].split(', '))
                    declined = set() if qa['declined?'][0] == 'none' else set(qa['declined?'][0].split(', '))
                    pending = missing - declined
                    self.assertEqual(missing, reported_missing)
                    self.assertTrue(declined <= missing)
                    key = qa['next fact?'][0]
                    decision = qa['next action?'][0]
                    if index == 0 and len(missing) < len(current):
                        coverage.add('initially_known')
                    if index == 0 and not missing:
                        coverage.add('all_initially_known')
                    if pending:
                        self.assertEqual(decision, 'clarify')
                        self.assertIn(key, pending)
                        self.assertIn('?', qa['assistant response?'][0])
                        self.assertEqual(qa['ask about fact?'][0], qa['assistant response?'][0])
                        if declined:
                            coverage.add('refusal_with_other_pending_facts')
                    else:
                        self.assertEqual(key, 'none')
                        self.assertEqual(qa['ask about fact?'][0], 'none')
                        self.assertEqual(decision, 'stop_clarify' if missing else 'advance')
                    if previous_key is not None:
                        answer = qa[previous_key + '?'][0]
                        if answer == 'unknown':
                            uncertain += 1
                            self.assertEqual(key, previous_key)
                            self.assertEqual(decision, 'clarify')
                            if uncertain >= 3:
                                coverage.add('three_unknown_answers')
                        else:
                            uncertain = 0
                            self.assertNotEqual(key, previous_key)
                            if answer == 'declined':
                                self.assertIn(previous_key, declined)
                                self.assertEqual(current[previous_key], 'unknown')
                            else:
                                self.assertEqual(current[previous_key], answer)
                    previous_key = None if key == 'none' else key
                self.assertIsNone(previous_key)
                generator.validate_story(story)
        self.assertEqual(coverage, {'initially_known', 'all_initially_known',
                                    'refusal_with_other_pending_facts', 'three_unknown_answers'})

    def test_validator_rejects_premature_stop_or_advice(self):
        prefix = ('1 User: Not sure.\n'
                  '2 missing?\tweather, season\t1\n'
                  '3 declined?\tweather\t1\n')
        with self.assertRaisesRegex(ValueError, 'pending'):
            generator.validate_story(prefix + '4 next action?\tstop_clarify\t2 3\n')
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            generator.validate_story(prefix + '4 next action?\tadvance\t2 3\n')


class DatasetTests(unittest.TestCase):
    def build(self, path, counts):
        with contextlib.redirect_stdout(io.StringIO()):
            return generator.build(path, counts)

    def test_balanced_distinct_reproducible_splits_and_fixed_evaluation(self):
        counts = {'train': 260, 'valid': 65, 'test': 65}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = self.build(root / 'first', counts)
            self.build(root / 'repeat', counts)
            self.build(root / 'larger', dict(counts, train=520))
            all_keys = set()
            for split, count in counts.items():
                text = (root / 'first' / f'{split}.txt').read_text(encoding='utf-8')
                self.assertEqual(text, (root / 'repeat' / f'{split}.txt').read_text(encoding='utf-8'))
                if split != 'train':
                    self.assertEqual(text, (root / 'larger' / f'{split}.txt').read_text(encoding='utf-8'))
                stories = text.strip().split('\n\n')
                self.assertEqual(len(stories), count)
                self.assertEqual(set(stats[split]['tasks'].values()), {count // 13})
                qa_count = 0
                for story in stories:
                    key = generator.dialogue_key(story)
                    self.assertNotIn(key, all_keys)
                    all_keys.add(key)
                    qa_count += generator.validate_story(story)
                self.assertEqual(stats[split]['qa'], qa_count)

    def test_deduplication_ignores_qa_and_terminal_response_variants(self):
        a = ('1 User: I want tea.\n2 assistant response?\tSounds good.\t1\n'
             '3 Assistant: Sounds good.\n')
        b = ('1 User: I want tea.\n2 assistant response?\tGreat choice.\t1\n'
             '3 Assistant: Great choice.\n')
        self.assertEqual(generator.dialogue_key(a), generator.dialogue_key(b))
        self.assertNotEqual(generator.dialogue_key(a), generator.dialogue_key(
            a + '4 User: Actually, coffee.\n'))

    def test_exhaustion_does_not_overwrite_existing_datasets(self):
        story = '1 User: Tea.\n2 object?\ttea\t1\n\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'train.txt'
            path.write_text('original', encoding='utf-8')
            with patch.object(generator, 'render', return_value=story):
                with self.assertRaisesRegex(ValueError, 'distinct'):
                    self.build(root, {'train': 2})
            self.assertEqual(path.read_text(encoding='utf-8'), 'original')

    def test_zero_and_negative_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root, {'train': 0})
            self.assertEqual((root / 'train.txt').read_text(encoding='utf-8'), '')
            with self.assertRaisesRegex(ValueError, 'non-negative'):
                self.build(root, {'train': -1})


if __name__ == '__main__':
    unittest.main()
