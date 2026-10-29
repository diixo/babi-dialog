from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book
from book_srl_corpus import CorpusLibrary


class FactTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = book.Library()
        cls.corpus = CorpusLibrary(book.DEFAULT_LIBRARY)
        cls.templates = [t for u in cls.library.units.values() for t in u['templates']]
        cls.examples = {text: facts for u in cls.corpus.completed_units
                        for _, text, facts in cls.library.generate(u)}

    def test_each_english_sentence_has_a_template_and_its_original_facts(self):
        expected = dict(self.corpus.by_id)
        self.assertEqual(len(self.templates), len(expected))
        for template in self.templates:
            source = expected.pop(template['source']['sentence'])
            self.assertEqual(source['id'], template['id'])
            self.assertEqual(source['text'], template['source']['text'])
            self.assertEqual(self.examples[source['text']], source['facts'])
        self.assertFalse(expected)

    def test_lexical_choices_keep_their_stored_source_evidence(self):
        for domain in self.library.domains.values():
            self.assertTrue(domain['rule'])
            for row in domain['values']:
                self.assertTrue(row['evidence'], row)
                for evidence in row['evidence']:
                    self.assertIn(evidence['unit'], range(1, 85))
                    if isinstance(evidence.get('sentence'), str):
                        source = self.corpus.by_id[evidence['sentence']]
                        self.assertEqual(source['unit'], evidence['unit'])
                        self.assertEqual(source['text'][evidence['start']:evidence['end']], evidence['text'])
                    else:
                        self.assertGreater(evidence['line'], 0)
                    self.assertEqual(evidence['end']-evidence['start'], len(evidence['text']))
                    self.assertEqual(evidence['text'].casefold(), row['word'].casefold())

    def test_shared_bindings_change_grammar_and_facts_together(self):
        facts = self.examples["She's writing an interesting article."]
        self.assertEqual((facts[0]['actor'], facts[0]['action'], facts[0]['object']),
                         ('she', 'write', 'interesting article'))
        self.assertEqual(facts[0]['aspect'], 'continuous')
        facts = self.examples['I want to give him an article as a present.']
        self.assertEqual((facts[0]['action'], facts[0]['intention'], facts[0]['operation']),
                         ('give', 'give', 'add'))
        self.assertEqual(facts[0]['recipient'], 'him')
        self.assertNotIn('I want to give him a article as a present.', self.examples)
        self.assertEqual(self.examples['He has a dog.'][0]['actor'], 'he')
        self.assertNotIn('He have a dog.', self.examples)

    def test_whole_frame_semantics_restrict_lexical_combinations(self):
        self.assertIn('The cow gives milk.', self.examples)
        self.assertNotIn('The cow gives coffee.', self.examples)
        self.assertNotIn('The radio I have is very slow.', self.examples)
        self.assertNotIn('I like strong water.', self.examples)
        self.assertNotIn('We have a bad tooth.', self.examples)
        self.assertIn('The child is reading a magazine.', self.examples)
        self.assertNotIn('The child is writing a magazine.', self.examples)
        self.assertIn('The child is writing an article.', self.examples)

    def test_isolated_sentences_do_not_invent_absent_context(self):
        self.assertEqual(self.examples['Have you seen them?'][0]['object'], 'them')
        self.assertEqual(self.examples['Yes, she does.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['Put them in the wardrobe.'][0]['object'], 'them')
        self.assertEqual(self.examples['He saw the dog and liked it at once.'][1]['object'], 'dog')

    def test_flat_questions_keep_multiple_actions_separate(self):
        text = 'He saw the dog and liked it at once.'
        output = book.render(text, self.examples[text])
        self.assertIn('action 1?\tsee\t1', output)
        self.assertIn('action 2?\tlike\t1', output)
        self.assertNotIn('Assistant:', output)
        self.assertIn('predicate 1?\tsaw\t1', output)
        self.assertIn('predicate 2?\tliked\t1', output)
        self.assertNotIn('next action?', output)
        for text, facts in self.examples.items():
            lines = book.render(text, facts).strip().splitlines()
            self.assertEqual(lines[0], '1 User: ' + text)
            for index, line in enumerate(lines[1:], 2):
                self.assertTrue(line.startswith(str(index) + ' '))
                self.assertEqual(line.split('\t')[-1], '1')

    def test_repeated_builds_and_split_family_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            first = book.build(self.library, out)
            data = {p.relative_to(out): p.read_bytes() for p in out.rglob('*.txt')}
            self.assertEqual(book.build(self.library, out), first)
            self.assertEqual(data, {p.relative_to(out): p.read_bytes() for p in out.rglob('*.txt')})
            self.assertEqual(set(data), {Path(f'{split}.txt') for split in ('train','valid','test')})
            self.assertEqual(first['units'], list(range(1,85)))
            membership = {}
            for split in ('train', 'valid', 'test'):
                for story in (out / (split + '.txt')).read_text(encoding='utf-8').strip().split('\n\n'):
                    text = story.splitlines()[0][8:]
                    self.assertNotIn(text, membership)
                    membership[text] = split
            families = {}
            for unit in first['units']:
                for family, text, _ in self.library.generate(unit):
                    families.setdefault(family, set()).add(membership[text])
            self.assertTrue(all(len(splits) == 1 for splits in families.values()))

    def test_missing_or_conflicting_annotations_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            path = out / 'train.txt'
            path.write_text('existing', encoding='utf-8')
            missing = deepcopy(self.library)
            missing.units[84]['templates'] = []
            with self.assertRaisesRegex(ValueError, 'no prepared templates'):
                book.build(missing, out)
            lib = deepcopy(self.library)
            duplicate = deepcopy(lib.units[1]['templates'][0])
            duplicate['family'] += '-conflict'
            duplicate['facts'][0]['actor'] = 'wrong'
            lib.units[1]['templates'].append(duplicate)
            with self.assertRaisesRegex(ValueError, 'Conflicting facts'):
                book.build(lib, out)
            self.assertEqual(path.read_text(encoding='utf-8'), 'existing')
            self.assertFalse((out / 'valid.txt').exists())

    def test_build_rejects_partial_lessons_instead_of_omitting_them(self):
        lib = deepcopy(self.library)
        pending = 84
        lib.coverage['completed_units'].remove(pending)
        lib.coverage['pending_units'] = [pending]
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            book.build(lib)

    def test_quantity_variants_respect_countability_and_keep_scarcity_positive(self):
        for text, amount in [('I have little money.', 'little'), ('I have a little money.', 'a little'),
                             ('I have few friends.', 'few'), ('I have a few friends.', 'a few')]:
            fact = self.examples[text][0]
            self.assertEqual(fact['quantity'], amount)
            self.assertEqual(fact['polarity'], 'positive')
        for text in ['I have few money.', 'I have a few news.', 'I have little friends.',
                     'He has so a lot of friends!', 'We spent too much bread.']:
            self.assertNotIn(text, self.examples)
        self.assertEqual(self.examples['He drinks too much wine.'][0]['object'], 'wine')
        self.assertEqual(self.examples['How much fish does the cat need?'][0]['actor'], 'cat')

    def test_reported_conditional_and_purpose_actions_are_not_asserted_events(self):
        facts = self.examples['He says he has a lot of problems.']
        self.assertEqual([(f['action'], f['status']) for f in facts],
                         [('say', 'asserted'), ('have', 'reported')])
        facts = self.examples["I'll help you if I have a few minutes."]
        self.assertEqual([(f['action'], f['status']) for f in facts],
                         [('help', 'conditional'), ('have', 'hypothetical')])
        facts = self.examples['We want him to stay with us in summer.']
        self.assertEqual((facts[1]['actor'], facts[1]['action'], facts[1]['status']),
                         ('he', 'stay', 'intended'))
        self.assertEqual(facts[1]['companion'], 'us')

    def test_possessive_variants_keep_joint_and_separate_owners(self):
        fact = self.examples["These are Ted and Bill's children."][0]
        self.assertEqual((fact['possessor'], fact['possession']), ('Ted and Bill', 'joint'))
        fact = self.examples["The teacher praised Nick's and Susan's answers."][0]
        self.assertEqual((fact['object'], fact['possessor'], fact['possession']),
                         ("Nick's and Susan's answers", 'Nick and Susan', 'separate'))
        for text in ["These are Ted's and Bill's children.",
                     "The teacher praised Nick and Susan's answers."]:
            self.assertNotIn(text, self.examples)
        self.assertEqual(self.examples['The front of the house needs painting.'][1]['object'],
                         'front of the house')

    def test_pronoun_variants_preserve_case_agreement_and_owner(self):
        facts = self.examples['He took his umbrella and left.']
        self.assertEqual([f['actor'] for f in facts], ['he', 'he'])
        self.assertEqual((facts[0]['object'], facts[0]['possessor']), ('his umbrella', 'he'))
        self.assertEqual(self.examples['I saw her yesterday.'][0]['object'], 'her')
        self.assertEqual(self.examples['Can you help me?'][0]['object'], 'user')
        for text in ['I saw she yesterday.', 'I saw me yesterday.', 'I love me.', 'Can you help you?',
                     'This is mine bag.', 'Where is my?', 'He took her umbrella and left.']:
            self.assertNotIn(text, self.examples)

    def test_possessive_references_use_only_the_current_sentence(self):
        self.assertEqual(self.examples['My bag is black, hers is brown.'][1]['actor'], 'her bag')
        self.assertEqual(self.examples['These are not her keys but his.'][1]['complement'], 'his keys')
        self.assertEqual(self.examples['Where is mine?'][0]['actor'], 'mine')
        self.assertEqual(self.examples['Can I use yours?'][0]['object'], 'yours')
        self.assertEqual(self.examples['His car is very expensive.'][0]['possessor'], 'he')
        self.assertEqual(self.examples['Put them on the shelf.'][0]['object'], 'them')
        self.assertEqual(self.examples['They liked the cat and bought it.'][1]['object'], 'cat')

    def test_pronoun_task_and_solution_share_a_split_family(self):
        templates = {t['id']: t for t in self.templates}
        for task, solution in [('u16-ex1-task02', 'u16-ex1-02'), ('u16-ex1-task04', 'u16-ex1-04'),
                               ('u16-ex2-task01', 'u16-ex2-01'), ('u16-ex2-task04', 'u16-ex2-04')]:
            self.assertEqual(templates[task]['family'], templates[solution]['family'])

    def test_commands_and_proposals_keep_the_existing_annotation_conventions(self):
        for text in ['Give the cat a little fish every day.', 'Take these skis.', 'Close your eyes.']:
            fact = self.examples[text][0]
            self.assertEqual((fact['actor'], fact['status'], fact['operation']),
                             ('addressee', 'required', 'none'))
        fact = self.examples["Let's go home."][0]
        self.assertEqual((fact['actor'], fact['action'], fact['intention'], fact['operation']),
                         ('we', 'go', 'go', 'add'))
        self.assertEqual(fact['status'], 'intended')

    def test_reflexive_roles_point_to_the_actor_in_each_syntactic_role(self):
        for text, role in [('I hurt myself.', 'object'), ('He made himself a cup of tea.', 'beneficiary'),
                           ('Jane looked at herself in the mirror.', 'target'),
                           ('They bought the books for themselves.', 'beneficiary')]:
            fact = self.examples[text][0]
            self.assertEqual(fact[role], fact['actor'])
            self.assertEqual(fact['reflexive'], 'yes')
        fact = self.examples['She likes to talk about herself.'][1]
        self.assertEqual((fact['actor'], fact['topic'], fact['reflexive']), ('she', 'she', 'yes'))
        self.assertEqual(self.examples['Behave yourself!'][0]['object'], 'addressee')
        output = book.render('I hurt myself.', self.examples['I hurt myself.'])
        self.assertIn('reflexive?\tyes\t1', output)

    def test_reflexive_bindings_change_the_subject_and_self_form_together(self):
        facts = self.examples['We burnt ourselves.']
        self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[0]['action']), ('we', 'we', 'burn'))
        self.assertEqual(self.examples['She made herself a cup of tea.'][0]['beneficiary'], 'she')
        self.assertEqual(self.examples['Steve looked at himself in the mirror.'][0]['target'], 'Steve')
        self.assertEqual(self.examples['They can shave themselves.'][0]['object'], 'they')
        for text in ['We burnt myself.', 'She made himself a cup of tea.', 'I can shave himself.',
                     'Steve looked at herself in the mirror.']:
            self.assertNotIn(text, self.examples)

    def test_emphasis_and_by_oneself_are_not_reflexive_objects(self):
        fact = self.examples['You yourself told us the story.'][0]
        self.assertEqual((fact['object'], fact['emphasis'], fact['reflexive']), ('story', 'actor', 'no'))
        fact = self.examples['We translated the article ourselves.'][0]
        self.assertEqual((fact['object'], fact['reflexive']), ('article', 'no'))
        fact = self.examples['The child always washes by himself.'][0]
        self.assertEqual((fact['object'], fact['manner'], fact['reflexive']), ('none', 'without help', 'no'))
        fact = self.examples['The child always washes himself.'][0]
        self.assertEqual((fact['object'], fact['reflexive']), ('child', 'yes'))
        facts = self.examples['I cut myself when I was cutting bread.']
        self.assertEqual([(f['object'], f['reflexive']) for f in facts], [('user', 'yes'), ('bread', 'no')])

    def test_indefinite_pronouns_bind_negation_and_singular_agreement(self):
        for text, actor, polarity in [('Somebody has arrived.', 'somebody', 'positive'),
                                      ('Everybody has arrived.', 'everybody', 'positive'),
                                      ('Nobody has arrived.', 'anybody', 'negative')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['actor'], fact['polarity']), (actor, polarity))
        self.assertEqual(self.examples['Nothing is ready.'][0]['polarity'], 'negative')
        self.assertEqual(self.examples['There is nothing to eat.'][0]['polarity'], 'negative')
        self.assertEqual(self.examples['There is something to eat.'][0]['polarity'], 'positive')
        for text in ['Everybody have arrived.', "Nobody hasn't arrived.", "He doesn't have no milk in the fridge."]:
            self.assertNotIn(text, self.examples)
        self.assertEqual(self.examples["I didn't understand everything in this text."][0]['object'],
                         'everything in this text')
        self.assertEqual(self.examples["I can't understand anything."][0]['object'], 'anything')

    def test_ellipsis_and_tag_questions_do_not_invent_context_or_invert_assertions(self):
        self.assertEqual(self.examples["Yes, I've read some."][0]['object'], 'some')
        self.assertEqual(self.examples['Yes, they have.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['Unfortunately, none of us do.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['He gave one to me and kept the other.'][1]['object'], 'the other')
        self.assertEqual(self.examples['How many others does he want to buy?'][1]['object'], 'others')
        fact = self.examples["Everybody has come, haven't they?"][0]
        self.assertEqual((fact['polarity'], fact['status']), ('positive', 'questioned'))

    def test_reciprocal_frames_require_groups_and_are_distinct_from_reflexives(self):
        fact = self.examples['Tom and Robert looked at each other.'][0]
        self.assertEqual((fact['actor'], fact['target'], fact['reciprocal'], fact['reflexive']),
                         ('Tom and Robert', 'each other', 'yes', 'no'))
        facts = self.examples['They looked at each other and laughed.']
        self.assertEqual(facts[0]['reciprocal'], 'yes')
        self.assertNotIn('reciprocal', facts[1])
        self.assertEqual(self.examples['They all looked at one another.'][0]['reciprocal'], 'yes')
        self.assertNotIn('He looked at each other and laughed.', self.examples)

    def test_another_keeps_singular_items_and_numbered_plural_durations(self):
        self.assertEqual(self.examples['Give me another apple.'][0]['object'], 'another apple')
        self.assertEqual(self.examples['I need another two days.'][0]['quantity'], 'another two')
        for text in ['Give me another an apple.', 'Can I have another apples?', 'I need another two day.',
                     'Where are the others students?']:
            self.assertNotIn(text, self.examples)

    def test_alternative_subjects_are_not_two_asserted_events(self):
        facts = self.examples['Either they or Jack knows it.']
        self.assertEqual(len(facts), 1)
        self.assertEqual((facts[0]['actor'], facts[0]['coordination'], facts[0]['coordination_role']),
                         ('they or Jack', 'or', 'actor'))
        fact = self.examples['Neither Fred nor George can drive.'][0]
        self.assertEqual((fact['actor'], fact['polarity'], fact['modality']),
                         ('Fred or George', 'negative', 'can'))
        fact = self.examples['Both James and Sam collect stamps.'][0]
        self.assertEqual((fact['actor'], fact['coordination']), ('James and Sam', 'and'))
        fact = self.examples['I want neither tea nor coffee.'][0]
        self.assertEqual((fact['object'], fact['coordination_role'], fact['polarity']),
                         ('tea or coffee', 'object', 'negative'))

    def test_coordinated_subjects_keep_nearest_subject_agreement(self):
        self.assertIn('Either his wife or his parents know it.', self.examples)
        fact = self.examples['Neither his parents nor he is married.'][0]
        self.assertEqual((fact['actor'], fact['polarity']), ('his parents or he', 'negative'))
        for text in ['Either his wife or his parents knows it.', 'Either they or Jack know it.',
                     'Neither his parents nor he am married.', 'Both James and Sam has been to London.']:
            self.assertNotIn(text, self.examples)
        self.assertIn('Both James and Sam have been to London.', self.examples)

    def test_review_lesson_retains_reflexive_reciprocal_and_emphatic_distinctions(self):
        fact = self.examples["They'll pay for themselves."][0]
        self.assertEqual((fact['actor'], fact['beneficiary'], fact['reflexive']), ('they', 'they', 'yes'))
        fact = self.examples["We've known one another for a long time."][0]
        self.assertEqual((fact['reciprocal'], fact['reflexive']), ('yes', 'no'))
        fact = self.examples['I painted the house myself.'][0]
        self.assertEqual((fact['object'], fact['emphasis'], fact['reflexive']), ('house', 'actor', 'no'))
        fact = self.examples['The child is big enough to dress himself.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['reflexive']), ('child', 'child', 'yes'))

    def test_linking_adjectives_manner_and_hardly_keep_different_roles(self):
        fact = self.examples['The perfume smells sweet.'][0]
        self.assertEqual(fact['complement'], 'sweet')
        self.assertNotIn('manner', fact)
        fact = self.examples['Nick works carefully.'][0]
        self.assertEqual(fact['manner'], 'carefully')
        self.assertNotIn('complement', fact)
        self.assertEqual(self.examples['Nick works hard.'][0]['manner'], 'hard')
        fact = self.examples['He hardly studies at all.'][0]
        self.assertEqual((fact['degree'], fact['polarity']), ('hardly at all', 'positive'))
        self.assertNotIn('manner', fact)
        self.assertNotIn('The perfume smells sweetly.', self.examples)
        fact = self.examples['The excursion was badly organized.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice'], fact['manner']),
                         ('unknown', 'excursion', 'passive', 'badly'))

    def test_comparisons_keep_target_degree_and_negation(self):
        fact = self.examples['Mark is younger than Sam.'][0]
        self.assertEqual((fact['actor'], fact['comparison'], fact['degree']), ('Mark', 'Sam', 'comparative'))
        fact = self.examples["This picture is not as colourful as that one."][0]
        self.assertEqual((fact['comparison'], fact['degree'], fact['polarity']),
                         ('that picture', 'equative', 'negative'))
        fact = self.examples['This is the highest mountain in the world.'][0]
        self.assertEqual((fact['degree'], fact['comparison']), ('superlative', 'mountains in the world'))
        for text in ['Sam is younger than Sam.', 'Mark is more handsome than Mark.',
                     'This picture is as more beautiful as that one.']:
            self.assertNotIn(text, self.examples)
        facts = self.examples['The less time I have, the more nervous I get.']
        self.assertTrue(all(f['status'] == 'conditional' for f in facts))
        self.assertEqual((facts[0]['quantity'], facts[1]['condition']), ('less', 'as I have less time'))

    def test_double_comparative_forms_keep_their_meaning_and_distribution(self):
        for text in ['The shop is farther along the road.', 'The shop is further along the road.']:
            self.assertEqual(self.examples[text][0]['attribute'], 'distance')
        self.assertEqual(self.examples['For further information speak to the secretary.'][0]['purpose'],
                         'further information')
        self.assertIn('My wife has an elder sister.', self.examples)
        self.assertIn('William is the eldest of the three brothers.', self.examples)
        for text in ['For farther information speak to the secretary.', 'My husband is a year elder than me.',
                     'Tina is the eldest of the three brothers.', "I'll phone him latest week."]:
            self.assertNotIn(text, self.examples)
        self.assertEqual(self.examples["I'll phone him next Monday."][0]['time'], 'next Monday')
        self.assertEqual(self.examples['It was published last week.'][0]['time'], 'last week')

    def test_former_latter_resolution_requires_an_antecedent_in_the_same_sentence(self):
        facts = self.examples["I sent the information to Steve but the latter didn't answer."]
        self.assertEqual((facts[0]['recipient'], facts[1]['actor']), ('Steve', 'Steve'))
        facts = self.examples['The teacher asked the student the question several times but the latter was silent.']
        self.assertEqual(facts[1]['actor'], 'student')
        facts = self.examples['The former is a businessman and the latter is a dentist.']
        self.assertEqual([f['actor'] for f in facts], ['the former', 'the latter'])
        self.assertEqual(self.examples["But the latter didn't even react."][0]['actor'], 'the latter')

    def test_comparison_rewrites_and_printed_variants_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u24-e6-task01', 'u24-e6-01'), ('u21-e3-task01', 'u21-e3-01'),
                              ('u21-t13a', 'u21-t13b'), ('u25-e2-b04a', 'u25-e2-b04b')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_adverb_comparisons_preserve_manner_target_and_passive_roles(self):
        fact = self.examples['Sarah speaks English much more fluently than her friend.'][0]
        self.assertEqual((fact['manner'], fact['comparison'], fact['degree'], fact['intensity']),
                         ('more fluently', 'her friend', 'comparative', 'much'))
        self.assertNotIn('complement', fact)
        fact = self.examples["My room is much more beautifully decorated for the holiday than my parents' room."][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice']), ('unknown', 'my room', 'passive'))
        self.assertEqual(fact['comparison'], "my parents' room")
        self.assertEqual(self.examples['He draws worst of all.'][0]['manner'], 'worst')
        for text in ['He draws the best of all.', 'Sarah speaks English much more fluent than her friend.',
                     'Speak more slow, please.']:
            self.assertNotIn(text, self.examples)

    def test_correlative_review_sentences_preserve_dependencies_and_local_references(self):
        facts = self.examples['The harder he works, the worse he sleeps.']
        self.assertTrue(all(fact['status'] == 'conditional' for fact in facts))
        self.assertEqual(facts[1]['condition'], 'as he works harder')
        facts = self.examples["Though this flat is smaller than mine, it's more comfortable and besides it's surprisingly cheap."]
        self.assertEqual([fact['actor'] for fact in facts], ['this flat'] * 3)
        self.assertEqual(facts[0]['comparison'], 'my flat')
        self.assertEqual(self.examples['The latter is married.'][0]['actor'], 'the latter')
        self.assertEqual(self.examples['Children like singing but often do it badly.'][1]['action'], 'sing')

    def test_word_order_examples_keep_manner_location_time_and_duration_separate(self):
        fact = self.examples['He slept soundly in his room all night long.'][0]
        self.assertEqual((fact['manner'], fact['location'], fact['duration']),
                         ('soundly', 'in his room', 'all night long'))
        fact = self.examples["They came home safely at seven o'clock."][0]
        self.assertEqual((fact['destination'], fact['manner'], fact['time']),
                         ('home', 'safely', "at seven o'clock"))
        self.assertNotIn('location', fact)
        self.assertEqual(self.examples['These flowers smell strong in the evening.'][0]['complement'], 'strong')
        self.assertEqual(self.examples['He answered very badly at the exam the day before yesterday.'][0]['manner'],
                         'very badly')
        for text in ['They came to home safely at seven o\'clock.', 'These flowers smell strongly in the evening.',
                     'He slept hard in his room all night long.']:
            self.assertNotIn(text, self.examples)

    def test_frequency_changes_preserve_negation_and_reciprocal_action(self):
        plain = self.examples["We don't see each other nowadays."][0]
        frequent = self.examples["We don't often see each other nowadays."][0]
        self.assertNotIn('frequency', plain)
        self.assertEqual(frequent['frequency'], 'often')
        self.assertEqual((frequent['reciprocal'], frequent['reflexive'], frequent['polarity']),
                         ('yes', 'no', 'negative'))
        fact = self.examples['You must never say it again.'][0]
        self.assertEqual((fact['status'], fact['polarity'], fact['action']), ('required', 'negative', 'say'))

    def test_have_got_is_present_possession_and_does_not_replace_event_have(self):
        for text in ['He has a car.', 'He has got a car.', 'They have got a car.',
                     'She has a house of her own.', 'She has got a house of her own.']:
            fact = self.examples[text][0]
            self.assertEqual((fact['action'], fact['tense'], fact['aspect']), ('have', 'present', 'simple'))
        facts = self.examples["I couldn't answer the telephone as I was having a shower."]
        self.assertEqual((facts[1]['action'], facts[1]['object'], facts[1]['tense'], facts[1]['aspect']),
                         ('have', 'shower', 'past', 'continuous'))
        fact = self.examples["Haven't you ever had a toothache?"][0]
        self.assertEqual((fact['action'], fact['aspect'], fact['polarity'], fact['status']),
                         ('have', 'perfect', 'negative', 'questioned'))
        for text in ['He has got classes three times a week.', 'I have got a shower every morning.',
                     'Kate has got tea for breakfast.', 'Does he has a car?', 'Has he have a car?']:
            self.assertNotIn(text, self.examples)

    def test_possession_bindings_keep_subject_number_and_questions_in_sync(self):
        self.assertEqual(self.examples['Robert has a bicycle.'][0]['object'], 'bicycle')
        self.assertEqual(self.examples['Does Mary have a camera?'][0]['status'], 'questioned')
        fact = self.examples["John and Jill haven't got a three-room flat."][0]
        self.assertEqual((fact['actor'], fact['polarity']), ('John and Jill', 'negative'))
        for text in ['John and Jill has got a three-room flat.', 'Does Mary has a camera?',
                     'Has John and Jill got a three-room flat?']:
            self.assertNotIn(text, self.examples)

    def test_negative_questions_and_short_answers_keep_independent_polarity(self):
        fact = self.examples["Doesn't Nick have breakfast at home?"][0]
        self.assertEqual((fact['action'], fact['polarity'], fact['status']), ('have', 'negative', 'questioned'))
        fact = self.examples['Yes, he does.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['polarity']), ('he', 'unknown', 'positive'))
        self.assertEqual(self.examples["Hasn't she got any?"][0]['object'], 'any')
        self.assertEqual(self.examples['No, she learns German.'][0]['polarity'], 'positive')
        facts = self.examples['Would you like to visit him?']
        self.assertEqual([(fact['action'], fact['status'], fact['intention']) for fact in facts],
                         [('like', 'questioned', 'none'), ('visit', 'hypothetical', 'none')])

    def test_no_necessity_is_distinct_from_a_prohibition(self):
        fact = self.examples["They won't have to get up early on Sunday."][0]
        self.assertEqual((fact['modality'], fact['status'], fact['negation_scope']),
                         ('have to', 'not required', 'modality'))
        fact = self.examples['You must never say it again.'][0]
        self.assertEqual((fact['modality'], fact['status'], fact['polarity']), ('must', 'required', 'negative'))

    def test_alternative_questions_keep_the_selected_role_in_one_questioned_fact(self):
        for text, role, value in [
            ('Does he or his friend have a very interesting book?', 'actor', 'he or his friend'),
            ('Does Jane like apple or orange juice?', 'object', 'apple juice or orange juice'),
            ('Are they in Spain or in France now?', 'location', 'in Spain or in France'),
            ('Does John make few or many mistakes in his tests?', 'quantity', 'few or many'),
            ("Is it your or your sister's dress?", 'possessor', 'addressee or your sister'),
            ('Does your Dad go to work by car or by underground?', 'means', 'car or underground'),
        ]:
            facts = self.examples[text]
            self.assertEqual(len(facts), 1)
            fact = facts[0]
            self.assertEqual((fact[role], fact['coordination_role'], fact['coordination'], fact['status']),
                             (value, role, 'or', 'questioned'))
        fact = self.examples['Does he or do his parents make breakfast?'][0]
        self.assertEqual(fact['actor'], 'he or his parents')
        self.assertNotIn('Do he or does his parents make breakfast?', self.examples)

    def test_new_task_rewrites_and_printed_alternatives_stay_in_the_same_family(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u28-e4-task09', 'u28-e4-09'), ('u29-e1-task01', 'u29-e1-01q'),
                              ('u29-e1-01q', 'u29-e1-01n'), ('u30-e1-task01', 'u30-e1-01'),
                              ('u30-e8-task04', 'u30-e8-04'), ('u30-e9-b07a', 'u30-e9-b07b'),
                              ('u27-e4-b07b', 'u27-e4-b07e')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_tag_questions_keep_main_clause_polarity_and_special_agreement(self):
        for text, actor, polarity in [("I am late, aren't I?", 'user', 'positive'),
                                      ('I am not late, am I?', 'user', 'negative'),
                                      ("They are late, aren't they?", 'they', 'positive'),
                                      ('George and Kate never quarrel, do they?', 'George and Kate', 'negative')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['actor'], fact['polarity'], fact['status']), (actor, polarity, 'questioned'))
        self.assertEqual(self.examples["Everybody thinks so, don't they?"][0]['actor'], 'everybody')
        for text in ["I am late, amn't I?", 'I am not late, are I?', "They is late, isn't they?",
                     "Everybody thinks so, doesn't he?"]:
            self.assertNotIn(text, self.examples)

    def test_imperative_and_proposal_tags_do_not_invert_the_requested_action(self):
        for text, action, polarity in [('Close the window, will you?', 'close', 'positive'),
                                      ("Don't close the window, will you?", 'close', 'negative'),
                                      ('Be quick, will you?', 'be', 'positive')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['action'], fact['polarity'], fact['status']), (action, polarity, 'required'))
        fact = self.examples["Let's have dinner now, shall we?"][0]
        self.assertEqual((fact['actor'], fact['action'], fact['intention'], fact['operation']),
                         ('we', 'have', 'have', 'add'))
        self.assertEqual(fact['status'], 'intended')

    def test_wh_subject_object_possessor_and_passive_agent_are_distinct(self):
        subject = self.examples['Who likes your friend?'][0]
        obj = self.examples['Who does your friend like?'][0]
        self.assertEqual((subject['actor'], subject['object']), ('unknown', 'your friend'))
        self.assertEqual((obj['actor'], obj['object']), ('your friend', 'unknown'))
        fact = self.examples['Whose sister knows French?'][0]
        self.assertEqual((fact['actor'], fact['possessor'], fact['object']), ('sister', 'unknown', 'French'))
        fact = self.examples['Who is this novel written by?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice']), ('unknown', 'this novel', 'passive'))
        fact = self.examples['Who does the teacher want to answer this question?'][1]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['status']),
                         ('unknown', 'answer', 'this question', 'hypothetical'))
        self.assertNotIn('Who know French?', self.examples)

    def test_prepositional_questions_mark_the_actual_missing_role(self):
        for text, role in [('Who does Rachel write letters to?', 'recipient'),
                           ('Who is Vera waiting for?', 'target'),
                           ('Who does Kate usually have dinner with?', 'companion'),
                           ('What is this book about?', 'topic'),
                           ('What do you need his telephone number for?', 'purpose'),
                           ('Who is this letter from?', 'origin')]:
            fact = self.examples[text][0]
            self.assertEqual(fact[role], 'unknown')
            self.assertEqual(fact['status'], 'questioned')
        fact = self.examples['What do you always thank people for?'][0]
        self.assertEqual((fact['object'], fact['reason']), ('people', 'unknown'))

    def test_person_description_questions_keep_occupation_character_and_appearance(self):
        for text, attribute in [('What is her husband?', 'occupation'),
                                ('What is her husband like?', 'character'),
                                ('What does her husband look like?', 'appearance'),
                                ('Who is he?', 'identity')]:
            self.assertEqual(self.examples[text][0]['attribute'], attribute)
        self.assertEqual(self.examples['What colour are her eyes?'][0]['attribute'], 'colour')
        self.assertEqual(self.examples['How old is his brother?'][0]['attribute'], 'age')

    def test_time_required_is_not_a_transfer_and_does_not_resolve_other_stories(self):
        facts = self.examples['How long does it take you to get to work?']
        self.assertEqual((facts[0]['object'], facts[0]['duration'], facts[0]['experiencer']),
                         ('time', 'unknown', 'addressee'))
        self.assertEqual((facts[1]['action'], facts[1]['destination']), ('get', 'work'))
        facts = self.examples['It takes me about two or three hours to do it.']
        self.assertEqual((facts[0]['duration'], facts[1]['object']), ('about two or three hours', 'it'))
        self.assertEqual(self.examples['It takes him about half an hour to read it.'][1]['object'], 'it')
        self.assertEqual(self.examples['My sister does.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['This word does.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['But they can.'][0]['action'], 'unknown')

    def test_journey_variants_use_compatible_routes_and_preserve_roles(self):
        facts = self.examples['How long does it take to get by bus from your house to the city centre?']
        self.assertEqual((facts[1]['origin'], facts[1]['destination'], facts[1]['means']),
                         ('your house', 'city centre', 'bus'))
        self.assertEqual(facts[0]['duration'], 'unknown')
        self.assertEqual(self.examples['It takes me about forty minutes to clean my room.'][0]['duration'],
                         'about forty minutes')
        for text in ['How long does it take to get on foot from Moscow to London?',
                     'How long does it take to get by bicycle from France to England?',
                     'How long does it take to get by underground from Glasgow to Edinburgh?',
                     'It takes me five years to clean my room.']:
            self.assertNotIn(text, self.examples)

    def test_indirect_questions_do_not_project_matrix_negation_or_assert_if_clauses(self):
        facts = self.examples["I don't know why they are late."]
        self.assertEqual([(f['action'], f['polarity'], f['status']) for f in facts],
                         [('know', 'negative', 'asserted'), ('be', 'positive', 'questioned')])
        facts = self.examples["I wonder why he doesn't do his work properly."]
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('positive', 'negative'))
        for text in ['Do you know if Sam has an elder sister?',
                     'Do you know whether Sam has an elder sister?']:
            facts = self.examples[text]
            self.assertEqual((facts[1]['action'], facts[1]['status']), ('have', 'questioned'))
            self.assertNotIn('condition', facts[1])
        facts = self.examples['Tell me what I must do.']
        self.assertEqual((facts[0]['status'], facts[1]['status']), ('required', 'questioned'))

    def test_embedded_word_order_binds_subject_number_and_event_verb(self):
        facts = self.examples['Where do you think her husband is?']
        self.assertEqual((facts[0]['action'], facts[1]['actor'], facts[1]['location']),
                         ('think', 'her husband', 'unknown'))
        facts = self.examples['Can you tell me when the train arrives?']
        self.assertEqual((facts[1]['actor'], facts[1]['action'], facts[1]['time']),
                         ('train', 'arrive', 'unknown'))
        for text in ['Where do you think is her husband?', 'Where do you think her husband are?',
                     'Can you tell me when does the train arrive?', 'Can you tell me when the film arrives?',
                     'Do you know whether has Sam an elder sister?']:
            self.assertNotIn(text, self.examples)
        facts = self.examples['Who do you think knows the answer?']
        self.assertEqual((facts[1]['actor'], facts[1]['object']), ('unknown', 'answer'))
        facts = self.examples['Do you think Frank and Eve are in love with each other?']
        self.assertEqual((facts[1]['actor'], facts[1]['reciprocal'], facts[1]['reflexive']),
                         ('Frank and Eve', 'yes', 'no'))

    def test_new_direct_indirect_rewrites_and_printed_alternatives_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u32-e2-13-task', 'u32-e2-13'), ('u33-e1-01-task', 'u33-e1-01'),
                              ('u34-e1-02-task', 'u34-e1-02'), ('u35-e1-b09-task', 'u35-e1-b09'),
                              ('u32-e4-a01', 'u32-e4-a01-alt'), ('u33-e3-05a', 'u33-e3-05b'),
                              ('u34-e1-example-if', 'u34-e1-example-whether')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_review_questions_keep_roles_without_borrowing_reply_context(self):
        facts = self.examples['Who takes you to the airport?']
        self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[0]['destination']),
                         ('unknown', 'addressee', 'airport'))
        self.assertNotIn('Who takes you to airport?', self.examples)
        facts = self.examples['I think Ron can.']
        self.assertEqual((facts[1]['actor'], facts[1]['action'], facts[1]['status']),
                         ('Ron', 'unknown', 'reported'))
        self.assertEqual(self.examples["I think it's Tom's."][1]['complement'], "Tom's")
        self.assertEqual(self.examples['No, I see him for the first time.'][0]['polarity'], 'positive')
        fact = self.examples["I can't remember whether Sam's sister is younger or older than him."][1]
        self.assertEqual((fact['comparison'], fact['coordination'], fact['status']),
                         ('Sam', 'or', 'questioned'))

    def test_existential_there_and_known_subject_location_are_distinct(self):
        exists = self.examples['There is a book on the table.'][0]
        located = self.examples['The book is on the table.'][0]
        self.assertEqual((exists['actor'], exists['action'], exists['object']), ('none', 'exist', 'book'))
        self.assertEqual((located['actor'], located['action'], located['object']), ('book', 'be', 'none'))
        self.assertEqual(exists['location'], located['location'])
        self.assertEqual(self.examples['Yes, there is.'][0]['object'], 'unknown')
        self.assertEqual(self.examples['There are interesting stories in it.'][0]['location'], 'in it')
        fact = self.examples['There will be a good film on TV tomorrow.'][0]
        self.assertEqual((fact['tense'], fact['time']), ('future', 'tomorrow'))

    def test_existential_coordination_binds_the_first_nouns_number(self):
        facts = self.examples['There is a dictionary and two pencils on the table.']
        self.assertEqual([(f['object'], f['quantity']) for f in facts], [('dictionary', 'one'), ('pencils', 'two')])
        reverse = self.examples['There are two pencils and a dictionary on the table.']
        self.assertEqual([(f['object'], f['quantity']) for f in reverse], [('pencils', 'two'), ('dictionary', 'one')])
        self.assertEqual(self.examples['There is some milk in the fridge.'][0]['object'], 'milk')
        for text in ['There are a dictionary and two pencils on the table.',
                     'There is two pencils and a dictionary on the table.',
                     'There are some butter in the fridge.', 'There is butter in the glass.']:
            self.assertNotIn(text, self.examples)

    def test_existential_quantity_negation_and_conditional_questions_keep_scope(self):
        fact = self.examples['There was too little money in my purse.'][0]
        self.assertEqual((fact['quantity'], fact['polarity']), ('too little', 'positive'))
        fact = self.examples["There isn't any butter in the fridge."][0]
        self.assertEqual((fact['object'], fact['quantity'], fact['polarity']), ('butter', 'any', 'negative'))
        facts = self.examples['How many stamps will there be in your collection if you buy this one?']
        self.assertEqual((facts[0]['status'], facts[0]['condition'], facts[1]['status']),
                         ('questioned', 'if you buy this one', 'hypothetical'))
        self.assertEqual(facts[1]['object'], 'this one')
        self.assertEqual(self.examples["There won't be many."][0]['object'], 'unknown')

    def test_how_what_exclamations_are_not_questions_with_unknown_answers(self):
        for text, complement in [('How brave he is!', 'brave'),
                                 ('What an easy question it is!', 'easy question'),
                                 ('What hot coffee it is!', 'hot coffee')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['complement'], fact['status'], fact['clause_type']),
                             (complement, 'asserted', 'exclamative'))
        fact = self.examples['How carefully he drives!'][0]
        self.assertEqual(fact['manner'], 'carefully')
        self.assertNotIn('complement', fact)
        fact = self.examples['Look how expensively the man is dressed!'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['voice'], fact['manner']),
                         ('unknown', 'man', 'passive', 'expensively'))

    def test_exclamative_slots_preserve_articles_countability_and_adverb_forms(self):
        self.assertEqual(self.examples['It is such an easy question!'][0]['complement'], 'easy question')
        self.assertEqual(self.examples['What a beautiful carpet it is!'][0]['complement'], 'beautiful carpet')
        self.assertEqual(self.examples['It was such important information!'][0]['complement'], 'important information')
        for text in ['It is such a easy question!', 'What an beautiful carpet it is!',
                     'It was such an important information!', 'They are such a friendly people!',
                     'How careful he drives!', 'He is so nervously!']:
            self.assertNotIn(text, self.examples)

    def test_present_simple_number_and_regular_sequence_keep_each_action(self):
        self.assertEqual(self.examples['This student studies hard.'][0]['action'], 'study')
        self.assertEqual(self.examples['My children speak English fluently.'][0]['actor'], 'my children')
        self.assertEqual(self.examples['Mr. Brown learns History.'][0]['action'], 'learn')
        for text in ['This student study hard.', 'My children studies hard.',
                     'My child studys hard.', 'Mr. Brown teachs History.']:
            self.assertNotIn(text, self.examples)
        facts = self.examples['Every day my father gets up at 7, does his morning exercises, takes a shower, has breakfast and goes to work.']
        self.assertEqual([f['action'] for f in facts], ['get up', 'do', 'take', 'have', 'go'])
        self.assertTrue(all(f['actor'] == 'my father' and f['aspect'] == 'simple' for f in facts))
        self.assertEqual(facts[0]['time'], 'at 7')
        self.assertNotIn('time', facts[-1])

    def test_content_correction_does_not_silently_negate_the_source_statement(self):
        positive = self.examples['Dolphins live in forests.'][0]
        negative = self.examples["Dolphins don't live in forests."][0]
        self.assertEqual((positive['polarity'], negative['polarity']), ('positive', 'negative'))
        self.assertEqual((positive['action'], negative['action']), ('live', 'live'))
        self.assertEqual(self.examples['It melts in spring.'][0]['actor'], 'it')
        self.assertEqual(self.examples['No, I seldom quarrel with my friends.'][0]['polarity'], 'positive')
        self.assertEqual(self.examples['No, I never have nightmares.'][0]['polarity'], 'negative')

    def test_emphasis_and_body_part_possession_keep_self_forms_and_objects_separate(self):
        fact = self.examples['She usually makes breakfast herself.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['emphasis'], fact['reflexive']),
                         ('she', 'breakfast', 'actor', 'no'))
        fact = self.examples['We are washing our hands.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['possessor'], fact['reflexive']),
                         ('we', 'our hands', 'we', 'no'))
        for text in ['She usually make breakfast himself.', 'We are washing my hands.',
                     'He am washing his hands.']:
            self.assertNotIn(text, self.examples)

    def test_continuous_templates_preserve_agreement_and_distinguish_plans_and_intentions(self):
        fact = self.examples['He is translating a letter.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['aspect']),
                         ('he', 'translate', 'letter', 'continuous'))
        self.assertEqual(self.examples['Am I reading now?'][0]['status'], 'questioned')
        fact = self.examples['We are moving to a new flat at the weekend.'][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['status']), ('present', 'continuous', 'planned'))
        fact = self.examples['I am going to a fancy party.'][0]
        self.assertEqual((fact['action'], fact['destination'], fact['operation']), ('go', 'fancy party', 'none'))
        fact = self.examples['They are going to build a new road here.'][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation']), ('build', 'build', 'add'))
        self.assertEqual(self.examples['You are always coming late.'][0]['attitude'], 'annoyance')
        self.assertEqual(self.examples['He is having a break.'][0]['object'], 'break')
        facts = self.examples["Perhaps you can help me if you aren't doing anything special."]
        self.assertEqual((facts[0]['status'], facts[1]['status'], facts[1]['polarity']),
                         ('conditional', 'hypothetical', 'negative'))
        for text in ['He are translating a letter.', 'He is translating an letter.',
                     'We is translating an exercise.', 'He is having a car.']:
            self.assertNotIn(text, self.examples)

    def test_lessons_36_40_rewrites_and_parenthetical_alternatives_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u36-e3-05-task', 'u36-e3-05'), ('u37-t09', 'u37-t10'),
                              ('u37-e3-18a', 'u37-e3-18b'), ('u39-e2-03-task', 'u39-e2-03'),
                              ('u39-e3-02-task', 'u39-e3-02-negative'),
                              ('u39-e4-example-usually', 'u39-e4-example-always'),
                              ('u40-e2-03a', 'u40-e2-03b')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_sensory_states_and_actions_have_different_roles_and_aspects(self):
        state = self.examples['The soup tastes delicious!'][0]
        action = self.examples['Why are you tasting the soup?'][0]
        self.assertEqual((state['actor'], state['object'], state['complement'], state['aspect']),
                         ('soup', 'none', 'delicious', 'simple'))
        self.assertEqual((action['actor'], action['object'], action['aspect'], action['status']),
                         ('addressee', 'soup', 'continuous', 'questioned'))
        self.assertNotIn('complement', action)
        self.assertEqual(self.examples['What are you thinking about?'][0]['topic'], 'unknown')
        for text in ['He write you letters every week.', 'They is writing a letter to you right now.',
                     'The soup smells deliciously!', 'The meat is smelling wonderful!']:
            self.assertNotIn(text, self.examples)

    def test_new_relative_and_negative_clauses_keep_local_scope(self):
        facts = self.examples['The picture you are looking at belongs to the Tretyakov Gallery.']
        self.assertEqual((facts[0]['actor'], facts[0]['owner'], facts[1]['target']),
                         ('picture', 'Tretyakov Gallery', 'picture'))
        facts = self.examples['You never listen to what I say.']
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        self.assertEqual(facts[1]['actor'], 'user')
        fact = self.examples['He has a temperature and he is hardly breathing!'][1]
        self.assertEqual((fact['aspect'], fact['degree'], fact['polarity']),
                         ('continuous', 'hardly', 'positive'))

    def test_schedules_arrangements_intentions_and_weather_predictions_remain_distinct(self):
        fact = self.examples['The Berlin train arrives in Paris at 7 p. m. sharp.'][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['status']), ('present', 'simple', 'scheduled'))
        fact = self.examples['She is moving to a new flat next week.'][0]
        self.assertEqual((fact['actor'], fact['aspect'], fact['status']), ('she', 'continuous', 'planned'))
        fact = self.examples["I'm going to make a cake for tea."][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation']), ('make', 'make', 'add'))
        fact = self.examples["It's going to rain."][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation'], fact['status']),
                         ('rain', 'none', 'none', 'predicted'))
        fact = self.examples["I'm not going to prove anything to them."][0]
        self.assertEqual((fact['polarity'], fact['intention'], fact['operation']), ('negative', 'none', 'none'))

    def test_future_conditions_do_not_assert_the_condition_or_reverse_unless(self):
        facts = self.examples["You'll get this job if you learn Chinese."]
        self.assertEqual((facts[0]['tense'], facts[0]['status'], facts[0]['condition']),
                         ('future', 'conditional', 'if you learn Chinese'))
        self.assertEqual((facts[1]['object'], facts[1]['tense'], facts[1]['status']),
                         ('Chinese', 'present', 'hypothetical'))
        facts = self.examples["I won't continue the conversation unless you calm down."]
        self.assertEqual(facts[0]['condition'], 'unless you calm down')
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        self.assertEqual(facts[1]['status'], 'hypothetical')
        self.assertNotIn("You'll get this job if you will learn Chinese.", self.examples)

    def test_indirect_future_questions_and_time_clauses_keep_different_tenses(self):
        facts = self.examples["I don't know when Mrs. Smith will recover."]
        self.assertEqual((facts[0]['polarity'], facts[1]['status'], facts[1]['tense']),
                         ('negative', 'questioned', 'future'))
        facts = self.examples['Everything will be ready before Frank comes.']
        self.assertEqual((facts[0]['time'], facts[1]['tense'], facts[1]['status']),
                         ('before Frank comes', 'present', 'hypothetical'))
        facts = self.examples['I will return you the book as soon as I finish it.']
        self.assertEqual((facts[0]['recipient'], facts[1]['object']), ('addressee', 'book'))
        facts = self.examples["While it is raining I'll read or watch TV."]
        self.assertEqual(facts[0]['aspect'], 'continuous')
        self.assertEqual([f['coordination'] for f in facts[1:]], ['or', 'or'])

    def test_future_and_past_sequences_preserve_each_predicate_and_its_own_roles(self):
        facts = self.examples["I'll arrive at about six o'clock, leave my things at the hotel, and then take a walk."]
        self.assertEqual([f['action'] for f in facts], ['arrive', 'leave', 'take'])
        self.assertTrue(all(f['tense'] == 'future' for f in facts))
        self.assertEqual(facts[1]['location'], 'at the hotel')
        self.assertNotIn('location', facts[2])
        facts = self.examples['On Sunday morning she did the shopping, cleaned the flat and made dinner for the whole family.']
        self.assertEqual([f['action'] for f in facts], ['do', 'clean', 'make'])
        self.assertTrue(all(f['time'] == 'on Sunday morning' and f['tense'] == 'past' for f in facts))
        self.assertEqual(facts[2]['beneficiary'], 'whole family')

    def test_past_questions_keep_recipient_companion_and_actor_separate(self):
        fact = self.examples['Who did you write a letter to in the morning?'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['recipient']),
                         ('addressee', 'write', 'letter', 'unknown'))
        self.assertEqual(self.examples['Who broke that cup the other day?'][0]['actor'], 'unknown')
        self.assertEqual(self.examples['Who did Nelly dance with at the party?'][0]['companion'], 'unknown')
        fact = self.examples['When was he born?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice']), ('unknown', 'he', 'passive'))

    def test_used_to_and_would_are_past_habits_not_new_plans(self):
        fact = self.examples['He used to be more friendly.'][0]
        self.assertEqual((fact['tense'], fact['status'], fact['modality']), ('past', 'habitual', 'used to'))
        fact = self.examples['We would write to each other every week.'][0]
        self.assertEqual((fact['actor'], fact['recipient'], fact['reciprocal'], fact['reflexive']),
                         ('we', 'we', 'yes', 'no'))
        self.assertEqual((fact['intention'], fact['operation'], fact['status']), ('none', 'none', 'habitual'))
        self.assertNotIn('He would write to each other every week.', self.examples)

    def test_past_and_perfect_bindings_preserve_auxiliaries_and_irregular_forms(self):
        fact = self.examples['She has bought a computer.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['aspect']), ('she', 'buy', 'perfect'))
        self.assertEqual(self.examples['Has she finished the picture?'][0]['status'], 'questioned')
        self.assertEqual(self.examples['They fell ill yesterday.'][0]['tense'], 'past')
        self.assertEqual(self.examples["She didn't fall ill yesterday."][0]['polarity'], 'negative')
        self.assertEqual(self.examples['Did they fall ill the other day?'][0]['status'], 'questioned')
        for text in ['She have bought a computer.', 'Has she finish the picture?',
                     'Did they fell ill yesterday?', "She didn't saw a film at the cinema yesterday.",
                     'He shall finish the work in a few days.']:
            self.assertNotIn(text, self.examples)

    def test_perfect_visit_location_and_unfinished_time_periods_stay_separate(self):
        fact = self.examples['Have you ever been to the Historic Museum?'][0]
        self.assertEqual((fact['destination'], fact['aspect'], fact['status']),
                         ('Historic Museum', 'perfect', 'questioned'))
        fact = self.examples['They have been in Paris since Monday.'][0]
        self.assertEqual((fact['location'], fact['time']), ('in Paris', 'since Monday'))
        self.assertNotIn('destination', fact)
        self.assertEqual(self.examples["He's visited his grandparents this summer."][0]['actor'], 'he')
        for text in ['Have you ever been to Historic Museum?', 'Have you ever been to the Greece?',
                     "He's visited his grandparents yesterday.", 'My friend has caught cold last night.']:
            self.assertNotIn(text, self.examples)

    def test_perfect_negation_applies_to_the_correct_event_and_ordinal(self):
        facts = self.examples["It's not the first time the cat has torn my curtains!"]
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        self.assertEqual((facts[1]['action'], facts[1]['aspect']), ('tear', 'perfect'))
        fact = self.examples["No, I've never been to the Zoo."][0]
        self.assertEqual((fact['polarity'], fact['frequency']), ('negative', 'ever'))
        facts = self.examples["This is the funniest joke I've ever heard."]
        self.assertEqual((facts[0]['complement'], facts[1]['object'], facts[1]['action']),
                         ('funniest joke', 'funniest joke', 'hear'))
        fact = self.examples['Has he ever broken his arm?'][0]
        self.assertEqual((fact['object'], fact['possessor'], fact['reflexive']), ('his arm', 'he', 'no'))
        self.assertNotIn('Has he ever broken your arm?', self.examples)

    def test_perfect_replies_keep_unresolved_references_local_to_the_sentence(self):
        self.assertEqual(self.examples["Yes, I've just finished it."][0]['object'], 'it')
        self.assertEqual(self.examples["I've washed it."][0]['object'], 'it')
        self.assertEqual(self.examples['Have you?'][0]['action'], 'unknown')
        facts = self.examples['It is, and it\'s getting colder and colder.']
        self.assertEqual((facts[0]['action'], facts[1]['action']), ('unknown', 'get'))
        facts = self.examples["I have heard a lot about this book but I haven't read it so far."]
        self.assertEqual(facts[1]['object'], 'this book')

    def test_trials_and_other_peoples_desires_do_not_assert_completion(self):
        facts = self.examples['Have you ever tried speaking English?']
        self.assertEqual((facts[0]['status'], facts[1]['status']), ('questioned', 'hypothetical'))
        facts = self.examples["No, I've never tried speaking Chinese."]
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('negative', 'positive'))
        self.assertEqual(facts[1]['status'], 'hypothetical')
        facts = self.examples['Yes, I am, and he wants me to sign a new contract.']
        self.assertEqual((facts[0]['action'], facts[2]['actor'], facts[2]['status']),
                         ('unknown', 'user', 'intended'))
        self.assertEqual(facts[2]['operation'], 'none')

    def test_lessons_41_45_rewrites_and_expanded_alternatives_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u41-e2-04b', 'u41-e2-04c'), ('u41-e2-05d', 'u41-e2-05e'),
                              ('u42-t01a', 'u42-t01b'), ('u42-e3-15a', 'u42-e3-15b'),
                              ('u44-t01', 'u44-t03'), ('u44-e2-02a', 'u44-e2-02b'),
                              ('u44-e3-03-task', 'u44-e3-03'), ('u45-e1-b01q', 'u45-e1-b01a'),
                              ('u45-e2-05q', 'u45-e2-05a'), ('u45-e6-a05a', 'u45-e6-a05b'),
                              ('u45-e6-b10a', 'u45-e6-b10b')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_since_starting_points_and_for_durations_are_separate_bindings(self):
        point = self.examples["Mr. Brown hasn't been at work since last week."][0]
        duration = self.examples["Mr. Brown hasn't been at work for the last week."][0]
        self.assertEqual(point['time'], 'since last week')
        self.assertNotIn('duration', point)
        self.assertEqual(duration['duration'], 'the last week')
        self.assertNotIn('time', duration)
        self.assertEqual(self.examples["I've known Ben for five years."][0]['duration'], 'five years')
        for text in ["I've known Ben since five years.", "Mr. Brown hasn't been at work for last week.",
                     "He hasn't called me since ages.", "I've known Ben for September."]:
            self.assertNotIn(text, self.examples)

    def test_since_clauses_keep_past_events_positive_under_a_negative_main_clause(self):
        facts = self.examples["He hasn't called me since he moved to a new flat."]
        self.assertEqual([(f['tense'], f['aspect'], f['polarity']) for f in facts],
                         [('present', 'perfect', 'negative'), ('past', 'simple', 'positive')])
        self.assertEqual(facts[1]['status'], 'asserted')
        facts = self.examples['George has been the best student since he entered the University.']
        self.assertEqual(facts[1]['actor'], 'George')
        facts = self.examples["On Monday the weather got worse and my Grandpa hasn't been out ever since."]
        self.assertEqual(facts[1]['time'], 'ever since the weather got worse')

    def test_state_verbs_stay_perfect_while_ongoing_activities_are_perfect_continuous(self):
        state = self.examples['How long has Mary known it?'][0]
        activity = self.examples['How long has Mary been reading it?'][0]
        self.assertEqual((state['actor'], state['aspect'], state['duration']), ('Mary', 'perfect', 'unknown'))
        self.assertEqual((activity['actor'], activity['aspect'], activity['duration']),
                         ('Mary', 'perfect_continuous', 'unknown'))
        self.assertEqual(self.examples['Since when has Mary been working here?'][0]['time'], 'unknown')
        self.assertNotIn('duration', self.examples['Since when has Mary been working here?'][0])
        for text in ['How long has Mary been knowing it?', 'How long have Mary known it?',
                     'Since when has Mary been living this flat?']:
            self.assertNotIn(text, self.examples)

    def test_perfect_continuous_activity_does_not_claim_a_completed_result(self):
        facts = self.examples['My Dad has been trying to fix the bicycle since morning.']
        self.assertEqual((facts[0]['action'], facts[0]['aspect']), ('try', 'perfect_continuous'))
        self.assertEqual((facts[1]['action'], facts[1]['status']), ('fix', 'hypothetical'))
        self.assertTrue(all(f['operation'] == 'none' for f in facts))
        fact = self.examples['She has been crying.'][0]
        self.assertEqual(fact['actor'], 'she')
        self.assertNotIn('reason', fact)
        fact = self.examples['Mary has been learning Chinese for several months.'][0]
        self.assertEqual((fact['action'], fact['object'], fact['duration']), ('learn', 'Chinese', 'several months'))

    def test_result_counts_and_process_durations_change_the_requested_role(self):
        result = self.examples['Mr. Priestly has planted ten flowers today.'][0]
        process = self.examples['Mr. Priestly has been planting flowers all day today.'][0]
        self.assertEqual((result['aspect'], result['quantity']), ('perfect', 'ten'))
        self.assertEqual((process['aspect'], process['duration']), ('perfect_continuous', 'all day'))
        self.assertNotIn('quantity', process)
        result = self.examples['How many pages of this book has he read?'][0]
        process = self.examples['How long has he been reading this book?'][0]
        self.assertEqual((result['object'], result['quantity']), ('pages of this book', 'unknown'))
        self.assertEqual((process['object'], process['duration']), ('this book', 'unknown'))

    def test_explicit_negative_perfect_continuous_is_preserved_when_present_in_book(self):
        fact = self.examples["I've been doing nothing for the last two days."][0]
        self.assertEqual((fact['action'], fact['object'], fact['polarity'], fact['aspect']),
                         ('do', 'anything', 'negative', 'perfect_continuous'))
        self.assertEqual(self.examples["I've done nothing for the last two days."][0]['aspect'], 'perfect')
        facts = self.examples["This plant hasn't grown since the weather changed."]
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        self.assertEqual(facts[0]['aspect'], 'perfect')

    def test_new_pronoun_links_use_only_the_current_sentence(self):
        facts = self.examples["I've been waiting for Pete's call since 3 o'clock but he hasn't called so far."]
        self.assertEqual((facts[0]['target'], facts[1]['actor']), ("Pete's call", 'Pete'))
        facts = self.examples["Nick has been saving money for a new player since he saw the advertisement, but he hasn't saved enough so far."]
        self.assertEqual([f['actor'] for f in facts], ['Nick', 'Nick', 'Nick'])
        self.assertEqual((facts[-1]['object'], facts[-1]['quantity']), ('money', 'enough'))
        self.assertEqual(self.examples['How long has she been cleaning it?'][0]['object'], 'it')
        self.assertEqual(self.examples['But I was.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['Then he gave it up and left for India.'][0]['object'], 'it')

    def test_reciprocal_and_body_part_bindings_preserve_participants(self):
        fact = self.examples['The children have been shouting at each other all morning.'][0]
        self.assertEqual((fact['actor'], fact['target'], fact['reciprocal'], fact['reflexive']),
                         ('children', 'children', 'yes', 'no'))
        facts = self.examples['Mary was constantly biting her nails, which is a most annoying habit!']
        self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[0]['possessor']),
                         ('Mary', 'her nails', 'Mary'))
        self.assertEqual((facts[0]['reflexive'], facts[0]['attitude']), ('no', 'annoyance'))
        self.assertEqual(facts[1]['actor'], 'biting her nails')
        self.assertNotIn('Mary was constantly biting his nails, which is a most annoying habit!', self.examples)

    def test_past_duration_does_not_force_perfect_and_today_does_not_force_present(self):
        facts = self.examples['She worked at school for five years and then quit.']
        self.assertTrue(all(f['tense'] == 'past' and f['aspect'] == 'simple' for f in facts))
        self.assertEqual(facts[0]['duration'], 'five years')
        self.assertEqual(self.examples['I read the book yesterday.'][0]['tense'], 'past')
        self.assertEqual(self.examples["I've been reading the magazine for a long time."][0]['aspect'], 'perfect_continuous')
        self.assertEqual(self.examples['I saw Nick before classes today.'][0]['tense'], 'past')
        self.assertEqual(self.examples['She got up very early this morning.'][0]['aspect'], 'simple')
        self.assertNotIn("I've been reading the magazine long ago.", self.examples)

    def test_past_background_and_interrupting_event_keep_different_aspects(self):
        facts = self.examples['Nick was walking down the street when he saw Joan.']
        self.assertEqual([f['actor'] for f in facts], ['Nick', 'Nick'])
        self.assertEqual([(f['tense'], f['aspect']) for f in facts], [('past', 'continuous'), ('past', 'simple')])
        facts = self.examples["I wasn't doing anything special when he called."]
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        fact = self.examples['At this time yesterday they were watching TV.'][0]
        self.assertEqual((fact['actor'], fact['time'], fact['aspect']), ('they', 'at this time yesterday', 'continuous'))
        for text in ['At this time yesterday they was watching TV.',
                     "What was you doing at 6 o'clock yesterday?", 'I were walking my dog.']:
            self.assertNotIn(text, self.examples)

    def test_past_intentions_and_arrangements_do_not_add_current_intentions(self):
        facts = self.examples['I was going to congratulate her but it turned out that she was away.']
        self.assertEqual((facts[0]['action'], facts[0]['tense'], facts[0]['status']), ('congratulate', 'past', 'intended'))
        self.assertEqual((facts[0]['intention'], facts[0]['operation']), ('none', 'none'))
        fact = self.examples['Mr. Dowly was coming a few minutes later.'][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['status']), ('past', 'continuous', 'planned'))
        facts = self.examples['Mum prepared everything necessary for the party we were giving on Saturday.']
        self.assertEqual((facts[0]['status'], facts[1]['status']), ('asserted', 'planned'))

    def test_reported_past_actions_and_unrealized_modal_events_are_not_flattened(self):
        facts = self.examples['As far as I remember, you were doing crossword puzzles and I was knitting.']
        self.assertEqual([f['action'] for f in facts], ['remember', 'do', 'knit'])
        self.assertEqual([f['status'] for f in facts[1:]], ['reported', 'reported'])
        fact = self.examples['Good Lord, it could have caught fire, then!'][0]
        self.assertEqual((fact['modality'], fact['status'], fact['operation']), ('could', 'possible', 'none'))
        facts = self.examples['I asked the nurse who was going upstairs to take the papers to Dr. Reed.']
        self.assertEqual([f['action'] for f in facts], ['ask', 'go', 'take'])
        self.assertEqual((facts[2]['actor'], facts[2]['recipient'], facts[2]['status']), ('nurse', 'Dr. Reed', 'required'))

    def test_quantifier_negation_and_prepositions_keep_semantic_roles(self):
        fact = self.examples['But perhaps not everybody will be able to come.'][0]
        self.assertEqual((fact['actor'], fact['negation_scope'], fact['certainty']),
                         ('everybody', 'not everybody', 'perhaps'))
        self.assertEqual(self.examples['Nobody has heard from him ever since.'][0]['actor'], 'anybody')
        fact = self.examples['How much did you pay for this picture?'][0]
        self.assertEqual((fact['object'], fact['amount'], fact['purpose']), ('money', 'unknown', 'this picture'))
        fact = self.examples['He has been concealing the truth from us all this time!'][0]
        self.assertEqual(fact['target'], 'us')
        self.assertNotIn('source', fact)

    def test_lessons_46_50_rewrites_contrasts_and_printed_alternatives_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u46-t08a', 'u46-t08b'), ('u46-e2-01-task', 'u46-e2-01'),
                              ('u46-e4-04-task', 'u46-e4-04'), ('u47-e2-example-task', 'u47-e2-example'),
                              ('u47-e3-06a', 'u47-e3-06b'), ('u48-t01', 'u48-t02'),
                              ('u48-e2-07a', 'u48-e2-07b'), ('u49-e1-a01', 'u49-e1-a02'),
                              ('u49-e2-c01a', 'u49-e2-c01b'), ('u49-e3-c06a', 'u49-e3-c06b'),
                              ('u50-e1-03a', 'u50-e1-03b'), ('u50-e1-07a', 'u50-e1-07b')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_new_past_background_bindings_and_participles_preserve_agreement(self):
        simple = self.examples["At 7 o'clock last night they returned home."][0]
        ongoing = self.examples["At 7 o'clock last night they were returning home."][0]
        self.assertEqual((simple['actor'], simple['aspect']), ('they', 'simple'))
        self.assertEqual((ongoing['actor'], ongoing['aspect']), ('they', 'continuous'))
        facts = self.examples['We sat reading a magazine.']
        self.assertEqual([f['action'] for f in facts], ['sit', 'read'])
        self.assertEqual((facts[1]['tense'], facts[1]['form']), ('nonfinite', 'participle'))
        self.assertNotIn("At 7 o'clock last night they was returning home.", self.examples)

    def test_lexical_had_and_perfect_auxiliary_do_not_share_an_aspect(self):
        facts = self.examples['She had dinner and took out the book she had bought on her way home.']
        self.assertEqual([(f['action'], f['aspect']) for f in facts],
                         [('have', 'simple'), ('take out', 'simple'), ('buy', 'perfect')])
        self.assertTrue(all(f['actor'] == 'she' and f['tense'] == 'past' for f in facts))
        fact = self.examples['I had had a headache since morning.'][0]
        self.assertEqual((fact['action'], fact['aspect'], fact['time']), ('have', 'perfect', 'since morning'))
        facts = self.examples['By the time he was 30 he already had two professions and had changed several jobs.']
        self.assertEqual([f['aspect'] for f in facts], ['simple', 'simple', 'perfect'])

    def test_past_perfect_deadlines_are_not_durations(self):
        fact = self.examples["We had written the article by 4 o'clock."][0]
        self.assertEqual((fact['actor'], fact['object'], fact['tense'], fact['aspect']),
                         ('we', 'article', 'past', 'perfect'))
        self.assertEqual(fact['time'], "by 4 o'clock")
        self.assertNotIn('duration', fact)
        self.assertNotIn("We had written the article for 4 o'clock.", self.examples)

    def test_new_past_reciprocals_and_body_parts_keep_participants_bound(self):
        fact = self.examples['We had known each other for ages.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['reciprocal'], fact['reflexive']),
                         ('we', 'we', 'yes', 'no'))
        fact = self.examples['She hurt her leg when she fell off the ladder.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['possessor'], fact['reflexive']),
                         ('she', 'her leg', 'she', 'no'))
        fact = self.examples['He combed his hair and looked in the mirror.'][0]
        self.assertEqual((fact['object'], fact['possessor'], fact['reflexive']), ('his hair', 'he', 'no'))
        for text in ['He had known each other for ages.', 'She hurt his leg when she fell off the ladder.',
                     'He combed her hair and looked in the mirror.']:
            self.assertNotIn(text, self.examples)

    def test_hardly_and_no_sooner_mark_immediate_sequence_without_negating_events(self):
        facts = self.examples['Hardly had I finished the exercise when they gave me another one.']
        self.assertEqual([f['polarity'] for f in facts], ['positive', 'positive'])
        self.assertEqual((facts[0]['aspect'], facts[0]['temporal_relation']), ('perfect', 'immediately before'))
        self.assertEqual((facts[1]['object'], facts[1]['recipient']), ('another exercise', 'user'))
        self.assertEqual(facts, self.examples['I had hardly finished the exercise when they gave me another one.'])
        facts = self.examples['No sooner had the children begun to play than they quarrelled.']
        self.assertTrue(all(f['polarity'] == 'positive' for f in facts))
        self.assertEqual((facts[0]['temporal_relation'], facts[-1]['action']), ('immediately before', 'quarrel'))

    def test_arrival_contrasts_preserve_question_and_event_aspects(self):
        for text, aspect in [('When Joan came we discussed the news.', 'simple'),
                             ('When Joan came we were discussing the news.', 'continuous'),
                             ('By the time Joan came we had already discussed the news.', 'perfect'),
                             ("When Joan came we'd been discussing the news for half an hour already.", 'perfect_continuous')]:
            facts = self.examples[text]
            self.assertEqual((facts[0]['actor'], facts[1]['aspect']), ('Joan', aspect))
        facts = self.examples['What were you discussing when Maggie came?']
        self.assertEqual((facts[0]['object'], facts[0]['status']), ('unknown', 'questioned'))
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('Maggie', 'asserted'))

    def test_painting_process_result_and_passive_patient_stay_distinct(self):
        for text, aspect in [('They painted the flat for three hours.', 'simple'),
                             ('They had been painting the flat for three hours when it suddenly began raining.', 'perfect_continuous'),
                             ('When they were painting the flat, it suddenly began raining.', 'continuous'),
                             ('When they had painted the flat, it suddenly began raining.', 'perfect')]:
            self.assertEqual(self.examples[text][0]['aspect'], aspect)
        fact = self.examples['They had already painted the flat twice when it suddenly began raining.'][0]
        self.assertEqual((fact['aspect'], fact['frequency']), ('perfect', 'twice'))
        self.assertNotIn('duration', fact)
        facts = self.examples['The flat they had been painting so long was burnt down at night.']
        self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[0]['voice']), ('unknown', 'flat', 'passive'))
        self.assertEqual((facts[1]['actor'], facts[1]['object'], facts[1]['aspect']), ('they', 'flat', 'perfect_continuous'))

    def test_past_perfect_stative_since_and_process_duration_remain_separate(self):
        facts = self.examples['By the time the firemen arrived, the house had smelt of smoke for a long time.']
        self.assertEqual((facts[1]['action'], facts[1]['aspect'], facts[1]['duration']), ('smell', 'perfect', 'a long time'))
        facts = self.examples['At midday he felt tired as he had been working in the garden since early morning.']
        self.assertEqual((facts[1]['aspect'], facts[1]['time']), ('perfect_continuous', 'since early morning'))
        self.assertNotIn('duration', facts[1])
        facts = self.examples['I had been driving for five hours when the accident happened.']
        self.assertEqual((facts[0]['duration'], facts[1]['aspect']), ('five hours', 'simple'))

    def test_past_question_counts_do_not_turn_into_process_durations(self):
        process = self.examples['She had been asking me questions all evening!'][0]
        result = self.examples['She had asked me twenty questions just for the last half an hour.'][0]
        self.assertEqual((process['aspect'], process['duration']), ('perfect_continuous', 'all evening'))
        self.assertNotIn('quantity', process)
        self.assertEqual((result['aspect'], result['quantity'], result['recipient']), ('perfect', 'twenty', 'user'))
        self.assertEqual(result['duration'], 'the last half an hour')

    def test_new_story_references_do_not_cross_sentence_boundaries(self):
        facts = self.examples['She had washed the curtains and was hanging them.']
        self.assertEqual([f['object'] for f in facts], ['curtains', 'curtains'])
        facts = self.examples['Kate planted a flower and put it on the window sill.']
        self.assertEqual([f['object'] for f in facts], ['flower', 'flower'])
        facts = self.examples['The next day it died as she had forgotten to water it.']
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[2]['object']), ('it', 'she', 'it'))
        self.assertEqual(self.examples['Frank met her at the station.'][0]['object'], 'her')
        self.assertEqual(self.examples['He was buying a new one.'][0]['object'], 'new one')
        self.assertEqual(self.examples['But I shall.'][0]['action'], 'unknown')
        self.assertEqual(self.examples['Perhaps I can here.'][0]['action'], 'unknown')
        fact = self.examples['He was quietly playing with his toys.'][0]
        self.assertEqual(fact['instrument'], 'his toys')
        self.assertNotIn('companion', fact)

    def test_reported_quotes_bind_speaker_and_explicit_addressee_locally(self):
        facts = self.examples['“You have never shown me your room, Betty,” he said.']
        self.assertEqual((facts[0]['actor'], facts[0]['action']), ('he', 'say'))
        self.assertEqual((facts[1]['actor'], facts[1]['object'], facts[1]['recipient']), ('Betty', "Betty's room", 'he'))
        self.assertEqual((facts[1]['status'], facts[1]['polarity']), ('reported', 'negative'))
        facts = self.examples['“You are very thin, darling,” she said.']
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('unknown', 'reported'))

    def test_review_past_desires_do_not_add_the_current_intention(self):
        fact = self.examples['I want to paint.'][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation']), ('paint', 'paint', 'add'))
        facts = self.examples['I rather wanted to be a painter when I was a boy, but my father made me go into business because he said there was no money in art.']
        self.assertTrue(all(f['operation'] == 'none' for f in facts))
        self.assertEqual((facts[1]['status'], facts[-1]['polarity'], facts[-1]['status']), ('intended', 'negative', 'reported'))
        facts = self.examples['I was going to tell them about my new job but they already knew everything and were just discussing it when I came.']
        self.assertEqual((facts[0]['status'], facts[0]['operation']), ('intended', 'none'))
        self.assertEqual(facts[2]['object'], 'my new job')

    def test_review_condition_and_correlative_comparison_keep_scope(self):
        facts = self.examples["Tom thinks that even if he brings all of us home, his Mum won't get angry."]
        self.assertEqual((facts[1]['actor'], facts[1]['status'], facts[1]['polarity']), ('Tom', 'hypothetical', 'positive'))
        self.assertEqual((facts[2]['actor'], facts[2]['status'], facts[2]['polarity']), ("Tom's Mum", 'conditional', 'negative'))
        facts = self.examples["For my part I find the longer I know people the more they puzzle me: my oldest friends are those of whom I can say that I don't know the first thing about them."]
        self.assertEqual((facts[1]['status'], facts[2]['status']), ('conditional', 'conditional'))
        self.assertEqual((facts[2]['actor'], facts[2]['condition']), ('people', 'as I know people longer'))
        self.assertEqual(facts[-1]['topic'], 'my oldest friends')

    def test_lessons_51_55_contrasts_and_rewrites_share_split_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u51-t01', 'u51-t02'), ('u51-t08a', 'u51-t08b'),
                              ('u52-e3-07a', 'u52-e3-07b'), ('u53-t07a', 'u53-t07b'),
                              ('u53-e3-a01', 'u53-e3-a17'), ('u53-e3-b07a', 'u53-e3-b07b'),
                              ('u54-e1-01-task-a', 'u54-e1-01'), ('u54-e3-07', 'u54-e3-14'),
                              ('u55-e3-b02a', 'u55-e3-b02b'), ('u55-e3-c01', 'u55-e3-c02'),
                              ('u55-e3-e01', 'u55-e3-e02')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_direct_speech_binds_speaker_recipient_and_local_possession(self):
        facts = self.examples['Alice says, “Tim, you must help me with the shopping today.”']
        self.assertEqual((facts[0]['actor'], facts[0]['recipient']), ('Alice', 'Tim'))
        self.assertEqual((facts[1]['actor'], facts[1]['object'], facts[1]['status']), ('Tim', 'Alice', 'required'))
        facts = self.examples['Mary says that her Dad comes home very late.']
        self.assertEqual((facts[0]['actor'], facts[1]['actor']), ('Mary', "Mary's Dad"))
        facts = self.examples['He says, “My brother knows Spanish.”']
        self.assertEqual(facts[1]['actor'], 'his brother')
        self.assertEqual(facts[1]['status'], 'reported')

    def test_quoted_you_does_not_borrow_the_indirect_answers_addressee(self):
        direct = self.examples['She asks, “Have you brought the book?”'][1]
        indirect = self.examples['She asks if I have brought the book.'][1]
        self.assertEqual((direct['actor'], indirect['actor']), ('unknown', 'user'))
        self.assertEqual((direct['status'], indirect['status']), ('questioned', 'questioned'))
        facts = self.examples['I ask my sister, “Did you get bad marks at school?”']
        self.assertEqual(facts[1]['actor'], 'my sister')
        self.assertEqual(self.examples['“So am I,” the man said.'][1]['action'], 'unknown')

    def test_reported_commands_keep_negative_scope_and_bound_recipient(self):
        facts = self.examples['Mother told Lucy to clean her room.']
        self.assertEqual((facts[0]['recipient'], facts[1]['actor'], facts[1]['object']), ('Lucy', 'Lucy', "Lucy's room"))
        self.assertEqual((facts[0]['tense'], facts[1]['tense'], facts[1]['status']), ('past', 'nonfinite', 'required'))
        facts = self.examples["The shop assistant says, “Don't touch this button.”"]
        self.assertEqual([f['polarity'] for f in facts], ['positive', 'negative'])
        self.assertEqual((facts[1]['actor'], facts[1]['object']), ('unknown', 'this button'))
        self.assertTrue(all(f['operation'] == 'none' for f in facts))

    def test_indirect_questions_keep_the_missing_semantic_role(self):
        fact = self.examples['Helen wonders where they hid the money.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['location']), ('they', 'money', 'unknown'))
        fact = self.examples['The teacher asks the pupils which of them can translate this sentence.'][1]
        self.assertEqual((fact['actor'], fact['actor_group'], fact['status']), ('unknown', 'pupils', 'questioned'))
        fact = self.examples['My Mum always wonders who I went to the cinema with.'][1]
        self.assertEqual((fact['actor'], fact['companion']), ('user', 'unknown'))
        fact = self.examples['Susan asks when your brother was born.'][1]
        self.assertEqual((fact['object'], fact['time'], fact['voice']), ('your brother', 'unknown', 'passive'))

    def test_backshift_changes_tense_aspect_and_deictic_time_together(self):
        direct = self.examples['Mary said, “I saw Max yesterday.”'][1]
        indirect = self.examples['Mary said that she had seen Max the day before.'][1]
        self.assertEqual((direct['actor'], direct['aspect'], direct['time']), ('Mary', 'simple', 'yesterday'))
        self.assertEqual((indirect['actor'], indirect['aspect'], indirect['time']), ('Mary', 'perfect', 'the day before'))
        facts = self.examples['They said that they were working that night.']
        self.assertEqual((facts[1]['actor'], facts[1]['aspect'], facts[1]['status']), ('they', 'continuous', 'planned'))
        fact = self.examples['They said that they had been working at it that week.'][1]
        self.assertEqual((fact['tense'], fact['aspect'], fact['time']), ('past', 'perfect_continuous', 'that week'))
        self.assertNotIn('They said that they was working that night.', self.examples)

    def test_reported_future_is_not_habitual_would_or_a_current_intention(self):
        fact = self.examples['He said that he would call me the following day.'][1]
        self.assertEqual((fact['tense'], fact['modality'], fact['status']), ('future', 'would', 'reported'))
        self.assertEqual((fact['operation'], fact['intention']), ('none', 'none'))
        facts = self.examples['She said, “I want to see him now.”']
        self.assertTrue(all(f['operation'] == 'none' for f in facts))
        self.assertEqual(facts[-1]['status'], 'intended')
        facts = self.examples['He said that he was leaving but he promised to come the following week.']
        self.assertEqual((facts[1]['status'], facts[-1]['status']), ('planned', 'promised'))

    def test_backshift_exceptions_keep_general_facts_and_past_time_clauses(self):
        fact = self.examples['The lecturer explained that the Sun is a big star.'][1]
        self.assertEqual((fact['tense'], fact['status']), ('present', 'reported'))
        fact = self.examples['The book said that Einstein invented the theory of relativity.'][1]
        self.assertEqual((fact['tense'], fact['aspect']), ('past', 'simple'))
        facts = self.examples['My brother said that he had met Chris when he was walking along the street.']
        self.assertEqual([f['aspect'] for f in facts[1:]], ['perfect', 'continuous'])
        self.assertTrue(all(f['tense'] == 'past' for f in facts))

    def test_reported_conditionals_and_pretence_do_not_assert_the_content(self):
        facts = self.examples["The teacher said that if we didn't study hard we would have problems at the exam."]
        self.assertEqual((facts[1]['status'], facts[1]['polarity']), ('hypothetical', 'negative'))
        self.assertEqual((facts[2]['status'], facts[2]['polarity'], facts[2]['modality']), ('conditional', 'positive', 'would'))
        facts = self.examples["The Tysons had an impression that something strange was going on in the room but they pretended they didn't notice it or at least weren't paying attention to it."]
        self.assertEqual([f['status'] for f in facts[-2:]], ['pretended', 'pretended'])
        self.assertTrue(all(f['polarity'] == 'negative' for f in facts[-2:]))

    def test_review_same_sentence_ellipsis_resolves_without_crossing_stories(self):
        facts = self.examples['I wondered if Helen and Pete had been travelling so much and they said they had.']
        self.assertEqual((facts[-1]['actor'], facts[-1]['action'], facts[-1]['aspect']),
                         ('Helen and Pete', 'travel', 'perfect_continuous'))
        facts = self.examples['The boys told Jimmy not to cry and promised to help him.']
        self.assertEqual((facts[1]['actor'], facts[-1]['object']), ('Jimmy', 'Jimmy'))
        self.assertEqual((facts[1]['status'], facts[-1]['status']), ('required', 'promised'))
        self.assertEqual(self.examples['Yes, they were, thank you.'][0]['action'], 'unknown')
        self.assertEqual(self.examples["It's already being dried."][0]['object'], 'it')

    def test_passive_actor_patient_and_instrument_have_separate_roles(self):
        fact = self.examples['The letter was written by a boy with a pen.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['instrument'], fact['voice']), ('boy', 'letter', 'pen', 'passive'))
        active = self.examples['The commission discussed my plan yesterday.'][0]
        passive = self.examples['My plan was discussed yesterday by the commission.'][0]
        self.assertEqual((active['actor'], active['object']), (passive['actor'], passive['object']))
        self.assertEqual((active['voice'], passive['voice']), ('active', 'passive'))

    def test_agentless_passives_do_not_inherit_actors_from_active_exercises(self):
        fact = self.examples['A patient was being examined when I came.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['aspect']), ('unknown', 'patient', 'continuous'))
        self.assertEqual(self.examples['This criminal was caught a year ago.'][0]['actor'], 'unknown')
        fact = self.examples["My brother's car is still being repaired."][0]
        self.assertEqual((fact['actor'], fact['object']), ('unknown', "my brother's car"))
        facts = self.examples['Something is broken every time the dishes are washed by my daughter.']
        self.assertEqual([f['actor'] for f in facts], ['unknown', 'my daughter'])

    def test_passive_bindings_preserve_agreement_and_compatible_work(self):
        for text, tense, aspect in [('The curtains were washed on Monday.', 'past', 'simple'),
                                    ('The curtains are being washed now.', 'present', 'continuous'),
                                    ('The curtains have already been washed.', 'present', 'perfect'),
                                    ('The curtains had been washed by that time.', 'past', 'perfect')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['action'], fact['object'], fact['tense'], fact['aspect']), ('wash', 'curtains', tense, aspect))
        self.assertEqual(self.examples['These curtains are washed every other day.'][0]['object'], 'these curtains')
        for text in ['The curtains is being washed now.', 'The curtains has already been washed.',
                     'This curtains are washed every other day.', 'The curtains are being painted now.']:
            self.assertNotIn(text, self.examples)

    def test_passive_negation_questions_and_nonfinite_causes_keep_scope(self):
        fact = self.examples['No way out has been found yet.'][0]
        self.assertEqual((fact['object'], fact['polarity'], fact['aspect']), ('way out', 'negative', 'perfect'))
        fact = self.examples['Whose report is being discussed now?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['possessor'], fact['status']), ('unknown', 'report', 'unknown', 'questioned'))
        facts = self.examples['How many times has the mincer been used since it was bought?']
        self.assertEqual((facts[0]['frequency'], facts[0]['status'], facts[1]['status']), ('unknown', 'questioned', 'asserted'))
        facts = self.examples["The children weren't made to help her in the garden."]
        self.assertEqual((facts[0]['polarity'], facts[1]['status']), ('negative', 'hypothetical'))
        facts = self.examples["He won't be cured if he doesn't go to hospital."]
        self.assertEqual([f['status'] for f in facts], ['conditional', 'hypothetical'])

    def test_lessons_56_60_rewrites_and_active_passive_pairs_share_split_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u56-t06d', 'u56-t06'), ('u56-e1-b01a', 'u56-e1-b01b'),
                              ('u57-t03d', 'u57-t03'), ('u57-t07a', 'u57-t07d'),
                              ('u57-e1-e01d', 'u57-e1-e01b'), ('u58-e3-09a', 'u58-e3-09b'),
                              ('u59-e1-08a', 'u59-e1-08b'), ('u60-t01a', 'u60-t01b'),
                              ('u60-e1-a05-task', 'u60-e1-a05'), ('u60-e1-b07-task', 'u60-e1-b07')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_indirect_passives_keep_recipient_and_transferred_object_separate(self):
        first = self.examples['I was given really good advice.'][0]
        second = self.examples['Really good advice was given to me.'][0]
        self.assertEqual(first, second)
        self.assertEqual((first['actor'], first['object'], first['recipient']),
                         ('unknown', 'really good advice', 'user'))
        active = self.examples['They gave me really good advice.'][0]
        self.assertEqual((active['actor'], active['recipient']), ('they', 'user'))
        first = self.examples["Even a birthday card wasn't sent to me by her!"][0]
        second = self.examples["I wasn't even sent a birthday card by her!"][0]
        self.assertEqual(first, second)
        self.assertEqual((first['actor'], first['object'], first['recipient'], first['polarity']),
                         ('she', 'birthday card', 'user', 'negative'))

    def test_prepositional_passives_retain_topic_target_and_questioned_agent(self):
        fact = self.examples['Their holidays are being talked about.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['topic']), ('unknown', 'none', 'their holidays'))
        fact = self.examples['He is being waited for by everybody.'][0]
        self.assertEqual((fact['actor'], fact['target'], fact['voice']), ('everybody', 'he', 'passive'))
        fact = self.examples['Who was this doctor recommended to you by?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['recipient'], fact['status']),
                         ('unknown', 'this doctor', 'addressee', 'questioned'))
        fact = self.examples['The message was written with a pen.'][0]
        self.assertEqual((fact['actor'], fact['instrument']), ('unknown', 'pen'))

    def test_passive_review_preserves_aspect_and_does_not_invent_context(self):
        fact = self.examples['The question has been discussed for two hours already.'][0]
        self.assertEqual((fact['aspect'], fact['duration'], fact['voice']), ('perfect', 'two hours', 'passive'))
        fact = self.examples['I have been waited for for a quarter of an hour already.'][0]
        self.assertEqual((fact['actor'], fact['target'], fact['aspect']), ('unknown', 'user', 'perfect'))
        facts = self.examples['I have been kept waiting for half an hour!']
        self.assertEqual((facts[0]['action'], facts[0]['object'], facts[1]['action']), ('keep', 'user', 'wait'))
        self.assertEqual(self.examples['It is being tuned now.'][0]['object'], 'it')
        self.assertEqual(self.examples['It is spoken in many countries.'][0]['object'], 'it')

    def test_new_passive_bindings_agree_with_recipient_and_patient_number(self):
        fact = self.examples['They were given really good advice.'][0]
        self.assertEqual((fact['actor'], fact['recipient']), ('unknown', 'they'))
        fact = self.examples['She has been waited for for a quarter of an hour already.'][0]
        self.assertEqual((fact['actor'], fact['target']), ('unknown', 'she'))
        fact = self.examples['Your trousers are being ironed.'][0]
        self.assertEqual((fact['object'], fact['action']), ('your trousers', 'iron'))
        for text in ['They was given really good advice.', 'She have been waited for for a quarter of an hour already.',
                     'Your trousers is being ironed.', 'I is being waited for by everybody.']:
            self.assertNotIn(text, self.examples)

    def test_passive_indirect_when_question_is_distinct_from_a_time_condition(self):
        facts = self.examples["You'll be told when your car will be sold."]
        self.assertEqual((facts[0]['recipient'], facts[1]['object']), ('addressee', 'your car'))
        self.assertEqual((facts[1]['time'], facts[1]['tense'], facts[1]['status']), ('unknown', 'future', 'questioned'))
        facts = self.examples["I'll take a picture of you when you are given the prize."]
        self.assertEqual((facts[1]['recipient'], facts[1]['tense'], facts[1]['status']), ('addressee', 'present', 'hypothetical'))
        facts = self.examples["If the work isn't finished by evening, you'll be in trouble."]
        self.assertEqual([f['status'] for f in facts], ['hypothetical', 'conditional'])
        self.assertTrue(all('mood' not in f for f in facts))

    def test_unreal_past_forms_keep_present_meaning_and_past_perfect_keeps_past(self):
        facts = self.examples['If I knew his number I would phone him.']
        self.assertEqual([f['tense'] for f in facts], ['present', 'present'])
        self.assertEqual([f['aspect'] for f in facts], ['simple', 'simple'])
        self.assertTrue(all(f['status'] == 'counterfactual' and f['mood'] == 'subjunctive' for f in facts))
        facts = self.examples['If I had known his number I would have phoned him long ago.']
        self.assertEqual([f['tense'] for f in facts], ['past', 'past'])
        self.assertEqual([f['aspect'] for f in facts], ['perfect', 'perfect'])
        self.assertEqual(facts[1]['condition'], 'if I had known his number')
        self.assertEqual(facts[1]['time'], 'long ago')

    def test_mixed_conditionals_preserve_time_of_each_clause(self):
        facts = self.examples["If you had prepared everything beforehand we wouldn't wait for you now."]
        self.assertEqual([(f['tense'], f['aspect']) for f in facts], [('past', 'perfect'), ('present', 'simple')])
        self.assertEqual((facts[1]['target'], facts[1]['polarity']), ('addressee', 'negative'))
        facts = self.examples['If I were rich I would have stopped at the Ritz.']
        self.assertEqual([(f['tense'], f['aspect']) for f in facts], [('present', 'simple'), ('past', 'perfect')])
        facts = self.examples["If you had bought a present last week, you wouldn't be nervous now."]
        self.assertEqual([f['time'] for f in facts], ['last week', 'now'])
        facts = self.examples["If you hadn't missed the lecture you wouldn't be asking me about it now."]
        self.assertEqual((facts[1]['aspect'], facts[1]['topic']), ('continuous', 'lecture'))

    def test_conditional_advice_and_reported_unreality_do_not_add_intentions(self):
        facts = self.examples['If I were you I would learn English.']
        self.assertEqual([f['actor'] for f in facts], ['user', 'user'])
        self.assertEqual(facts[0]['complement'], 'addressee')
        self.assertTrue(all(f['operation'] == 'none' for f in facts))
        facts = self.examples["He said that if he ate less he wouldn't be so fat."]
        self.assertEqual((facts[0]['tense'], facts[0]['status']), ('past', 'asserted'))
        self.assertTrue(all(f['tense'] == 'present' and f['status'] == 'counterfactual' for f in facts[1:]))
        facts = self.examples['He said that if he knew this song he would sing it now.']
        self.assertEqual((facts[-1]['object'], facts[-1]['time']), ('this song', 'now'))

    def test_unreal_questions_keep_missing_roles_and_subjunctive_scope(self):
        facts = self.examples['Where would you have gone last summer if you had had a chance?']
        question = next(f for f in facts if f['action'] == 'go')
        self.assertEqual((question['destination'], question['tense'], question['aspect']), ('unknown', 'past', 'perfect'))
        self.assertEqual((question['status'], question['mood']), ('questioned', 'subjunctive'))
        facts = self.examples['What would you do if you were the headmaster?']
        question = next(f for f in facts if f['action'] == 'do')
        self.assertEqual((question['object'], question['actor'], question['status']), ('unknown', 'addressee', 'questioned'))

    def test_unreal_passives_preserve_patient_and_unstated_agent(self):
        facts = self.examples["If you had bought a ticket you wouldn't have been fined."]
        self.assertEqual((facts[1]['actor'], facts[1]['object'], facts[1]['voice']), ('unknown', 'addressee', 'passive'))
        self.assertEqual((facts[1]['status'], facts[1]['polarity']), ('counterfactual', 'negative'))
        facts = self.examples["I would have met him yesterday if I hadn't been held up at work."]
        held = next(f for f in facts if f['action'] == 'hold up')
        self.assertEqual((held['actor'], held['object'], held['voice']), ('unknown', 'user', 'passive'))
        self.assertEqual((held['tense'], held['aspect'], held['polarity']), ('past', 'perfect', 'negative'))

    def test_but_for_does_not_add_an_asserted_opposite_fact(self):
        for text, tense, aspect in [('But for the bad weather we would go for a walk now.', 'present', 'simple'),
                                    ('But for the bad weather we would have gone for a walk yesterday.', 'past', 'perfect'),
                                    ("But for the heavy snowfall the flight wouldn't have been cancelled.", 'past', 'perfect')]:
            facts = self.examples[text]
            self.assertEqual(len(facts), 1)
            self.assertEqual((facts[0]['tense'], facts[0]['aspect'], facts[0]['status']), (tense, aspect, 'counterfactual'))
            self.assertTrue(facts[0]['condition'].startswith('but for '))

    def test_wishes_preserve_desired_polarity_time_and_are_not_current_operations(self):
        for text, tense, aspect, polarity in [("I wish I didn't work.", 'present', 'simple', 'negative'),
                                              ("I wish he hadn't said it.", 'past', 'perfect', 'negative'),
                                              ('I wish I had won the first prize.', 'past', 'perfect', 'positive'),
                                              ('I wish he would come soon.', 'future', 'simple', 'positive')]:
            facts = self.examples[text]
            self.assertEqual(len(facts), 2)
            self.assertEqual((facts[0]['action'], facts[0]['status']), ('wish', 'asserted'))
            self.assertEqual((facts[1]['tense'], facts[1]['aspect'], facts[1]['polarity']), (tense, aspect, polarity))
            self.assertEqual(facts[1]['status'], 'counterfactual')
            self.assertTrue(all(f['operation'] == 'none' for f in facts))
        fact = self.examples['I want to be a housewife.'][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation']), ('be', 'be', 'add'))
        facts = self.examples['I read a lot because I want to know a lot.']
        self.assertEqual((facts[0]['status'], facts[-1]['status'], facts[-1]['operation']), ('asserted', 'intended', 'add'))

    def test_wish_negation_modality_and_frequency_do_not_change_the_wrong_scope(self):
        fact = self.examples["I wish I didn't have to wake up so early every day."][1]
        self.assertEqual((fact['modality'], fact['polarity'], fact['negation_scope']), ('have to', 'negative', 'modality'))
        facts = self.examples["If I had cleaned the room yesterday I wouldn't have to do it now."]
        fact = next(f for f in facts if f.get('necessity') == 'have to')
        self.assertEqual(fact['modality'], 'would')
        self.assertEqual((fact['action'], fact['object'], fact['negation_scope']), ('clean', 'room', 'modality'))
        fact = self.examples["If I were you I wouldn't always trust people."][1]
        self.assertEqual((fact['frequency'], fact['negation_scope']), ('always', 'not always'))
        for text in ["I wish he weren't so lazy.", "I wish he wasn't so lazy."]:
            self.assertEqual(self.examples[text][1]['polarity'], 'negative')

    def test_wish_references_do_not_borrow_task_antecedents(self):
        for text in ["I wish I hadn't bought it.", 'I wish I had washed it yesterday.', 'I wish I had left it open.']:
            self.assertEqual(self.examples[text][1]['object'], 'it')
        facts = self.examples['Mr. Crook wrote a novel and then burnt it.']
        self.assertEqual([f['object'] for f in facts], ['novel', 'novel'])
        facts = self.examples['If I saw a hat that I like I would buy it.']
        self.assertEqual(next(f for f in facts if f['action'] == 'buy')['object'], 'hat')
        facts = self.examples['If she liked cats she would have bought one long ago.']
        self.assertEqual([f['object'] for f in facts], ['cats', 'cat'])

    def test_conditional_body_parts_and_passive_reciprocity_keep_participants(self):
        fact = self.examples["If it hadn't been slippery he wouldn't have broken his arm."][1]
        self.assertEqual((fact['actor'], fact['object'], fact['possessor'], fact['reflexive']), ('he', 'his arm', 'he', 'no'))
        facts = self.examples["I dyed my hair but now I don't like the colour."]
        self.assertEqual((facts[0]['object'], facts[0]['possessor'], facts[0]['reflexive']), ('my hair', 'user', 'no'))
        facts = self.examples["Though Mr. Smith had been described to me I couldn't recognize him until we were introduced to each other."]
        self.assertEqual((facts[-1]['object'], facts[-1]['recipient'], facts[-1]['reciprocal'], facts[-1]['reflexive']), ('we', 'we', 'yes', 'no'))

    def test_conditional_name_slots_bind_actor_possessor_and_condition_together(self):
        facts = self.examples["If Adam weren't nervous he wouldn't have shouted at his child."]
        self.assertEqual([f['actor'] for f in facts], ['Adam', 'Adam'])
        self.assertEqual((facts[1]['target'], facts[1]['condition']), ("Adam's child", "if Adam weren't nervous"))
        facts = self.examples['If Jane had cleaned her coat she would be able to put it on.']
        self.assertEqual([f['actor'] for f in facts], ['Jane', 'Jane'])
        self.assertEqual([f['object'] for f in facts], ["Jane's coat", "Jane's coat"])
        self.assertEqual(facts[1]['condition'], 'if Jane had cleaned her coat')

    def test_wish_slots_bind_agreement_possession_content_and_skill(self):
        facts = self.examples['They wish they had a house of their own.']
        self.assertEqual([f['actor'] for f in facts], ['they', 'they'])
        self.assertEqual((facts[0]['content'], facts[1]['object']), ('they had a house of their own', 'house of their own'))
        facts = self.examples['She wishes she were a model.']
        self.assertEqual([f['actor'] for f in facts], ['she', 'she'])
        fact = self.examples['I wish I could swim.'][1]
        self.assertEqual((fact['action'], fact['modality'], fact['status']), ('swim', 'could', 'counterfactual'))
        facts = self.examples['He wishes he spoke French better.']
        self.assertEqual((facts[0]['content'], facts[1]['object']), ('he spoke French better', 'French'))
        for text in ['They wishes they had a house of their own.', 'She wish she were a model.',
                     'If she liked cats she would have bought ones long ago.']:
            self.assertNotIn(text, self.examples)

    def test_lessons_61_65_rewrites_questions_and_printed_variants_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u61-t05a', 'u61-t05c'), ('u61-e1-a07-task', 'u61-e1-a07b'),
                              ('u61-e2-a04a', 'u61-e2-a04b'), ('u63-e3-06-task', 'u63-e3-06'),
                              ('u63-e4-06-task', 'u63-e4-06'), ('u64-e1-02a', 'u64-e1-02b'),
                              ('u64-e1-03a', 'u64-e1-03b'), ('u64-e4-05a', 'u64-e4-05b'),
                              ('u65-e1-04q', 'u65-e1-04b'), ('u65-e5-14-task', 'u65-e5-14'),
                              ('u65-e6-14a', 'u65-e6-14b')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_could_ability_and_polite_requests_have_different_time_and_status(self):
        fact = self.examples['He could read before he was five.'][0]
        self.assertEqual((fact['tense'], fact['status'], fact['modal_meaning']), ('past', 'possible', 'ability'))
        fact = self.examples['Could you answer my question?'][0]
        self.assertEqual((fact['tense'], fact['status'], fact['modal_meaning']), ('present', 'required', 'request'))
        self.assertEqual(fact['politeness'], 'polite')
        fact = self.examples['Could I ask you a question?'][0]
        self.assertEqual((fact['actor'], fact['recipient'], fact['modal_meaning']), ('user', 'addressee', 'permission'))
        fact = self.examples['Could you skate when you were six?'][0]
        self.assertEqual((fact['tense'], fact['status']), ('past', 'questioned'))

    def test_ability_tenses_and_stacked_modals_do_not_claim_completed_events(self):
        facts = self.examples["I can't drive now but next year I will be able to drive."]
        self.assertEqual([f['tense'] for f in facts], ['present', 'future'])
        self.assertEqual([f['polarity'] for f in facts], ['negative', 'positive'])
        self.assertTrue(all(f['status'] == 'possible' for f in facts))
        fact = self.examples["She hasn't been able to work well lately."][0]
        self.assertEqual((fact['aspect'], fact['modality'], fact['polarity']), ('perfect', 'be able to', 'negative'))
        fact = self.examples['He might be able to do it.'][0]
        self.assertEqual((fact['modality'], fact['ability'], fact['status']), ('might', 'be able to', 'possible'))
        fact = self.examples['The child must be able to eat by himself.'][0]
        self.assertEqual((fact['modality'], fact['ability'], fact['status'], fact['reflexive']), ('must', 'be able to', 'required', 'no'))

    def test_manage_completion_and_failure_do_not_leak_through_nested_actions(self):
        ability = self.examples['I could get him on the phone.'][0]
        facts = self.examples['I managed to get him on the phone.']
        self.assertEqual((ability['status'], facts[0]['status'], facts[1]['status']), ('possible', 'asserted', 'reported'))
        facts = self.examples["Ted didn't manage to read the word."]
        self.assertEqual((facts[0]['action'], facts[0]['polarity'], facts[1]['status']), ('manage', 'negative', 'hypothetical'))
        fact = self.examples["Ted wasn't able to read the word."][0]
        self.assertEqual((fact['polarity'], fact['outcome']), ('negative', 'not completed'))
        facts = self.examples["I tried not to argue with her but didn't manage to."]
        self.assertEqual((facts[1]['polarity'], facts[-1]['polarity']), ('negative', 'negative'))
        self.assertEqual(facts[-1]['content'], 'not to argue with her')

    def test_cannot_inference_is_distinct_from_prohibition_and_inability(self):
        fact = self.examples["He can't be reading now."][0]
        self.assertEqual((fact['status'], fact['modal_meaning'], fact['negation_scope']), ('inferred', 'inference', 'modality'))
        fact = self.examples["You can't smoke here."][0]
        self.assertEqual((fact['status'], fact['modal_meaning']), ('required', 'prohibition'))
        fact = self.examples["She can't swim."][0]
        self.assertEqual((fact['status'], fact['modal_meaning']), ('possible', 'ability'))
        self.assertEqual(self.examples['Can he be reading?'][0]['status'], 'questioned')

    def test_epistemic_modal_perfect_and_ongoing_aspects_keep_their_time(self):
        for text, tense, aspect in [('Can he have come?', 'past', 'perfect'),
                                    ('Can he have been waiting so long?', 'present', 'perfect_continuous'),
                                    ('He must have already come.', 'past', 'perfect'),
                                    ('He must be reading.', 'present', 'continuous')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['tense'], fact['aspect']), (tense, aspect))
            self.assertIn(fact['status'], ('questioned', 'inferred'))
        fact = self.examples["It can't have rained half an hour ago."][0]
        self.assertEqual((fact['actor'], fact['time']), ('none', 'half an hour ago'))
        fact = self.examples['Can he have been playing the piano since morning?'][0]
        self.assertEqual(fact['since'], 'morning')
        self.assertNotIn('duration', fact)

    def test_modal_failure_scope_does_not_assert_the_embedded_event(self):
        facts = self.examples["He can't have failed to come."]
        self.assertEqual((facts[0]['action'], facts[0]['polarity'], facts[0]['negation_scope']), ('fail', 'negative', 'modality'))
        self.assertEqual((facts[1]['action'], facts[1]['status'], facts[1]['polarity']), ('come', 'hypothetical', 'positive'))
        facts = self.examples['He must have failed to come.']
        self.assertEqual((facts[0]['action'], facts[0]['polarity'], facts[0]['status']), ('fail', 'positive', 'inferred'))
        self.assertEqual(facts[1]['status'], 'hypothetical')
        facts = self.examples['Can she have failed to give you a present for your birthday?']
        self.assertEqual((facts[0]['status'], facts[1]['recipient'], facts[1]['object']), ('questioned', 'addressee', 'present'))
        facts = self.examples["He can't have failed you last week."]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]['object'], 'addressee')

    def test_must_obligation_and_inference_are_distinct_in_the_same_verb_frame(self):
        required = self.examples['He had to sell his car.'][0]
        inferred = self.examples['He must have sold his car.'][0]
        self.assertEqual((required['action'], required['object']), (inferred['action'], inferred['object']))
        self.assertEqual((required['status'], required['aspect']), ('required', 'simple'))
        self.assertEqual((inferred['status'], inferred['aspect']), ('inferred', 'perfect'))
        fact = self.examples['He must have never been here before.'][0]
        self.assertEqual((fact['polarity'], fact['negation_scope'], fact['status']), ('negative', 'predicate', 'inferred'))
        fact = self.examples['Don must have misunderstood his wife again.'][0]
        self.assertEqual((fact['action'], fact['object'], fact['polarity']), ('misunderstand', "Don's wife", 'positive'))

    def test_be_to_arrangements_do_not_assert_completion_and_passives_keep_actor(self):
        fact = self.examples['We were to meet at five.'][0]
        self.assertEqual((fact['tense'], fact['modality'], fact['status']), ('past', 'be to', 'planned'))
        fact = self.examples['The medicine is to be taken after meals.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice'], fact['status']), ('unknown', 'medicine', 'passive', 'required'))
        facts = self.examples['The guide was to meet the group and take them to the hotel yesterday.']
        self.assertEqual([f['object'] for f in facts], ['group', 'group'])
        self.assertTrue(all(f['status'] == 'planned' for f in facts))

    def test_no_necessity_is_not_an_order_to_refrain_from_the_action(self):
        fact = self.examples["I don't have to go to work today."][0]
        self.assertEqual((fact['status'], fact['negation_scope']), ('not required', 'modality'))
        facts = self.examples["The bell had gone and I didn't have to answer the question."]
        self.assertEqual((facts[-1]['status'], facts[-1]['negation_scope']), ('not required', 'modality'))
        fact = self.examples["You mustn't shout at children."][0]
        self.assertEqual((fact['status'], fact['polarity'], fact['modal_meaning']), ('required', 'negative', 'prohibition'))
        self.assertNotIn('negation_scope', fact)

    def test_reported_modal_commands_bind_all_actions_to_the_named_recipient(self):
        facts = self.examples['“You are to look after the children properly, feed them and take them for a walk,” Mrs. Swift said to the baby-sitter.']
        self.assertEqual((facts[0]['actor'], facts[0]['recipient']), ('Mrs. Swift', 'baby-sitter'))
        self.assertEqual([f['action'] for f in facts[1:]], ['look after', 'feed', 'take'])
        self.assertTrue(all(f['actor'] == 'baby-sitter' and f['object'] == 'children' and f['status'] == 'required' for f in facts[1:]))
        facts = self.examples['“When am I to return the book?” Steve asked the librarian.']
        self.assertEqual((facts[1]['actor'], facts[1]['time'], facts[1]['status']), ('Steve', 'unknown', 'questioned'))
        facts = self.examples['He said that I must come at once.']
        self.assertEqual((facts[0]['tense'], facts[1]['tense']), ('past', 'present'))

    def test_likelihood_keeps_future_probability_without_asserting_a_negative_event(self):
        for text, modality in [('He is likely to pass his exam.', 'be likely to'),
                                ('They are unlikely to win tomorrow.', 'be unlikely to')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['tense'], fact['status'], fact['polarity'], fact['modality']),
                             ('future', 'possible', 'positive', modality))
        facts = self.examples['Now he is to be in California and in a week in Colorado.']
        self.assertEqual([f['location'] for f in facts], ['in California', 'in Colorado'])
        self.assertTrue(all(f['status'] == 'planned' for f in facts))

    def test_may_permission_is_distinct_from_possibility_and_reproach(self):
        fact = self.examples['You may come in.'][0]
        self.assertEqual((fact['status'], fact['modal_meaning']), ('permitted', 'permission'))
        fact = self.examples['You may get there by bus.'][0]
        self.assertEqual((fact['status'], fact['modal_meaning']), ('possible', 'circumstantial possibility'))
        fact = self.examples['Might I come in?'][0]
        self.assertEqual((fact['tense'], fact['status'], fact['modal_meaning']), ('present', 'questioned', 'permission'))
        fact = self.examples['He said I might get there by bus.'][1]
        self.assertEqual((fact['tense'], fact['modal_meaning']), ('past', 'circumstantial possibility'))

    def test_may_prohibitions_keep_passive_roles_and_unless_scope(self):
        fact = self.examples['Alcoholic drinks may not be sold to people under 18.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['recipient']), ('unknown', 'alcoholic drinks', 'people under 18'))
        self.assertEqual((fact['voice'], fact['status'], fact['polarity']), ('passive', 'required', 'negative'))
        facts = self.examples['Children under 7 may not enter unless they are with an adult.']
        self.assertEqual(facts[0]['condition'], 'unless they are with an adult')
        self.assertEqual((facts[1]['actor'], facts[1]['companion'], facts[1]['status']), ('children under 7', 'adult', 'hypothetical'))
        self.assertEqual(self.examples['Yes, you may.'][0]['action'], 'unknown')
        self.assertEqual(self.examples["No, you can't."][0]['action'], 'unknown')

    def test_might_reproaches_do_not_become_asserted_past_events_or_current_intentions(self):
        fact = self.examples['You might write him a letter.'][0]
        self.assertEqual((fact['status'], fact['tense'], fact['attitude']), ('advised', 'present', 'reproach'))
        fact = self.examples['You might have written him a letter long ago.'][0]
        self.assertEqual((fact['status'], fact['tense'], fact['aspect'], fact['attitude']), ('counterfactual', 'past', 'perfect', 'reproach'))
        for text in ['You might get up earlier.', 'They might have painted the fence.', 'He might be more polite.']:
            fact = self.examples[text][0]
            self.assertEqual((fact['intention'], fact['operation']), ('none', 'none'))

    def test_modal_task_context_is_not_carried_into_standalone_answers(self):
        self.assertEqual(self.examples['She must have looked ill.'][0]['actor'], 'she')
        self.assertEqual(self.examples['John must have proposed to her.'][0]['recipient'], 'her')
        self.assertEqual(self.examples['Can you have read it yet?'][0]['object'], 'it')
        self.assertEqual(self.examples['You must publish it.'][0]['object'], 'it')
        facts = self.examples['Ask your neighbour to let you have his car.']
        self.assertEqual(facts[-1]['object'], "your neighbour's car")
        facts = self.examples['Susan says I must drive her to the station.']
        self.assertEqual((facts[-1]['object'], facts[-1]['destination']), ('Susan', 'station'))

    def test_modal_slots_keep_ability_and_necessity_auxiliaries_and_articles_bound(self):
        fact = self.examples["They weren't able to start the car."][0]
        self.assertEqual((fact['actor'], fact['polarity']), ('they', 'negative'))
        fact = self.examples["He hasn't been able to sleep well lately."][0]
        self.assertEqual((fact['actor'], fact['aspect']), ('he', 'perfect'))
        self.assertEqual(self.examples['He has to stay at home every evening.'][0]['actor'], 'he')
        self.assertEqual(self.examples['The pills are to be taken after meals.'][0]['object'], 'pills')
        facts = self.examples['I was late because I had to type a letter.']
        self.assertEqual((facts[0]['reason'], facts[1]['object']), ('I had to type a letter', 'letter'))
        fact = self.examples['You might have written him an article long ago.'][0]
        self.assertEqual((fact['object'], fact['recipient']), ('article', 'him'))
        for text in ['They was not able to start the car.', 'He have to stay at home every evening.',
                     'The pills is to be taken after meals.', 'I was late because I had to type an letter.']:
            self.assertNotIn(text, self.examples)

    def test_modal_slots_bind_ongoing_verbs_possessors_and_duration_types(self):
        fact = self.examples['Can they be working now?'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['aspect'], fact['status']), ('they', 'work', 'continuous', 'questioned'))
        fact = self.examples['She is likely to pass her exam.'][0]
        self.assertEqual((fact['actor'], fact['object']), ('she', 'her exam'))
        fact = self.examples['Can Robert have been riding his bicycle since yesterday?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['since']), ('Robert', "Robert's bicycle", 'yesterday'))
        fact = self.examples['We must have been waiting for a bus for three hours.'][0]
        self.assertEqual((fact['target'], fact['duration']), ('bus', 'three hours'))
        self.assertNotIn('Can Robert have been riding his bicycle since three hours?', self.examples)

    def test_lessons_66_70_alternatives_task_answers_and_modal_contrasts_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u66-t07', 'u66-e4-01b'), ('u66-t16', 'u66-t17'),
                              ('u66-e4-03a', 'u66-e4-03b'), ('u66-e4-07a', 'u66-e4-07b'),
                              ('u66-e4-09a', 'u66-e4-09b'), ('u67-e1-03-task', 'u67-e1-07'),
                              ('u67-e2-11-task', 'u67-e2-11'), ('u69-t09', 'u69-t10'),
                              ('u69-e1-03-task-a', 'u69-e1-03'), ('u69-e2-a05', 'u69-e2-a06'),
                              ('u70-t10', 'u70-t11'), ('u70-e1-03-task-q', 'u70-e1-05'),
                              ('u70-e3-03-task', 'u70-e3-03')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_possibility_negates_predicate_but_permission_can_be_denied(self):
        fact = self.examples['He may not know about it.'][0]
        self.assertEqual((fact['status'], fact['modal_meaning'], fact['polarity'], fact['negation_scope']),
                         ('possible', 'possibility', 'negative', 'predicate'))
        facts = self.examples['You may not enter without knocking.']
        self.assertEqual((facts[0]['status'], facts[0]['modal_meaning']), ('required', 'prohibition'))
        self.assertEqual((facts[1]['action'], facts[1]['status']), ('knock', 'hypothetical'))
        fact = self.examples['He might be reading.'][0]
        self.assertEqual((fact['modality'], fact['status']), ('might', 'possible'))
        self.assertNotIn('attitude', fact)

    def test_possibility_preserves_nested_negation_and_ongoing_aspect(self):
        facts = self.examples['She may not want to see you.']
        self.assertEqual((facts[0]['action'], facts[0]['polarity']), ('want', 'negative'))
        self.assertEqual((facts[1]['action'], facts[1]['polarity'], facts[1]['status']),
                         ('see', 'positive', 'hypothetical'))
        fact = self.examples['He may not have been doing anything all this time.'][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['status']), ('present', 'perfect_continuous', 'possible'))
        fact = self.examples['He may have shouted.'][0]
        self.assertEqual((fact['tense'], fact['aspect']), ('past', 'perfect'))

    def test_unnecessary_completed_action_differs_from_absent_obligation(self):
        fact = self.examples["You needn't have gone there."][0]
        self.assertEqual((fact['status'], fact['tense'], fact['aspect'], fact['outcome']),
                         ('not required', 'past', 'perfect', 'completed'))
        facts = self.examples["I didn't have to come as they sent me the answer by post."]
        self.assertEqual((facts[0]['status'], facts[0]['aspect'], facts[0]['negation_scope']),
                         ('not required', 'simple', 'modality'))
        self.assertNotIn('outcome', facts[0])
        self.assertEqual((facts[1]['actor'], facts[1]['action'], facts[1]['recipient']), ('they', 'send', 'user'))

    def test_lexical_need_and_modal_need_are_separate(self):
        fact = self.examples["I don't need this pen now."][0]
        self.assertEqual((fact['action'], fact['object'], fact['polarity']), ('need', 'this pen', 'negative'))
        self.assertNotIn('modality', fact)
        fact = self.examples['Did you need any help?'][0]
        self.assertEqual((fact['action'], fact['tense'], fact['status']), ('need', 'past', 'questioned'))
        fact = self.examples['Need I go there?'][0]
        self.assertEqual((fact['action'], fact['modality'], fact['status']), ('go', 'need', 'questioned'))

    def test_necessity_does_not_import_neighbouring_story_objects(self):
        for text, role, expected in [('No, you needn\'t.', 'action', 'unknown'),
                                     ('Her husband has paid it.', 'object', 'it'),
                                     ('I could give you mine.', 'object', 'mine'),
                                     ('He has them.', 'object', 'them')]:
            self.assertEqual(self.examples[text][0][role], expected)
        fact = self.examples["The child needn't put on his hat."][0]
        self.assertEqual((fact['actor'], fact['object']), ('child', "child's hat"))

    def test_past_advice_records_actual_outcome_without_reversing_polarity(self):
        fact = self.examples['You should have put on something warm.'][0]
        self.assertEqual((fact['status'], fact['polarity'], fact['outcome']), ('counterfactual', 'positive', 'not completed'))
        fact = self.examples["You shouldn't have run so much."][0]
        self.assertEqual((fact['status'], fact['polarity'], fact['outcome']), ('counterfactual', 'negative', 'completed'))
        self.assertEqual((fact['tense'], fact['aspect'], fact['operation']), ('past', 'perfect', 'none'))
        fact = self.examples["You shouldn't go there alone."][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['status']), ('present', 'simple', 'advised'))
        self.assertNotIn('outcome', fact)

    def test_past_advice_questions_and_ellipses_do_not_assert_outcome(self):
        for text in ['Should I have told him about it?', 'Should I have warned you about my arrival?']:
            fact = self.examples[text][0]
            self.assertEqual((fact['status'], fact['aspect']), ('questioned', 'perfect'))
            self.assertNotIn('outcome', fact)
        self.assertEqual(self.examples["Why didn't you?"][0]['action'], 'unknown')
        facts = self.examples["You shouldn't have kept them waiting."]
        self.assertEqual((facts[0]['outcome'], facts[1]['actor'], facts[1]['status']), ('completed', 'they', 'reported'))

    def test_modal_review_keeps_english_source_and_desired_ability(self):
        fact = self.examples['You must obey the police.'][0]
        self.assertEqual((fact['action'], fact['polarity']), ('obey', 'positive'))
        facts = self.examples['I would like to be able to cook as well as you do.']
        self.assertEqual([f['action'] for f in facts], ['like', 'cook', 'cook'])
        self.assertEqual([f['status'] for f in facts], ['desired', 'desired', 'reported'])
        self.assertTrue(all(f['operation'] == 'none' for f in facts))
        self.assertEqual(self.examples['Liz did it.'][0]['object'], 'it')

    def test_modal_review_preserves_passive_agents_and_stative_perfect(self):
        fact = self.examples['He must have been hurt by your words.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['voice']),
                         ('your words', 'hurt', 'he', 'passive'))
        fact = self.examples['Can he have been involved in this robbery?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('unknown', 'he', 'questioned'))
        fact = self.examples['Can they have been in love all this time?'][0]
        self.assertEqual((fact['tense'], fact['aspect']), ('present', 'perfect'))
        self.assertEqual(self.examples['She must have had little practice.'][0]['polarity'], 'positive')

    def test_gerund_simple_active_and_passive_are_not_continuous(self):
        active = self.examples['She likes interviewing people.'][1]
        passive = self.examples['She likes being interviewed.'][1]
        self.assertEqual((active['actor'], active['object'], active['voice']), ('she', 'people', 'active'))
        self.assertEqual((passive['actor'], passive['object'], passive['voice']), ('unknown', 'she', 'passive'))
        for fact in [active, passive]:
            self.assertEqual((fact['verb_form'], fact['tense'], fact['aspect'], fact['status']),
                             ('gerund', 'nonfinite', 'simple', 'preferred'))

    def test_perfect_gerund_marks_anteriority_and_passive_recipient(self):
        active = self.examples['He regretted having told them the truth.'][1]
        passive = self.examples['He regretted having been told the truth.'][1]
        self.assertEqual((active['actor'], active['recipient']), ('he', 'them'))
        self.assertEqual((passive['actor'], passive['recipient'], passive['voice']), ('unknown', 'he', 'passive'))
        for fact in [active, passive]:
            self.assertEqual((fact['aspect'], fact['tense'], fact['temporal_relation']),
                             ('perfect', 'nonfinite', 'before main verb'))
        fact = self.examples['They regret going to the party.'][1]
        self.assertEqual((fact['aspect'], fact['temporal_relation']), ('simple', 'before main verb'))

    def test_active_gerund_after_need_has_passive_meaning(self):
        facts = self.examples['My shoes need repairing.']
        self.assertEqual((facts[0]['actor'], facts[0]['action']), ('my shoes', 'need'))
        fact = facts[1]
        self.assertEqual((fact['actor'], fact['action'], fact['object']), ('unknown', 'repair', 'my shoes'))
        self.assertEqual((fact['voice'], fact['form_voice'], fact['status']), ('passive', 'active', 'required'))
        self.assertNotIn('outcome', fact)

    def test_gerund_owner_controls_embedded_actor_without_negating_action(self):
        for text, actor in [("I don't mind opening the window.", 'user'),
                            ("I don't mind his opening the window.", 'he'),
                            ("I don't mind him opening the window.", 'he')]:
            facts = self.examples[text]
            self.assertEqual((facts[0]['actor'], facts[0]['polarity']), ('user', 'negative'))
            self.assertEqual((facts[1]['actor'], facts[1]['polarity'], facts[1]['status']), (actor, 'positive', 'permitted'))
        self.assertEqual(self.examples['Why do you insist on leaving?'][1]['actor'], 'addressee')
        self.assertEqual(self.examples['Why do you insist on her leaving?'][1]['actor'], 'she')

    def test_gerunds_preserve_subject_adverbial_attribute_and_predicative_roles(self):
        for text, action, function, actor in [
                ('Reading is useful.', 'read', 'subject', 'unknown'),
                ('On arriving in London he went to the hotel.', 'arrive', 'adverbial', 'he'),
                ('I had the pleasure of reading the book by this author.', 'read', 'attribute', 'user'),
                ('His hobby is reading adventure books.', 'read', 'predicative', 'he')]:
            fact = next(f for f in self.examples[text] if f['action'] == action)
            self.assertEqual((fact['syntactic_function'], fact['actor'], fact['tense']), (function, actor, 'nonfinite'))

    def test_gerund_subject_transformation_does_not_invent_a_speaker(self):
        original = self.examples["It's boring to clean the flat."]
        transformed = self.examples['Cleaning the flat is boring.']
        self.assertEqual({k:v for k,v in original[0].items() if k!='predicate'},
                         {k:v for k,v in transformed[0].items() if k!='predicate'})
        self.assertEqual((original[0]['predicate'], transformed[0]['predicate']), ("'s", 'is'))
        self.assertEqual((original[1]['verb_form'], transformed[1]['verb_form']), ('infinitive', 'gerund'))
        for fact in [original[1], transformed[1]]:
            self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['status']),
                             ('unknown', 'clean', 'flat', 'hypothetical'))
        self.assertEqual(self.examples['Speaking before a large audience made him nervous.'][1]['actor'], 'he')

    def test_passive_gerunds_keep_recipients_and_unknown_actors(self):
        for text, obj, recipient in [('I remember being asked for help.', 'help', 'user'),
                                     ('He was tired of being asked questions.', 'questions', 'he'),
                                     ("I'm looking forward to being shown the place.", 'place', 'user')]:
            fact = self.examples[text][1]
            self.assertEqual((fact['actor'], fact['object'], fact['recipient'], fact['voice']),
                             ('unknown', obj, recipient, 'passive'))
        fact = self.examples['The manager objects to the work being done now.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('unknown', 'work', 'unwanted'))
        fact = self.examples['I avoid being visited by strangers.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('strangers', 'user', 'unwanted'))

    def test_gerund_passive_changes_control_of_nested_infinitive(self):
        active = self.examples['He objected to forcing me to do it.']
        passive = self.examples['He objected to being forced to do it.']
        self.assertEqual((active[1]['actor'], active[1]['object'], active[2]['actor']), ('he', 'user', 'user'))
        self.assertEqual((passive[1]['actor'], passive[1]['object'], passive[2]['actor']), ('unknown', 'he', 'he'))
        for facts in [active, passive]:
            self.assertEqual((facts[1]['status'], facts[2]['status']), ('unwanted', 'required'))
        self.assertEqual(self.examples['Do you mind telling us how to do it?'][2]['actor'], 'we')
        self.assertEqual(self.examples['Do you mind being told how to do it?'][2]['actor'], 'addressee')

    def test_gerund_requests_keep_explicit_actor_and_requested_action(self):
        for text, actor in [('Do you mind my turning on the TV-set?', 'user'),
                            ('Do you mind switching off the light?', 'addressee'),
                            ('Do you mind her talking to him?', 'she')]:
            facts = self.examples[text]
            self.assertEqual((facts[0]['actor'], facts[0]['status']), ('addressee', 'questioned'))
            self.assertEqual(facts[1]['actor'], actor)
        self.assertEqual(self.examples['Do you mind switching off the light?'][1]['status'], 'required')
        self.assertEqual(self.examples['Do you mind?'][0]['action'], 'mind')

    def test_new_modal_slots_bind_articles_lemmas_and_duration_types(self):
        fact = self.examples['They may not be writing an article.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['object']), ('they', 'write', 'article'))
        fact = self.examples['We may have been sleeping.'][0]
        self.assertEqual((fact['actor'], fact['action'], fact['aspect']), ('we', 'sleep', 'perfect_continuous'))
        self.assertNotIn('They may not be writing a article.', self.examples)
        self.assertIn('He must have been taking a shower for three hours.', self.examples)
        self.assertIn('It may have been raining since yesterday there.', self.examples)
        self.assertNotIn('It may have been raining since three hours there.', self.examples)

    def test_gerund_slots_bind_agreement_ownership_and_passive_roles(self):
        facts = self.examples['They like being interviewed.']
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['object']), ('they', 'unknown', 'they'))
        facts = self.examples["I don't mind their opening the door."]
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['object']), ('user', 'they', 'door'))
        fact = self.examples['I regretted having been told the truth.'][1]
        self.assertEqual((fact['actor'], fact['recipient']), ('unknown', 'user'))
        self.assertNotIn('They likes being interviewed.', self.examples)
        self.assertEqual(self.examples['Thinking is useful.'][1]['action'], 'think')
        self.assertEqual(self.examples['On arriving in Oxford he went to the hotel.'][1]['time'], 'on arriving in Oxford')

    def test_lessons_71_75_task_answers_and_gerund_contrasts_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u71-t03q', 'u71-t03a'), ('u71-e1-01-task', 'u71-e1-01'),
                              ('u72-t09', 'u72-t11'), ('u72-t03a', 'u72-e2-15q'),
                              ('u73-e2-01-task', 'u73-e2-01'), ('u74-e1-a03a', 'u74-e1-a03b'),
                              ('u75-t01', 'u75-t02'), ('u75-t03', 'u75-t04'),
                              ('u75-t12', 'u75-t13b'), ('u75-e1-05-task', 'u75-e1-05'),
                              ('u75-e2-05-task', 'u75-e2-05'), ('u75-e3-05-task-b', 'u75-e3-08')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_gerund_denial_does_not_assert_the_denied_event(self):
        for text, actor, action in [('He denies robbing the bank.', 'he', 'rob'),
                                    ('The man denies stealing anything.', 'man', 'steal')]:
            facts = self.examples[text]
            self.assertEqual((facts[0]['action'], facts[0]['polarity']), ('deny', 'positive'))
            self.assertEqual((facts[1]['actor'], facts[1]['action'], facts[1]['status']), (actor, action, 'denied'))
            self.assertEqual(facts[1]['tense'], 'nonfinite')

    def test_cannot_help_gerund_means_involuntary_action_not_absence(self):
        facts = self.examples["He couldn't help laughing."]
        self.assertEqual((facts[0]['modal_meaning'], facts[0]['polarity']), ('avoidance', 'negative'))
        self.assertEqual((facts[1]['action'], facts[1]['polarity'], facts[1]['status'], facts[1]['control']),
                         ('laugh', 'positive', 'reported', 'involuntary'))
        fact = self.examples["The letter was so disrespectful that she couldn't help tearing it."][-1]
        self.assertEqual((fact['actor'], fact['object']), ('she', 'letter'))

    def test_gerund_need_worth_and_elliptical_objects_keep_semantic_roles(self):
        fact = self.examples['The floor needs washing.'][1]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['voice'], fact['form_voice']),
                         ('unknown', 'wash', 'floor', 'passive', 'active'))
        self.assertEqual(fact['status'], 'required')
        fact = self.examples['The exhibition is worth visiting.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('unknown', 'exhibition', 'advised'))
        self.assertEqual(self.examples['It needs repairing.'][1]['object'], 'it')
        self.assertEqual(self.examples['They need drying.'][1]['object'], 'they')

    def test_before_after_and_without_gerunds_keep_temporal_and_negative_scope(self):
        facts = self.examples['Read the instruction before opening the box.']
        self.assertEqual((facts[0]['status'], facts[1]['actor'], facts[1]['temporal_relation']),
                         ('required', 'addressee', 'after main verb'))
        self.assertEqual(facts[1]['status'], 'hypothetical')
        facts = self.examples['After doing his morning exercises he took a shower.']
        self.assertEqual((facts[0]['temporal_relation'], facts[0]['status']), ('before main verb', 'reported'))
        facts = self.examples['He sat down without asking permission.']
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('positive', 'negative'))
        self.assertEqual((facts[1]['actor'], facts[1]['syntactic_function']), ('he', 'adverbial'))

    def test_continuation_cessation_and_questioned_cessation_are_separate(self):
        fact = self.examples['She kept laughing.'][1]
        self.assertEqual((fact['status'], fact['continuation']), ('reported', 'yes'))
        fact = self.examples['She gave up smoking.'][1]
        self.assertEqual((fact['action'], fact['status']), ('smoke', 'ceased'))
        facts = self.examples['Can she have given up dancing?']
        self.assertEqual((facts[0]['status'], facts[1]['status'], facts[1]['phase']),
                         ('questioned', 'hypothetical', 'cessation'))

    def test_gerund_owners_and_prepositional_passive_do_not_become_matrix_subject(self):
        facts = self.examples['Why do you insist on her returning?']
        self.assertEqual([f['actor'] for f in facts], ['addressee', 'she'])
        facts = self.examples["I'm afraid of her doing it in any case."]
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['status']), ('user', 'she', 'feared'))
        fact = self.examples['I object to being spoken to like this.'][1]
        self.assertEqual((fact['actor'], fact['recipient'], fact['voice']), ('unknown', 'user', 'passive'))
        self.assertEqual(self.examples["I'm not used to being told what to do."][2]['actor'], 'user')

    def test_infinitive_perfect_and_continuous_preserve_relative_time(self):
        fact = self.examples['He pretended to be reading a newspaper.'][1]
        self.assertEqual((fact['verb_form'], fact['tense'], fact['aspect'], fact['status']),
                         ('infinitive', 'nonfinite', 'continuous', 'pretended'))
        self.assertEqual(fact['temporal_relation'], 'simultaneous with main verb')
        fact = self.examples['He pretended to have been reading a book for some time.'][1]
        self.assertEqual((fact['aspect'], fact['duration'], fact['temporal_relation']),
                         ('perfect_continuous', 'some time', 'before main verb'))
        fact = self.examples['They claim to have solved the problem.'][1]
        self.assertEqual((fact['aspect'], fact['status']), ('perfect', 'reported'))
        self.assertTrue(all(f['operation'] == 'none' for f in self.examples['They claim to have solved the problem.']))

    def test_infinitive_passive_does_not_assign_the_raiser_as_agent(self):
        fact = self.examples['The letters seem to have been signed.'][1]
        self.assertEqual((fact['actor'], fact['action'], fact['object'], fact['voice']), ('unknown', 'sign', 'letters', 'passive'))
        fact = self.examples['He is happy to have been invited to the party.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['aspect']), ('unknown', 'he', 'perfect'))
        fact = self.examples['He wanted to be asked about it.'][1]
        self.assertEqual((fact['actor'], fact['recipient'], fact['status']), ('unknown', 'he', 'intended'))

    def test_make_and_allow_passives_keep_embedded_subject_and_marker(self):
        active = self.examples['They made me say it.']
        passive = self.examples['I was made to say it.']
        self.assertEqual((active[0]['actor'], active[1]['actor'], active[1]['marker']), ('they', 'user', 'bare'))
        self.assertEqual((passive[0]['actor'], passive[0]['object'], passive[1]['actor']), ('unknown', 'user', 'user'))
        self.assertNotIn('marker', passive[1])
        facts = self.examples['I was allowed to visit the sick man.']
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('user', 'permitted'))
        facts = self.examples["I can't make the child go to bed."]
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('child', 'hypothetical'))

    def test_infinitive_negation_and_advice_have_their_own_scope(self):
        facts = self.examples['They claim not to understand anything.']
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('positive', 'negative'))
        facts = self.examples['Jack decided not to answer the letter.']
        self.assertEqual((facts[1]['actor'], facts[1]['polarity'], facts[1]['status']), ('Jack', 'negative', 'intended'))
        fact = self.examples["You'd better spend the evening at home."][0]
        self.assertEqual((fact['tense'], fact['modality'], fact['status']), ('present', 'had better', 'advised'))
        fact = self.examples['The doctor forbade him to smoke.'][1]
        self.assertEqual((fact['status'], fact['polarity']), ('required', 'negative'))

    def test_for_infinitive_identifies_actor_independently_of_possessor(self):
        facts = self.examples['He has a lot of books for you to read.']
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['object']), ('he', 'addressee', 'books'))
        self.assertEqual(self.examples['He has a lot of books to read.'][1]['actor'], 'he')
        self.assertEqual(self.examples['There are a lot of books to read.'][1]['actor'], 'unknown')
        self.assertEqual(self.examples['There are other facts to prove this theory.'][1]['actor'], 'other facts')

    def test_degree_infinitives_preserve_unachievable_action_and_object(self):
        fact = self.examples['The article is too difficult for him to translate.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('he', 'article', 'impossible'))
        fact = self.examples['The house is not large enough for such a big family to live in.'][1]
        self.assertEqual((fact['actor'], fact['location'], fact['status']), ('such a big family', 'in the house', 'impossible'))
        facts = self.examples['The car is not cheap enough for me to buy.']
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('negative', 'positive'))
        self.assertEqual(self.examples['These books are easy enough for the child to read.'][1]['status'], 'possible')
        self.assertEqual(self.examples["I can't buy it."][0]['object'], 'it')

    def test_ordinal_infinitive_describes_reported_event_and_bound_possessor(self):
        facts = self.examples['Frank was the first to tell his father the pleasant news.']
        self.assertEqual((facts[0]['complement'], facts[1]['actor'], facts[1]['recipient']),
                         ('first', 'Frank', "Frank's father"))
        self.assertEqual((facts[1]['order'], facts[1]['status']), ('first', 'reported'))
        fact = self.examples['He was the second to get to the finishing line.'][1]
        self.assertEqual((fact['action'], fact['destination'], fact['order']), ('get', 'finishing line', 'second'))

    def test_causative_have_is_distinct_from_doing_and_modal_perfect(self):
        direct = self.examples['I must cut my hair.']
        self.assertEqual(len(direct), 1)
        self.assertEqual((direct[0]['actor'], direct[0]['action']), ('user', 'cut'))
        facts = self.examples['I must have my hair cut.']
        self.assertEqual((facts[0]['actor'], facts[0]['action'], facts[0]['causative']), ('user', 'have', 'yes'))
        self.assertEqual((facts[0]['tense'], facts[0]['aspect'], facts[0]['modal_meaning']), ('present', 'simple', 'obligation'))
        self.assertEqual((facts[1]['actor'], facts[1]['object'], facts[1]['causer'], facts[1]['status']),
                         ('unknown', 'my hair', 'user', 'required'))
        facts = self.examples['I want to have my hair cut.']
        self.assertEqual((facts[0]['operation'], facts[1]['actor'], facts[1]['operation']), ('add', 'unknown', 'none'))

    def test_causative_questions_do_not_assert_service_completion(self):
        facts = self.examples['Have you had your hair cut?']
        self.assertEqual([f['status'] for f in facts], ['questioned', 'questioned'])
        self.assertEqual((facts[0]['aspect'], facts[1]['aspect'], facts[1]['tense']), ('perfect', 'simple', 'nonfinite'))
        self.assertEqual(facts[1]['participle_form'], 'past')
        facts = self.examples['When did the Smiths have their house redecorated?']
        self.assertEqual((facts[0]['time'], facts[1]['object'], facts[1]['causer']), ('unknown', "Smiths' house", 'Smiths'))
        self.assertEqual(self.examples['She has had a new dress made.'][1]['status'], 'reported')

    def test_stop_gerund_and_purpose_infinitive_preserve_different_actions(self):
        facts = self.examples['He stopped reading.']
        self.assertEqual((facts[0]['object'], facts[1]['verb_form'], facts[1]['status']), ('reading', 'gerund', 'ceased'))
        facts = self.examples['He stopped to read an advertisement.']
        self.assertEqual(facts[0]['purpose'], 'to read an advertisement')
        self.assertEqual((facts[1]['syntactic_function'], facts[1]['status']), ('purpose', 'intended'))
        self.assertEqual(self.examples['Stop shouting.'][1]['status'], 'unwanted')

    def test_remember_and_regret_keep_retrospective_and_prospective_forms(self):
        remembered = self.examples['I remember having a rest there.'][1]
        self.assertEqual((remembered['action'], remembered['aspect'], remembered['temporal_relation']),
                         ('have', 'simple', 'before main verb'))
        fact = self.examples['Remember to meet him at the station.'][1]
        self.assertEqual((fact['verb_form'], fact['status'], fact['temporal_relation']), ('infinitive', 'required', 'after main verb'))
        fact = self.examples['I regret telling her the truth.'][1]
        self.assertEqual(fact['temporal_relation'], 'before main verb')
        fact = self.examples['I regret to tell you that he has fallen ill.'][1]
        self.assertEqual((fact['verb_form'], fact['temporal_relation']), ('infinitive', 'simultaneous with main verb'))
        self.assertEqual(self.examples["You won't regret having a rest there."][1]['aspect'], 'simple')

    def test_try_experiment_and_effort_do_not_assert_success(self):
        facts = self.examples['Try speaking French with him.']
        self.assertEqual((facts[0]['method'], facts[1]['verb_form'], facts[1]['status']), ('experiment', 'gerund', 'advised'))
        facts = self.examples['I tried to speak Italian with him but I failed.']
        self.assertEqual((facts[0]['method'], facts[1]['verb_form'], facts[1]['status']), ('effort', 'infinitive', 'hypothetical'))
        self.assertEqual(facts[-1]['action'], 'fail')
        self.assertEqual(self.examples['Try doing it in the evening.'][1]['object'], 'it')

    def test_used_to_distinguishes_past_habit_from_current_familiarity(self):
        facts = self.examples["I'm used to living in a small flat."]
        self.assertEqual((facts[0]['tense'], facts[1]['tense'], facts[1]['verb_form']), ('present', 'nonfinite', 'gerund'))
        fact = self.examples['I used to live in a small flat.'][0]
        self.assertEqual((fact['tense'], fact['modality'], fact['status']), ('past', 'used to', 'habitual'))
        facts = self.examples["I haven't yet got used to living in this house."]
        self.assertEqual((facts[0]['polarity'], facts[1]['status']), ('negative', 'hypothetical'))
        facts = self.examples["I used to drink a lot of coffee but now I don't."]
        self.assertEqual((facts[0]['tense'], facts[1]['tense'], facts[1]['action'], facts[1]['polarity']),
                         ('past', 'present', 'drink', 'negative'))

    def test_busy_and_permissions_keep_nonfinite_status_and_controller(self):
        self.assertEqual(self.examples['He is busy reading.'][1]['status'], 'reported')
        self.assertEqual(self.examples['He is too busy to read.'][1]['status'], 'impossible')
        generic = self.examples["The lecturer doesn't allow missing classes."][1]
        specific = self.examples["The teacher doesn't allow the children to miss classes."][1]
        self.assertEqual((generic['actor'], generic['verb_form'], generic['status'], generic['polarity']),
                         ('unknown', 'gerund', 'required', 'negative'))
        self.assertEqual((specific['actor'], specific['verb_form'], specific['status'], specific['polarity']),
                         ('children', 'infinitive', 'required', 'negative'))

    def test_participles_keep_nonfinite_form_and_relative_time(self):
        fact = self.examples['The boy playing in the garden is my son\'s friend.'][0]
        self.assertEqual((fact['actor'], fact['verb_form'], fact['participle_form'], fact['tense'], fact['aspect']),
                         ('boy', 'participle', 'present', 'nonfinite', 'simple'))
        self.assertEqual(fact['temporal_relation'], 'simultaneous with main verb')
        fact = self.examples['Having finished the article he went to bed.'][0]
        self.assertEqual((fact['aspect'], fact['temporal_relation']), ('perfect', 'before main verb'))
        facts = self.examples['Not having found the necessary papers he felt angry.']
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity']), ('negative', 'positive'))

    def test_passive_participles_preserve_author_recipient_and_target(self):
        fact = self.examples['He was looking at the pictures drawn by the child.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['voice'], fact['participle_form']), ('child', 'pictures', 'passive', 'past'))
        facts = self.examples['Having been taught good manners he was admired by everybody.']
        self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[0]['recipient'], facts[0]['aspect']),
                         ('unknown', 'good manners', 'he', 'perfect'))
        self.assertEqual((facts[1]['actor'], facts[1]['object']), ('everybody', 'he'))
        fact = self.examples['Being looked at she felt confused.'][0]
        self.assertEqual((fact['actor'], fact['target'], fact['participle_form']), ('unknown', 'she', 'present'))

    def test_passive_participle_negative_does_not_make_nested_request_actual(self):
        facts = self.examples['Not having been asked to come in he felt hurt.']
        self.assertEqual((facts[0]['voice'], facts[0]['aspect'], facts[0]['polarity']), ('passive', 'perfect', 'negative'))
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('he', 'hypothetical'))
        facts = self.examples["Not having been told the truth he didn't know how to act."]
        self.assertEqual((facts[0]['recipient'], facts[1]['polarity'], facts[2]['status']), ('he', 'negative', 'questioned'))

    def test_past_participle_does_not_imply_perfect_aspect_or_passive_voice(self):
        fact = self.examples['He likes to wear worn-out clothes.'][-1]
        self.assertEqual((fact['actor'], fact['action'], fact['participle_form'], fact['voice'], fact['aspect']),
                         ('clothes', 'wear out', 'past', 'active', 'simple'))
        fact = self.examples['The clothes sold here are expensive.'][0]
        self.assertEqual((fact['voice'], fact['aspect'], fact['temporal_relation']), ('passive', 'simple', 'simultaneous with main verb'))
        fact = self.examples['At last I found the missing page.'][1]
        self.assertEqual((fact['actor'], fact['action'], fact['complement']), ('page', 'be', 'missing'))

    def test_nonfinite_slots_bind_agreement_lemmas_and_owned_causative_objects(self):
        facts = self.examples['They seem to be sleeping now.']
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['action']), ('they', 'they', 'sleep'))
        facts = self.examples['She must have her clock fixed.']
        self.assertEqual((facts[0]['object'], facts[1]['actor'], facts[1]['action'], facts[1]['causer']),
                         ('her clock', 'unknown', 'fix', 'she'))
        self.assertEqual(self.examples['He stopped to read a note.'][0]['purpose'], 'to read a note')
        self.assertNotIn('They seems to be sleeping now.', self.examples)
        self.assertNotIn('He stopped to read an note.', self.examples)
        facts = self.examples['I must have my watch repaired.']
        self.assertEqual((facts[0]['tense'], facts[0]['aspect']), ('present', 'simple'))

    def test_nonfinite_slots_preserve_passive_actors_and_gerund_owners(self):
        fact = self.examples['He was looking at the pictures drawn by the girl.'][1]
        self.assertEqual((fact['actor'], fact['object']), ('girl', 'pictures'))
        fact = self.examples['Being reconstructed the hotel was empty.'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['voice']), ('unknown', 'hotel', 'passive'))
        facts = self.examples["I don't mind their phoning me."]
        self.assertEqual((facts[0]['actor'], facts[1]['actor'], facts[1]['object']), ('user', 'they', 'user'))
        facts = self.examples['Do you mind their joining you?']
        self.assertEqual((facts[1]['actor'], facts[1]['object']), ('they', 'addressee'))
        self.assertNotIn('Do you mind your joining you?', self.examples)

    def test_lessons_76_80_transformations_and_variants_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u76-e1-04-task', 'u76-e1-04'), ('u76-e2-03-task', 'u76-e2-03'),
                              ('u76-t23', 'u76-e8-03-task'), ('u76-e10-a02a', 'u76-e10-a02b'),
                              ('u77-t09', 'u77-t11'), ('u77-t10', 'u77-t12'),
                              ('u77-e1-a08-task', 'u77-e1-a08'), ('u78-t18', 'u78-t19'),
                              ('u78-e3-02-task-a', 'u78-e3-02'),
                              ('u78-e4-01-watch-repair', 'u78-e4-01-clock-fix'),
                              ('u79-t01', 'u79-t02'), ('u79-t15', 'u79-t16'),
                              ('u80-e2-07-task', 'u80-e2-07'), ('u80-e4-03-task', 'u80-e4-03')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_all_84_lessons_have_source_templates_and_no_pending_work(self):
        self.assertEqual(self.library.coverage['completed_units'], list(range(1, 85)))
        self.assertEqual(self.library.coverage['pending_units'], [])
        self.assertEqual({u: len(self.library.units[u]['templates']) for u in range(81, 85)},
                         {81: 73, 82: 154, 83: 91, 84: 100})

    def test_perception_separates_observer_from_whole_event_and_process(self):
        whole = self.examples['I saw him buy a newspaper.']
        process = self.examples['I saw him buying a newspaper.']
        for facts in (whole, process):
            self.assertEqual([(f['actor'], f['action']) for f in facts],
                             [('user', 'see'), ('he', 'buy')])
            self.assertEqual((facts[1]['tense'], facts[1]['aspect'], facts[1]['status']),
                             ('nonfinite', 'simple', 'reported'))
        self.assertEqual((whole[1]['verb_form'], whole[1]['marker'], whole[1]['event_view']),
                         ('infinitive', 'bare', 'whole'))
        self.assertEqual((process[1]['verb_form'], process[1]['participle_form'], process[1]['event_view']),
                         ('participle', 'present', 'process'))

    def test_complex_object_keeps_desired_actor_and_negative_complement(self):
        facts = self.examples['I want you to close the window.']
        self.assertEqual([(f['actor'], f['action'], f['operation']) for f in facts],
                         [('user', 'want', 'none'), ('addressee', 'close', 'none')])
        self.assertEqual(facts[1]['status'], 'intended')
        facts = self.examples['He expects us not to be late.']
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity'], facts[1]['status']),
                         ('positive', 'negative', 'expected'))
        self.assertEqual(facts[1]['actor'], 'we')
        self.assertEqual(self.examples['Make him go there.'][1]['status'], 'required')
        self.assertEqual(self.examples['Let me pass.'][1]['status'], 'permitted')

    def test_hearsay_and_realization_do_not_become_direct_perception(self):
        facts = self.examples['I hear you speak English.']
        self.assertEqual(facts[0]['evidence'], 'hearsay')
        self.assertEqual((facts[1]['actor'], facts[1]['tense'], facts[1]['status']),
                         ('addressee', 'present', 'reported'))
        self.assertNotIn('event_view', facts[1])
        self.assertNotIn('verb_form', facts[1])
        heard = self.examples['I heard them speaking English.'][1]
        self.assertEqual((heard['actor'], heard['tense'], heard['event_view']),
                         ('they', 'nonfinite', 'process'))
        facts = self.examples["I saw that he didn't understand anything."]
        self.assertEqual(facts[0]['sense'], 'realize')
        self.assertEqual((facts[1]['tense'], facts[1]['polarity']), ('past', 'negative'))

    def test_questions_about_perception_do_not_assert_the_embedded_event(self):
        facts = self.examples['Did you hear the child reading well?']
        self.assertEqual([f['status'] for f in facts], ['questioned', 'questioned'])
        facts = self.examples['Did you hear how he sings?']
        self.assertEqual((facts[1]['manner'], facts[1]['tense'], facts[1]['status']),
                         ('unknown', 'present', 'questioned'))
        self.assertNotIn('event_view', facts[1])

    def test_last_lessons_do_not_resolve_pronouns_from_neighbouring_stories(self):
        for text, index, role, value in [
            ('I want you to open it.', 1, 'object', 'it'),
            ("I'd like him to do the shopping.", 1, 'actor', 'he'),
            ('Make him learn them.', 1, 'object', 'them'),
            ('I saw him buy it.', 1, 'object', 'it'),
            ('The mother watched him.', 0, 'object', 'him'),
            ("Don't forget to do it.", 1, 'object', 'it'),
            ('We are still painting it.', 0, 'object', 'it'),
            ('Neither were mine.', 0, 'actor', 'mine'),
            ('Neither have his friends.', 0, 'actor', 'his friends'),
            ('So is that one.', 0, 'actor', 'that one')]:
            with self.subTest(text=text):
                self.assertEqual(self.examples[text][index][role], value)

    def test_coordinated_perception_preserves_both_actions_and_local_references(self):
        facts = self.examples['I heard him open the door and come into the room.']
        self.assertEqual([(f['actor'], f['action']) for f in facts],
                         [('user', 'hear'), ('he', 'open'), ('he', 'come into')])
        facts = self.examples['I noticed her pick up something and put it into her pocket.']
        self.assertEqual([(f['actor'], f['object']) for f in facts[1:]],
                         [('she', 'something'), ('she', 'something')])
        self.assertEqual(facts[2]['destination'], 'her pocket')
        facts = self.examples['I noticed him walking towards me and stopped to shake hands.']
        self.assertEqual([(f['actor'], f['action']) for f in facts],
                         [('user', 'notice'), ('he', 'walk'), ('user', 'stop'), ('user', 'shake hands')])
        self.assertEqual(facts[3]['status'], 'intended')

    def test_observed_attempt_does_not_imply_successful_teaching(self):
        facts = self.examples['I watched Mike trying to teach the child to skate.']
        self.assertEqual([(f['actor'], f['action'], f['status']) for f in facts],
                         [('user', 'watch', 'asserted'), ('Mike', 'try', 'reported'),
                          ('Mike', 'teach', 'hypothetical'), ('child', 'skate', 'hypothetical')])
        facts = self.examples['I like children and I like to watch them playing.']
        self.assertEqual((facts[-1]['actor'], facts[-1]['status']), ('children', 'hypothetical'))

    def test_nonfinite_negation_scope_and_arrangements_keep_events_unasserted(self):
        facts = self.examples["He doesn't seem to have recovered after the illness."]
        self.assertEqual((facts[0]['polarity'], facts[1]['polarity'], facts[1]['status']),
                         ('negative', 'positive', 'hypothetical'))
        self.assertEqual(facts[1]['aspect'], 'perfect')
        facts = self.examples["I didn't expect him to forget about your birthday."]
        self.assertEqual((facts[0]['polarity'], facts[1]['status']), ('negative', 'hypothetical'))
        facts = self.examples["I've arranged for you to meet the president."]
        self.assertEqual((facts[1]['actor'], facts[1]['status']), ('addressee', 'planned'))
        facts = self.examples["You can't make an omelette without breaking eggs."]
        self.assertEqual((facts[1]['polarity'], facts[1]['status']), ('negative', 'hypothetical'))

    def test_prohibitions_keep_the_specific_or_generic_actor(self):
        for text in ["They don't let us park cars here.", "They don't allow us to park cars here."]:
            fact = self.examples[text][1]
            self.assertEqual((fact['actor'], fact['action'], fact['polarity'], fact['status']),
                             ('we', 'park', 'negative', 'required'))
        fact = self.examples["They don't allow talking loudly here."][1]
        self.assertEqual((fact['actor'], fact['verb_form'], fact['polarity']),
                         ('unknown', 'gerund', 'negative'))

    def test_forgetting_and_failure_differ_from_successful_managing(self):
        fact = self.examples['I forgot to remind him of the meeting.'][1]
        self.assertEqual((fact['status'], fact['outcome']), ('hypothetical', 'not completed'))
        facts = self.examples['The inspector asked me to show my ticket but I failed to do it.']
        self.assertEqual(facts[1]['status'], 'required')
        self.assertEqual((facts[3]['object'], facts[3]['outcome']), ('my ticket', 'not completed'))
        fact = self.examples['On arriving at the station I managed to find a porter at once.'][-1]
        self.assertEqual((fact['action'], fact['status'], fact['outcome']), ('find', 'reported', 'completed'))

    def test_perception_slots_bind_case_lemma_and_embedded_actor(self):
        for text, form, view in [('I heard her leave.', 'infinitive', 'whole'),
                                 ('I heard her leaving.', 'participle', 'process')]:
            facts = self.examples[text]
            self.assertEqual((facts[0]['actor'], facts[0]['object'], facts[1]['actor']),
                             ('user', 'her', 'she'))
            self.assertEqual((facts[1]['action'], facts[1]['verb_form'], facts[1]['event_view']),
                             ('leave', form, view))
        facts = self.examples['I watched Robert trying to teach the child to skate.']
        self.assertEqual([f['actor'] for f in facts], ['user', 'Robert', 'Robert', 'child'])
        fact = self.examples['I insist on his refusing their idea.'][1]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('he', 'their idea', 'required'))
        self.assertNotIn('I heard she leave.', self.examples)

    def test_still_negation_distinguishes_continued_absence_from_cessation(self):
        absence = self.examples["He still hasn't taken a decision."][0]
        cessation = self.examples["He isn't still waiting."][0]
        self.assertEqual((absence['polarity'], absence['continuation'], absence['negation_scope']),
                         ('negative', 'still', 'predicate'))
        self.assertEqual((cessation['polarity'], cessation['continuation'], cessation['negation_scope']),
                         ('negative', 'still', 'continuation'))
        facts = self.examples["It hasn't stopped snowing yet."]
        self.assertEqual((facts[0]['polarity'], facts[1]['actor'], facts[1]['continuation']),
                         ('negative', 'none', 'yes'))

    def test_already_and_yet_do_not_override_aspect_or_question_status(self):
        fact = self.examples['He already knows about it.'][0]
        self.assertEqual((fact['tense'], fact['aspect'], fact['completion']), ('present', 'simple', 'already'))
        fact = self.examples['Has he come already?'][0]
        self.assertEqual((fact['aspect'], fact['status'], fact['attitude']), ('perfect', 'questioned', 'surprise'))
        fact = self.examples["Hasn't he come yet?"][0]
        self.assertEqual((fact['polarity'], fact['status'], fact['completion']), ('negative', 'questioned', 'yet'))
        facts = self.examples['He has already written the test but he is still doing the checking up.']
        self.assertEqual([(f['action'], f['aspect']) for f in facts], [('write', 'perfect'), ('do', 'continuous')])
        self.assertEqual((facts[0]['completion'], facts[1]['continuation']), ('already', 'still'))

    def test_temporal_markers_preserve_plans_and_idioms(self):
        fact = self.examples["I'm still going to meet them."][0]
        self.assertEqual((fact['action'], fact['object'], fact['tense'], fact['status']),
                         ('meet', 'them', 'future', 'planned'))
        fact = self.examples['They have already made it up.'][0]
        self.assertEqual((fact['action'], fact['sense'], fact['reciprocal']), ('make up', 'reconcile', 'yes'))
        self.assertEqual(self.examples["Yes, I'm ready."][0]['polarity'], 'positive')
        self.assertNotIn('completion', self.examples["Yes, I'm ready."][0])

    def test_temporal_subject_slots_bind_auxiliaries_and_possessives(self):
        fact = self.examples["They haven't washed their faces yet."][0]
        self.assertEqual((fact['actor'], fact['object']), ('they', 'their faces'))
        fact = self.examples['Am I still riding my bicycle?'][0]
        self.assertEqual((fact['actor'], fact['object'], fact['status']), ('user', 'my bicycle', 'questioned'))
        self.assertEqual(self.examples['I am not still waiting.'][0]['actor'], 'user')
        self.assertEqual(self.examples['They already know about it.'][0]['actor'], 'they')
        for text in ["I isn't still waiting.", 'They already knows about it.', 'Are he here yet?',
                     "She haven't washed her face yet."]:
            self.assertNotIn(text, self.examples)

    def test_additive_adverbs_preserve_polarity_and_intentions(self):
        for text, addition, polarity in [('I also know Steve.', 'also', 'positive'),
                                         ('I met him yesterday too.', 'too', 'positive'),
                                         ("I don't know him either.", 'either', 'negative')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['addition'], fact['polarity']), (addition, polarity))
            self.assertNotIn('degree', fact)
        fact = self.examples['I want to go there too.'][0]
        self.assertEqual((fact['action'], fact['intention'], fact['operation']), ('go', 'go', 'add'))
        fact = self.examples["You shouldn't ask him about it either."][0]
        self.assertEqual((fact['polarity'], fact['status'], fact['modal_meaning']), ('negative', 'advised', 'advice'))

    def test_short_additive_replies_leave_the_absent_predicate_unknown(self):
        for text, auxiliary, actor, polarity in [('So were we.', 'were', 'we', 'positive'),
                                                ('Neither have his friends.', 'have', 'his friends', 'negative'),
                                                ('Neither can I.', 'can', 'user', 'negative')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['auxiliary'], fact['actor'], fact['polarity']), (auxiliary, actor, polarity))
            for field in ('action', 'object', 'aspect', 'voice'):
                self.assertEqual(fact[field], 'unknown')
        self.assertEqual(self.examples['Neither can I.'][0]['modal_meaning'], 'unknown')

    def test_short_reply_slots_bind_auxiliaries_without_inventing_predicates(self):
        for text, actor, auxiliary in [('Neither are they.', 'they', 'are'),
                                       ('So does she.', 'she', 'does'),
                                       ('So have I.', 'user', 'have'),
                                       ('So was he.', 'he', 'was')]:
            fact = self.examples[text][0]
            self.assertEqual((fact['actor'], fact['auxiliary'], fact['action']), (actor, auxiliary, 'unknown'))
        self.assertNotIn('Neither am they.', self.examples)
        self.assertNotIn('So does we.', self.examples)

    def test_last_lesson_lexical_slots_preserve_meaning_and_determiners(self):
        facts = self.examples['Margaret can also drive a bus.']
        self.assertEqual((facts[0]['actor'], facts[0]['action'], facts[0]['object']), ('Margaret', 'drive', 'bus'))
        self.assertNotIn('Margaret can also drive a bicycle.', self.examples)
        self.assertEqual(self.examples['I also planted some flowers last year.'][0]['quantity'], 'some')
        self.assertEqual(self.examples["I didn't plant any flowers last year either."][0]['quantity'], 'any')

    def test_lessons_81_84_transformations_and_optional_variants_share_families(self):
        templates = {t['id']: t for t in self.templates}
        for first, second in [('u81-t05', 'u81-t08'), ('u81-t06', 'u81-t09'),
                              ('u81-t12a', 'u81-t12b'), ('u81-e1-01q', 'u81-e1-01'),
                              ('u82-e1-a02-task1', 'u82-e1-a02'), ('u82-e1-b14-task1', 'u82-e1-b14'),
                              ('u82-e2-a04', 'u82-e2-a05'), ('u83-e3-01-task', 'u83-e3-01q'),
                              ('u83-e3-01q', 'u83-e3-01n'), ('u83-e5-01a', 'u83-e5-01b'),
                              ('u84-e1-a01-task', 'u84-e1-a01'), ('u84-e3-11-task', 'u84-e3-11'),
                              ('u84-e4-03a', 'u84-e4-03b'), ('u84-t13', 'u84-e4-04a')]:
            self.assertEqual(templates[first]['family'], templates[second]['family'])

    def test_invalid_binding_or_compatibility_is_rejected_on_loading(self):
        catalogue = book.read_json(book.DEFAULT_LIBRARY / 'fact_templates.json')
        vocab = book.read_json(book.DEFAULT_LIBRARY / 'slot_values.json')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'slot_values.json').write_text(json.dumps(vocab), encoding='utf-8')
            (out / 'sentences.json').write_bytes((book.DEFAULT_LIBRARY / 'sentences.json').read_bytes())
            catalogue['lessons'][0]['templates'][0]['original']['missing'] = 'word'
            (out / 'fact_templates.json').write_text(json.dumps(catalogue), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'bindings do not match slots'):
                book.Library(out)

if __name__ == '__main__':
    unittest.main()
