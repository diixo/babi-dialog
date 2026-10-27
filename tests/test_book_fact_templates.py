from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book
import book_srl_annotations as annotations
from book_srl_corpus import CorpusLibrary


class FactTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = book.Library()
        cls.corpus = CorpusLibrary(annotations.DEFAULT_LIBRARY)
        cls.templates = [t for u in cls.library.units.values() for t in u['templates']]
        cls.examples = {text: facts for u in range(1, 13)
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
        self.assertNotIn('predicate', output)
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
            first = book.build(self.library, out, per_unit=True)
            data = {p.relative_to(out): p.read_bytes() for p in out.rglob('*.txt')}
            self.assertEqual(book.build(self.library, out, per_unit=True), first)
            self.assertEqual(data, {p.relative_to(out): p.read_bytes() for p in out.rglob('*.txt')})
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
            with self.assertRaisesRegex(ValueError, 'no prepared templates'):
                book.build(self.library, out, [84])
            lib = deepcopy(self.library)
            duplicate = deepcopy(lib.units[1]['templates'][0])
            duplicate['family'] += '-conflict'
            duplicate['facts'][0]['actor'] = 'wrong'
            lib.units[1]['templates'].append(duplicate)
            with self.assertRaisesRegex(ValueError, 'Conflicting facts'):
                book.build(lib, out, [1])
            self.assertEqual(path.read_text(encoding='utf-8'), 'existing')
            self.assertFalse((out / 'valid.txt').exists())

    def test_default_build_excludes_partial_lessons_but_reports_their_pending_work(self):
        lib = deepcopy(self.library)
        lib.units[13]['templates'] = [deepcopy(lib.units[12]['templates'][0])]
        stats = book.build(lib)
        self.assertNotIn(13, stats['units'])
        self.assertIn(13, lib.coverage['pending_units'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            book.build(lib, units=[13])

    def test_invalid_binding_or_compatibility_is_rejected_on_loading(self):
        catalogue = annotations.read_json(book.DEFAULT_LIBRARY / 'fact_templates.json')
        vocab = annotations.read_json(book.DEFAULT_LIBRARY / 'slot_values.json')
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
