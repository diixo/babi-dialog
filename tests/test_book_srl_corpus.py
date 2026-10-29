from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import build_book_srl as book
from book_srl_corpus import CorpusLibrary, CORPUS_NAME
from prepare_book_templates import prepare


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = CorpusLibrary(book.DEFAULT_LIBRARY)

    def test_json_contains_only_english_sentences_with_manual_facts(self):
        self.assertEqual(len(self.library.records), 8046)
        self.assertEqual(self.library.completed_units, list(range(1, 85)))
        serialized = (book.DEFAULT_LIBRARY / CORPUS_NAME).read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'[\u0400-\u04ff]', serialized))
        excluded = {'Waiter!', 'A ceiling.', 'A dress of silk.', 'A hospital.',
                    'At the President Hotel.', 'Two coffees and a tea, please.',
                    'None.', 'Not now.', 'Nowhere.', 'Anywhere.', 'How about the others?',
                    'Either.', 'Neither.', 'Both.', 'And you?',
                    'a tall slim young girl', 'delicious French strawberry ice-cream',
                    'Tea, please.', 'Three or four lectures usually.', 'Yes, thanks.',
                    'Not yet.', 'Quite early, at seven.', 'By underground.',
                    'Why?', 'Oh, really?', 'Congratulations!', 'His secretary',
                    'who that woman is', 'if / whether Mary speaks French',
                    'Just two.', 'Not bad, thanks.', 'Just one in fact.',
                    'What a funny accent!', 'What difficult work!', 'What dirty clothes!',
                    'he speaks', 'she kisses', 'to make — making',
                    'A friend of mine.', 'No wonder.', 'Nothing special.',
                    'With pleasure!', 'When the teacher checks our papers.',
                    'What wonderful news!', 'Me too.', 'At 5.', 'No, sorry.',
                    'to copy — copied', 'created', 'the longest poem — to learn',
                    "I've just took this book from the library.",
                    'He hasn\'t never got a bad mark.',
                    'Oh, for five years already!', 'Oh, since our childhood.',
                    'No, for about five minutes.', 'Oh, for several days.',
                    'For about two hours.', 'Fifteen.', 'About six hours.',
                    'for a long time', 'long ago', 'have (has) been',
                    'I know Ben since my childhood.', "I've heard the news the other day.",
                    'We asked they to stay with us.', 'Her is the red car.',
                    'By the time he was 35.', 'Funny!', 'A lady.', 'At eighteen.',
                    'He had been finished his book by December last year.',
                    "Nick tells that it's time to go.", 'Myra asked why had I lied to her.',
                    'Mother told Maude that she clean her room.', 'A pleasant summer?',
                    'The radio.', 'What?', 'Present Indefinite', 'was (were) done',
                    'Oh, dear!', 'Sorry!', 'Nothing.', 'I wish ...', 'If I were.',
                    'Has the baby be fed yet?', 'When will be the picture finished?',
                    'The question has been being discussed for two hours already.',
                    'The message was written by a pencil.', 'take pills; write a book',
                    'can be doing', 'must have done', 'to be (un)likely to',
                    'Can (could) I ask you a question?', 'He asks the stewardess:',
                    'Children under 16 ... to see the film.',
                    'may (might) be doing', 'need not (needn\'t)', 'having been done',
                    'interviewing', 'being interviewed', 'What do you think',
                    'No, you', "You should't go there alone.",
                    'We could sent you the answer by post.',
                    'to be used to doing', 'to have something done', 'having been done',
                    'the exhibition, to visit', 'be busy to do',
                    'He is too week to walk.', 'He is happy to have been invited',
                    'I must have my watch (clock) repaired (fixed).',
                    "I don't mind (I don't object to) being criticized.",
                    'No, not yet.', 'Nether do we.',
                    'I hear (that) they have just moved to a new flat.',
                    "She hasn't washed (her face) yet."}
        for record in self.library.records:
            self.assertEqual(set(record), {'id', 'unit', 'text', 'facts', 'predicate_spans'})
            self.assertNotIn(record['text'], excluded)
            self.assertNotIn('.......', record['text'])
            self.assertNotIn('<br>', record['text'])
            self.assertTrue(record['facts'])
        self.assertTrue(any(r['text'] == 'Have you seen them?' for r in self.library.records))
        self.assertTrue(any(r['text'] == "Yes, she does." for r in self.library.records))

    def test_clutter_duplicate_ids_and_unannotated_examples_are_rejected(self):
        original = json.loads((book.DEFAULT_LIBRARY / CORPUS_NAME).read_text(encoding='utf-8'))
        cases = [
            lambda data: data['sentences'][0].update(text='\u0423\u0440\u043e\u043a 1'),
            lambda data: data['sentences'][0].update(text='I have ....... book.'),
            lambda data: data['sentences'][0].update(raw='original paragraph'),
            lambda data: data['sentences'][0].update(facts=[]),
            lambda data: data['sentences'][0].update(unit=85),
            lambda data: data['sentences'].append(deepcopy(data['sentences'][0])),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / CORPUS_NAME
            for change in cases:
                data = deepcopy(original)
                change(data)
                path.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(ValueError):
                    CorpusLibrary(Path(folder))

    def test_preparation_and_build_read_json_only_and_preserve_generated_results(self):
        expected = book.build(book.Library())
        real_open = Path.open
        reads = []
        def json_only(path, mode='r', *args, **kwargs):
            if 'r' in mode:
                if path.suffix != '.json':
                    raise AssertionError(f'Non-JSON input: {path}')
                reads.append(path.name)
            return real_open(path, mode, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('sentences.json', 'fact_templates.json', 'slot_values.json'):
                (directory / name).write_bytes((book.DEFAULT_LIBRARY / name).read_bytes())
            original = (directory / 'fact_templates.json').read_bytes()
            with patch.object(Path, 'open', json_only):
                prepare(directory)
                actual = book.build(book.Library(directory), directory / 'out', per_unit=True)
            self.assertEqual(actual, expected)
            self.assertEqual(original, (directory / 'fact_templates.json').read_bytes())
            self.assertTrue((directory / 'out' / 'train.txt').exists())
            self.assertEqual(set(reads), {'sentences.json', 'fact_templates.json', 'slot_values.json'})

    def test_manual_fact_changes_cannot_silently_keep_outdated_template_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('sentences.json', 'fact_templates.json', 'slot_values.json'):
                (directory / name).write_bytes((book.DEFAULT_LIBRARY / name).read_bytes())
            path = directory / CORPUS_NAME
            data = json.loads(path.read_text(encoding='utf-8'))
            data['sentences'][0]['facts'][0]['actor'] = 'someone else'
            path.write_text(json.dumps(data), encoding='utf-8')
            original = (directory / 'fact_templates.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'manual facts'):
                prepare(directory)
            self.assertEqual(original, (directory / 'fact_templates.json').read_bytes())

    def test_new_annotated_english_sentence_gets_a_fixed_template(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('sentences.json', 'fact_templates.json', 'slot_values.json'):
                (directory / name).write_bytes((book.DEFAULT_LIBRARY / name).read_bytes())
            path = directory / CORPUS_NAME
            data = json.loads(path.read_text(encoding='utf-8'))
            data['sentences'].append(dict(id='manual-english-example', unit=1,
                text='Mary reads.', predicate_spans=[[[5,10]]], facts=[dict(actor='mary', predicate='reads', action='read',
                    intention='none', operation='none', object='none')]))
            path.write_text(json.dumps(data), encoding='utf-8')
            prepare(directory)
            library = book.Library(directory)
            examples = {text:facts for _,text,facts in library.generate(1)}
            self.assertEqual(examples['Mary reads.'][0]['actor'], 'mary')
            template = next(t for t in library.units[1]['templates'] if t['id']=='manual-english-example')
            self.assertEqual(template['slots'], {})


if __name__ == '__main__':
    unittest.main()
