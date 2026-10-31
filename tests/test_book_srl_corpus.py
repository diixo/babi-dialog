from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import build_book_srl as book
from book_srl_corpus import CorpusLibrary, CORPUS_NAME


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
            self.assertEqual(set(record), {'id', 'unit', 'text', 'facts', 'predicate_spans', 'fragment_spans', 'verb_spans'})
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
            lambda data: data['sentences'][0].pop('fragment_spans'),
            lambda data: data['sentences'][0].update(verb_spans=[]),
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
                with redirect_stdout(io.StringIO()):
                    book.main(['--library', str(directory), '--out', str(directory / 'out')])
                actual = book.build(book.Library(directory), directory / 'out')
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
                book.synchronize_templates(directory)
            self.assertEqual(original, (directory / 'fact_templates.json').read_bytes())

    def test_new_annotated_english_sentence_gets_a_fixed_template(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('sentences.json', 'fact_templates.json', 'slot_values.json'):
                (directory / name).write_bytes((book.DEFAULT_LIBRARY / name).read_bytes())
            path = directory / CORPUS_NAME
            data = json.loads(path.read_text(encoding='utf-8'))
            data['sentences'].append(dict(id='manual-english-example', unit=1,
                text='Mary reads.', predicate_spans=[[[5,10]]],
                fragment_spans=[[[0,10]]], verb_spans=[[[5,10]]], facts=[dict(actor='mary', predicate='reads', action='read',
                    intention='none', operation='none', object='none')]))
            path.write_text(json.dumps(data), encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                book.main(['--library', str(directory), '--out', str(directory / 'out')])
            library = book.Library(directory)
            examples = {text:facts for _,text,facts,_ in library.generate(1)}
            self.assertEqual(examples['Mary reads.'][0]['actor'], 'mary')
            template = next(t for t in library.units[1]['templates'] if t['id']=='manual-english-example')
            self.assertEqual(template['slots'], {})


class UnifiedBuildTests(unittest.TestCase):
    @staticmethod
    def record(identifier='first', name='Mary', unit=1):
        text=f'{name} reads.'
        return dict(id=identifier, unit=unit, text=text,
                    predicate_spans=[[[len(name)+1, len(text)-1]]],
                    fragment_spans=[[[0,len(text)-1]]],
                    verb_spans=[[[len(name)+1,len(text)-1]]],
                    facts=[dict(actor=name, predicate='reads', action='read',
                                intention='none', operation='none', object='none')])

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def fixture(self, directory):
        record=self.record()
        records=[record]+[self.record(f'unit-{unit}',f'Reader {unit}',unit)
                          for unit in range(2,85)]
        source=dict(schema_version=1, completed_units=list(range(1,85)), sentences=records)
        text='{person.word} reads.'
        template=dict(id=record['id'], family='reading-family', text=text,
                      slots={'person':'people'}, original={'person':'mary'}, constraints=[],
                      facts=[dict(record['facts'][0], actor='{person.word}')],
                      predicate_spans=[[[text.index('reads'), len(text)-1]]],
                      fragment_spans=[[[0,len(text)-1]]],
                      verb_spans=[[[text.index('reads'),len(text)-1]]],
                      source=dict(sentence=record['id'], unit=1, text=record['text']))
        lessons=[dict(unit=1,templates=[template])]
        for record in records[1:]:
            fixed=dict(id=record['id'],family=record['id'],text=record['text'],
                       slots={},original={},constraints=[],facts=deepcopy(record['facts']),
                       predicate_spans=deepcopy(record['predicate_spans']),
                       fragment_spans=deepcopy(record['fragment_spans']),
                       verb_spans=deepcopy(record['verb_spans']),
                       source=dict(sentence=record['id'],unit=record['unit'],text=record['text']))
            lessons.append(dict(unit=record['unit'],templates=[fixed]))
        catalogue=dict(schema_version=1,
                       coverage=dict(sentences=84,completed_units=list(range(1,85)),pending_units=[]),
                       lessons=lessons)
        vocabulary=dict(schema_version=1, domains={'people':dict(rule='Singular subjects', values=[
            dict(id='mary', word='Mary'), dict(id='nora', word='Nora')])})
        for name,value in [('sentences.json',source), ('fact_templates.json',catalogue), ('slot_values.json',vocabulary)]:
            self.write_json(directory/name, value)
        return source,catalogue,vocabulary

    @staticmethod
    def run_cli(directory, *options):
        with redirect_stdout(io.StringIO()) as output:
            result=book.main(['--library',str(directory),'--out',str(directory/'out'),*options])
        return result,output.getvalue()

    def test_one_command_synchronizes_added_removed_sentences_and_preserves_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            source,catalogue,_=self.fixture(directory)
            original=deepcopy(catalogue['lessons'][0]['templates'][0])
            removed=deepcopy(original)
            removed['id']='removed'
            catalogue['lessons'][0]['templates'].append(removed)
            source['sentences'].append(self.record('new','Dan'))
            self.write_json(directory/'sentences.json',source)
            self.write_json(directory/'fact_templates.json',catalogue)
            result,output=self.run_cli(directory)
            self.assertEqual(result,0)
            library=book.Library(directory)
            self.assertEqual(library.coverage['sentences'],85)
            self.assertEqual(library.units[1]['templates'][0],original)
            self.assertEqual([t['id'] for t in library.units[1]['templates']],['first','new'])
            self.assertEqual(library.units[1]['templates'][1]['slots'],{})
            text=''.join((directory/'out'/f'{split}.txt').read_text(encoding='utf-8')
                         for split in ('train','valid','test'))
            for sentence in ('Mary reads.','Nora reads.','Dan reads.'):
                self.assertIn('1 User: '+sentence,text)
            for record in source['sentences']:
                self.assertIn('1 User: '+record['text'],text)
            self.assertEqual({p.name for p in (directory/'out').iterdir()},
                             {'train.txt','valid.txt','test.txt'})
            for unit in range(1,85):
                count=2 if unit==1 else 1
                self.assertIn(f'{unit:02d}: {count} sentence templates; ready',output)
            self.assertIn('84/84 book units',output)
            self.assertIn('Fragments: 86 single-fragment (100.00%), 0 multi-fragment (0.00%)',output)

    def test_missing_catalogue_is_created_by_the_same_build_command(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            self.fixture(directory)
            (directory/'fact_templates.json').unlink()
            self.assertEqual(self.run_cli(directory)[0],0)
            library=book.Library(directory)
            self.assertEqual(library.units[1]['templates'][0]['slots'],{})
            self.assertEqual(book.build(library)['examples'],84)

    def test_annotation_flag_builds_both_complete_datasets_in_one_command(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            for name in ('sentences.json','fact_templates.json','slot_values.json',
                         'nouns.json','verbs.json','templates.json'):
                (directory/name).write_bytes((book.DEFAULT_LIBRARY/name).read_bytes())
            catalogue=directory/'fact_templates.json'
            before=(catalogue.read_bytes(),catalogue.stat().st_mtime_ns)
            out=directory/'out'
            with patch.object(book,'DEFAULT_LIBRARY',directory), patch.object(book,'DEFAULT_OUT',out):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(book.main(['--annotate']),0)
            self.assertEqual((catalogue.read_bytes(),catalogue.stat().st_mtime_ns),before)
            self.assertIn('84/84 book units',output.getvalue())
            self.assertEqual({p.name for p in out.iterdir()},
                             {'train.txt','valid.txt','test.txt','annotations'})
            self.assertEqual({p.name for p in (out/'annotations').iterdir()},
                             {'train.txt','valid.txt','test.txt'})
            for target,expected in ((out,12231),(out/'annotations',901)):
                stories=[]
                for split in ('train','valid','test'):
                    content=(target/f'{split}.txt').read_text(encoding='utf-8')
                    stories.extend(content.strip().split('\n\n'))
                self.assertEqual(len(stories),expected)
                self.assertTrue(all(story.startswith('1 User: ') for story in stories))
                self.assertTrue(all('Assistant:' not in story for story in stories))

    def test_annotation_flag_checks_both_libraries_before_writing_any_file(self):
        for problem in ('book facts','annotation placeholder','missing annotation file','incomplete annotation lessons'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
                directory=Path(folder)
                source,_,_=self.fixture(directory)
                # A valid source addition would require a catalogue update on success.
                source['sentences'].append(self.record('new','Dan'))
                if problem=='book facts':
                    source['sentences'][0]['facts'][0]['actor']='incorrect'
                self.write_json(directory/'sentences.json',source)
                for name in ('nouns.json','verbs.json','templates.json'):
                    (directory/name).write_bytes((book.DEFAULT_LIBRARY/name).read_bytes())
                if problem=='missing annotation file':
                    (directory/'nouns.json').unlink()
                elif problem in ('annotation placeholder','incomplete annotation lessons'):
                    path=directory/'templates.json'
                    data=book.read_json(path)
                    if problem=='annotation placeholder':
                        data['lessons'][-1]['templates'][0]['sentences'][0]['text']='{missing.field}.'
                    else:
                        data['lessons'].pop()
                    self.write_json(path,data)
                catalogue=directory/'fact_templates.json'
                before=(catalogue.read_bytes(),catalogue.stat().st_mtime_ns)
                out=directory/'out'
                snapshots={}
                for target in (out,out/'annotations'):
                    target.mkdir(parents=True,exist_ok=True)
                    for split in ('train','valid','test'):
                        path=target/f'{split}.txt'
                        path.write_text(f'existing {target.name} {split}',encoding='utf-8')
                        snapshots[path.relative_to(out)]=path.read_bytes()
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                    self.run_cli(directory,'--annotate')
                self.assertEqual(error.exception.code,2)
                self.assertEqual((catalogue.read_bytes(),catalogue.stat().st_mtime_ns),before)
                self.assertEqual({p.relative_to(out):p.read_bytes() for p in out.rglob('*.txt')},snapshots)

    def test_cli_accepts_library_output_and_annotation_options(self):
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as result:
            book.main(['--help'])
        self.assertEqual(result.exception.code,0)
        self.assertIn('--library',output.getvalue())
        self.assertIn('--out',output.getvalue())
        self.assertIn('--annotate',output.getvalue())
        for options in (['--units','1'],['--per-unit'],['--coverage'],['--validate-only'],
                        ['--mode','single'],['--list'],['--source','templates']):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                directory=Path(folder)
                with redirect_stderr(io.StringIO()) as error_output, self.assertRaises(SystemExit) as error:
                    self.run_cli(directory,*options)
                self.assertEqual(error.exception.code,2)
                self.assertIn('unrecognized arguments',error_output.getvalue())
                self.assertNotIn(options[0],output.getvalue())
                self.assertFalse(list(directory.iterdir()))

    def test_invalid_sync_or_incomplete_corpus_preserves_catalogue_and_existing_outputs(self):
        for problem in ('manual facts','duplicate','variant','conflict','incomplete','empty lesson',
                        'fragment evidence','verb evidence'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
                directory=Path(folder)
                source,catalogue,vocabulary=self.fixture(directory)
                if problem=='manual facts':source['sentences'][-1]['facts'][0]['actor']='someone else'
                elif problem=='duplicate':catalogue['lessons'][0]['templates'].append(deepcopy(catalogue['lessons'][0]['templates'][0]))
                elif problem=='variant':
                    vocabulary['domains']['people']['values'][1]['word']=''
                elif problem=='conflict':
                    record=self.record('conflicting','Nora')
                    record['facts'][0]['actor']='someone else'
                    source['sentences'].append(record)
                elif problem=='fragment evidence':
                    catalogue['lessons'][0]['templates'][0]['fragment_spans']=[[[0,1]]]
                elif problem=='verb evidence':
                    source['sentences'][-1]['verb_spans']=[[[0,6]]]
                else:
                    source['completed_units'].remove(84)
                    if problem=='empty lesson':source['sentences'].pop()
                for name,value in [('sentences.json',source), ('fact_templates.json',catalogue), ('slot_values.json',vocabulary)]:
                    self.write_json(directory/name,value)
                original=(directory/'fact_templates.json').read_bytes()
                out=directory/'out';out.mkdir()
                for split in ('train','valid','test'):(out/f'{split}.txt').write_text('existing '+split,encoding='utf-8')
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    self.run_cli(directory)
                self.assertEqual(error.exception.code,2)
                self.assertEqual((directory/'fact_templates.json').read_bytes(),original)
                for split in ('train','valid','test'):
                    self.assertEqual((out/f'{split}.txt').read_text(encoding='utf-8'),'existing '+split)

    def test_unchanged_build_does_not_rewrite_catalogue(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);self.fixture(directory)
            self.run_cli(directory)
            path=directory/'fact_templates.json'
            before=(path.read_bytes(),path.stat().st_mtime_ns)
            outputs={p.name:p.read_bytes() for p in (directory/'out').glob('*.txt')}
            self.run_cli(directory)
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),before)
            self.assertEqual({p.name:p.read_bytes() for p in (directory/'out').glob('*.txt')},outputs)


if __name__ == '__main__':
    unittest.main()
