from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import build_book_srl as book
from book_srl_corpus import CorpusLibrary


def groups(story):
    result = []
    for row in story.strip().splitlines()[1:]:
        number, _, body = row.partition(' ')
        question, answer, references = body.split('\t')
        if question == 'fragment?':
            result.append({'line': int(number)})
        result[-1][question[:-1]] = (answer, list(map(int, references.split())))
    return result


class ActorTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = CorpusLibrary(book.DEFAULT_LIBRARY)
        cls.annotations = book.AnnotationLibrary()

    def rendered(self, identifier):
        record = self.corpus.by_id[identifier]
        return groups(book.render(record['text'], record['facts'], record))

    def annotation(self, unit, text):
        return next(e for e in self.annotations.generate(unit) if e.text == text)

    def test_time_and_location_participants_do_not_overwrite_circumstances(self):
        cases = [
            ('u39-e3-04-task', 'time', 'spring', 'when', 'after autumn'),
            ('u05-8f52201baa7aa43e-1-v1-s1', 'location', 'Newcastle', 'where',
             'in the north-east of England'),
            ('u36-e1-12a', 'time', 'your birthday', 'when', 'unknown'),
            ('u01-6923b817c7e2a2af-1-v1-s1', 'location', 'station', 'where', 'unknown'),
        ]
        for identifier, kind, value, circumstance, context in cases:
            with self.subTest(identifier=identifier):
                record = self.corpus.by_id[identifier]
                index = next(i for i, f in enumerate(record['facts']) if f.get(kind) == value)
                fact = record['facts'][index]
                self.assertEqual((fact['actor'], fact[kind], fact[circumstance]), (kind, value, context))
                group = self.rendered(identifier)[index]
                self.assertEqual(group['actor'][0], kind)
                self.assertEqual(group[kind][0], value)
                self.assertEqual(group[circumstance][0], context)
                self.assertEqual(group['actor'][1], group[kind][1])

    def test_organizations_remain_agents_in_active_and_passive_sentences(self):
        for identifier in ('u60-t01a', 'u60-t01b'):
            group = self.rendered(identifier)[0]
            self.assertEqual(group['actor'], ('organization', [group['line']]))
            self.assertEqual(group['organization'], ('commission', [group['line']]))
            self.assertEqual(group['object'][0], 'my report')
            self.assertEqual(group['when'][0], 'yesterday')

    def test_shared_person_and_fronted_question_use_the_same_context_for_type_and_value(self):
        group = self.rendered('u24-e11-b07b')[2]
        self.assertEqual(group['person'], ('he', [1, group['line']]))
        self.assertEqual(group['actor'], ('person', [1, group['line']]))
        group = self.rendered('u35-t08')[1]
        self.assertEqual(group['person'], ('unknown', [group['line']]))
        self.assertEqual(group['actor'], ('person', [group['line']]))
        example = self.annotation(35, 'Who do you think wrote this book?')
        group = groups(book.render_annotation(example))[1]
        self.assertEqual(group['person'], ('unknown', [1, group['line']]))
        self.assertEqual(group['actor'], ('person', [1, group['line']]))

    def test_names_people_and_unclassified_objects_in_annotation_output(self):
        example = self.annotation(1, 'Mary bought a book. The book is on the table.')
        first, second = groups(book.render_annotation(example))
        self.assertEqual(first['actor'], ('person', [first['line']]))
        self.assertEqual(first['person'], ('Mary', [first['line']]))
        self.assertEqual(second['actor'], ('book', [second['line']]))
        self.assertEqual(second['where'], ('on the table', [second['line']]))
        self.assertNotIn('person', second)
        group = groups(book.render_annotation(self.annotation(60, 'The car was repaired by Mary.')))[0]
        self.assertEqual(group['actor'][0], 'person')
        self.assertEqual(group['person'][0], 'Mary')
        self.assertEqual(group['object'][0], 'car')

    def test_plural_candidates_have_explicit_types_and_keep_one_split_family(self):
        for value, kind in [('children', 'person'), ('men', 'person'), ('women', 'person'),
                            ('cities', 'location'), ('books', None), ('sheep', None)]:
            with self.subTest(value=value):
                example = self.annotation(11, f'The {value} are here.')
                self.assertEqual(example.family, 'u11_plural_agreement')
                roles = example.sentences[0]['predicates'][0]['roles']
                self.assertEqual(roles['actor'], kind or value)
                if kind:
                    self.assertEqual(roles[kind], value)
                self.assertEqual(roles['where'], 'here')

    def test_context_disambiguates_human_nationalities_and_newspaper_names(self):
        for identifier, expected in [
            ('u24-e4-example', 'Chinese'),  # Language, not a group of people.
            ('u04-37dd1c01a5bc6e20-1-v1-s1', 'person'),
            ('u06-8af354572697a536-1-v1-s1', 'Today'),  # Newspaper, not today's date.
            ('u40-e1-02c', 'time'),
            ('u01-2a04dd7e24d42104-1-v1-s1', 'they'),  # Pens, not people.
        ]:
            self.assertEqual(self.corpus.by_id[identifier]['facts'][0]['actor'], expected)

    def test_schema_rejects_missing_mismatched_and_obsolete_value_fields(self):
        record = deepcopy(self.corpus.by_id['u24-e11-b07b'])
        original = record['facts'][0]
        invalid = [
            {k: v for k, v in original.items() if k != 'person'},
            dict(original, actor='organization'),
            dict(original, organization='company'),
            dict(original, person='none'),
            dict(original, time='yesterday'),
            dict(original, location='in London'),
        ]
        with tempfile.TemporaryDirectory() as folder:
            for fact in invalid:
                with self.subTest(fact=fact):
                    with self.assertRaises(ValueError):
                        book.validate(record['text'], [fact])
                    record['facts'][0] = fact
                    data = dict(schema_version=1, completed_units=[record['unit']], sentences=[record])
                    (Path(folder)/'sentences.json').write_text(json.dumps(data), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        CorpusLibrary(folder)
        example = self.annotation(1, 'Mary bought a book. The book is on the table.')
        sentences = deepcopy(example.sentences)
        del sentences[0]['predicates'][0]['roles']['person']
        with self.assertRaises(ValueError):
            book.validate_annotation_example(replace(example, sentences=sentences))


if __name__ == '__main__':
    unittest.main()
