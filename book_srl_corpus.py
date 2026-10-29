"""Read the English sentences and their facts from JSON; no book parser."""
from pathlib import Path
import re

from book_srl_annotations import read_json
from book_srl_predicates import validate_predicates

CORPUS_NAME = 'sentences.json'


class CorpusLibrary:
    def __init__(self, directory):
        self.directory = Path(directory)
        data = read_json(self.directory / CORPUS_NAME)
        self.completed_units = data['completed_units']
        if (not isinstance(self.completed_units, list)
                or any(type(u) is not int or not 1 <= u <= 84 for u in self.completed_units)
                or len(set(self.completed_units)) != len(self.completed_units)):
            raise ValueError('Invalid completed_units in sentences.json')
        self.records = data['sentences']
        if not isinstance(self.records, list) or not self.records:
            raise ValueError('sentences.json must contain English sentences')
        self.by_id = {}
        self.by_unit = {unit: [] for unit in range(1, 85)}
        for record in self.records:
            if not isinstance(record, dict) or set(record) != {'id', 'unit', 'text', 'facts', 'predicate_spans'}:
                raise ValueError('Each sentence needs id, unit, text, facts and predicate_spans')
            identifier, unit, text = record['id'], record['unit'], record['text']
            if not isinstance(identifier, str) or not identifier or identifier in self.by_id:
                raise ValueError(f'Duplicate or invalid sentence ID: {identifier!r}')
            if type(unit) is not int or unit not in self.by_unit:
                raise ValueError(f'{identifier}: invalid lesson number')
            if (not isinstance(text, str) or not text.strip() or not re.search('[A-Za-z]', text)
                    or re.search(r'[\u0400-\u04ff]|[\r\n\t{}<>]|\.{4,}|_{2,}', text)):
                raise ValueError(f'{identifier}: expected a complete English sentence')
            if not isinstance(record['facts'], list) or not record['facts']:
                raise ValueError(f'{identifier}: missing manual facts')
            for fact in record['facts']:
                if not isinstance(fact, dict) or not {'actor','action','intention','operation','object'} <= fact.keys():
                    raise ValueError(f'{identifier}: incomplete manual facts')
                if any(not isinstance(v, str) or not v.strip() or re.search(r'[\u0400-\u04ff]', v)
                       for v in fact.values()):
                    raise ValueError(f'{identifier}: facts must use English labels')
                if fact.get('clause_type') in ('ellipsis', 'vocative'):
                    raise ValueError(f'{identifier}: fragments do not belong in sentences.json')
            self.by_id[identifier] = record
            validate_predicates(text, record['facts'], record['predicate_spans'], identifier)
            self.by_unit[unit].append(record)
        if any(not self.by_unit[unit] for unit in self.completed_units):
            raise ValueError('A completed lesson must contain sentences')
