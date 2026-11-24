"""Read the English sentences and their facts from JSON; no book parser."""
import json
from pathlib import Path
import re

from book_srl_predicates import SPAN_FIELDS, validate_fact_spans

CORPUS_NAME = 'sentences.json'
UTTERANCE_TYPES = {'question', 'non_question'}
ACTOR_TYPES = {'person', 'time', 'location', 'organization'}


def actor_value(fact):
    """Return the participant, following its explicitly annotated type if present."""
    actor = fact['actor']
    return fact[actor] if actor in ACTOR_TYPES else actor


def validate_actor(fact):
    actor = fact['actor']
    expected = {actor} if actor in ACTOR_TYPES else set()
    if ACTOR_TYPES & fact.keys() != expected:
        raise ValueError('Actor type must match its value field; use when/where for circumstances')
    if expected and (not isinstance(fact[actor], str) or not fact[actor].strip()
                     or fact[actor] == 'none'):
        raise ValueError(f'Missing {actor} participant value')


def read_json(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'{path}: duplicate JSON key {key!r}')
            result[key] = value
        return result
    try:
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read {path}: {error}') from error
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError(f'{path}: expected schema_version 1')
    return data


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
            required = {'id', 'unit', 'text', 'facts', *SPAN_FIELDS}
            if (not isinstance(record, dict) or not required <= record.keys()
                    or record.keys() - required - {'embedded_roles'}):
                raise ValueError('Each sentence needs id, unit, text, facts and predicate/fragment/verb spans')
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
                validate_actor(fact)
                if fact.get('type') not in UTTERANCE_TYPES:
                    raise ValueError(f'{identifier}: each fact needs type question or non_question')
                if fact.get('clause_type') in ('ellipsis', 'vocative'):
                    raise ValueError(f'{identifier}: fragments do not belong in sentences.json')
            self.by_id[identifier] = record
            validate_fact_spans(text, record['facts'], record, identifier)
            self.by_unit[unit].append(record)
        if any(not self.by_unit[unit] for unit in self.completed_units):
            raise ValueError('A completed lesson must contain sentences')
        # External examples live in the same corpus, without pretending to be
        # lessons from the book. A complete source row is one training sample.
        self.additional_records = data.get('additional', [])
        if not isinstance(self.additional_records, list):
            raise ValueError('additional must be a list of reviewed samples')
        self.additional_by_id = {}
        for record in self.additional_records:
            required = {'id', 'text', 'sentences', 'source', 'facts', *SPAN_FIELDS}
            optional = {'embedded_roles', 'source_text', 'alternatives'}
            if (not isinstance(record, dict) or not required <= record.keys()
                    or record.keys() - required - optional):
                raise ValueError('Each additional sample needs text, sentences, source, facts and spans')
            identifier = record['id']
            if (not isinstance(identifier, str) or not identifier
                    or identifier in self.by_id or identifier in self.additional_by_id):
                raise ValueError(f'Duplicate or invalid additional sample ID: {identifier!r}')
            text, sentences = record['text'], record['sentences']
            if (not isinstance(text, str) or not text.strip()
                    or re.search(r'[\u0400-\u04ff\r\n\t{}]', text)
                    or not isinstance(sentences, list) or not sentences
                    or any(not isinstance(s, str) or not re.search('[A-Za-z]', s) for s in sentences)
                    or ' '.join(sentences) != text):
                raise ValueError(f'{identifier}: sentences must reconstruct one English source row')
            source = record.get('source', {})
            if (not isinstance(source, dict) or set(source) != {'file', 'line'}
                    or not isinstance(source.get('file'), str) or not source['file']
                    or type(source.get('line')) is not int or source['line'] < 1):
                raise ValueError(f'{identifier}: missing source file/line metadata')
            if not isinstance(record.get('facts'), list) or not record['facts']:
                raise ValueError(f'{identifier}: missing manual facts')
            for fact in record['facts']:
                if (not isinstance(fact, dict)
                        or not {'actor', 'predicate', 'action', 'intention', 'operation', 'object'} <= fact.keys()
                        or any(not isinstance(v, str) or not v.strip()
                               or re.search(r'[\u0400-\u04ff]', v) for v in fact.values())):
                    raise ValueError(f'{identifier}: incomplete manual facts')
                validate_actor(fact)
                if fact.get('type') not in UTTERANCE_TYPES:
                    raise ValueError(f'{identifier}: missing utterance type')
            validate_fact_spans(text, record['facts'], record, identifier)
            cursor = 0
            for sentence in sentences:
                end = cursor + len(sentence)
                if not any(re.search('[A-Za-z]', text[max(cursor,a):min(end,b)])
                           for selected in record['fragment_spans'] for a,b in selected
                           if max(cursor,a) < min(end,b)):
                    raise ValueError(f'{identifier}: a sentence has no annotated fragment')
                cursor = end + 1
            self.additional_by_id[identifier] = record
