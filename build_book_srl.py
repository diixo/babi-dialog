#!/usr/bin/env python3
"""Synchronize book templates and build single-utterance fact-extraction examples.

Text and answers use the same bindings. There is no parser, random sampling,
dialogue generation or automatic semantic annotation in the builder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import re

from book_srl_annotations import read_json, substitute, split_for
from book_srl_corpus import CorpusLibrary
from book_srl_predicates import expand_predicates, validate_predicates

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / 'data' / 'book_srl'
DEFAULT_OUT = ROOT / 'products-book-srl'
CORE_FIELDS = ('actor', 'predicate', 'action', 'intention', 'operation', 'object')
OPERATIONS = {'none', 'add', 'remove', 'replace', 'keep', 'clear', 'complete'}


def validate(text, facts):
    if not isinstance(text, str) or not text.strip() or any(c in text for c in '\n\r\t{}'):
        raise ValueError('A template must produce one nonempty utterance')
    if not isinstance(facts, list) or not facts:
        raise ValueError(f'Missing facts for {text!r}')
    for fact in facts:
        if any(field not in fact for field in CORE_FIELDS):
            raise ValueError(f'Missing fact fields for {text!r}')
        for field, value in fact.items():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', field):
                raise ValueError(f'Invalid fact field: {field!r}')
            if not isinstance(value, str) or not value.strip() or any(c in value for c in '\t\r\n{}'):
                raise ValueError(f'Invalid fact value: {field}={value!r}')
        if fact['operation'] not in OPERATIONS:
            raise ValueError(f'Invalid operation: {fact["operation"]!r}')


def render(text, facts):
    validate(text, facts)
    rows = ['User: ' + text]
    for index, fact in enumerate(facts, 1):
        suffix = f' {index}' if len(facts) > 1 else ''
        fields = list(CORE_FIELDS) + [field for field in fact if field not in CORE_FIELDS]
        for field in fields:
            rows.append(f'{field.replace("_", " ")}{suffix}?\t{fact[field]}\t1')
    return '\n'.join(f'{i} {row}' for i, row in enumerate(rows, 1)) + '\n\n'


class Library:
    def __init__(self, directory=DEFAULT_LIBRARY, *, catalogue=None, corpus=None):
        if catalogue is None:
            catalogue = read_json(Path(directory) / 'fact_templates.json')
        vocabulary = read_json(Path(directory) / 'slot_values.json')
        if catalogue.get('schema_version') != 1 or vocabulary.get('schema_version') != 1:
            raise ValueError('Unsupported template library version')
        self.domains = vocabulary['domains']
        self.catalogue = catalogue
        self.coverage = catalogue['coverage']
        if corpus is None:
            corpus = CorpusLibrary(directory)
        if (self.coverage['sentences'] != len(corpus.records)
                or self.coverage['completed_units'] != corpus.completed_units
                or self.coverage['pending_units'] != [u for u in range(1,85) if u not in corpus.completed_units]):
            raise ValueError('Sentence catalogue changed; run build_book_srl.py to synchronize it')
        self.units = {lesson['unit']: lesson for lesson in catalogue['lessons']}
        if len(catalogue['lessons']) != 84 or set(self.units) != set(range(1,85)):
            raise ValueError('The catalogue must account for all 84 lessons')
        seen = set()
        for lesson in self.units.values():
            for template in lesson['templates']:
                if template['id'] in seen:
                    raise ValueError(f'Duplicate template: {template["id"]}')
                seen.add(template['id'])
                domains = self.template_domains(template)
                if set(template['original']) != set(domains):
                    raise ValueError(f'{template["id"]}: original bindings do not match slots')
                original = {name: next((row for row in rows if row['id']==template['original'].get(name)), None)
                            for name, rows in domains.items()}
                if any(value is None for value in original.values()):
                    raise ValueError(f'{template["id"]}: original binding is missing')
                for constraint in template.get('constraints', []):
                    names=constraint['slots']
                    if len(names)!=2 or any(name not in domains for name in names):
                        raise ValueError(f'{template["id"]}: invalid compatibility constraint')
                    choices=[{row['id'] for row in domains[name]} for name in names]
                    if not constraint['pairs'] or any(len(pair)!=2 or any(value not in choices[i]
                            for i,value in enumerate(pair)) for pair in constraint['pairs']):
                        raise ValueError(f'{template["id"]}: invalid compatible pair')
                if not self.compatible(template, original):
                    raise ValueError(f'{template["id"]}: original bindings are incompatible')
                text, spans, _ = expand_predicates(template['text'], original,
                    template.get('predicate_spans'), template['id'])
                facts = substitute(template['facts'], original)
                validate(text, facts)
                validate_predicates(text, facts, spans, template['id'])
                if text != template['source']['text']:
                    raise ValueError(f'{template["id"]}: original sentence cannot be reconstructed')
                if template['source']['unit'] != lesson['unit']:
                    raise ValueError(f'{template["id"]}: source lesson mismatch')
                source = corpus.by_id.get(template['source']['sentence'])
                if source is None or source['id'] != template['id'] or source['unit'] != lesson['unit']:
                    raise ValueError(f'{template["id"]}: unknown source sentence')
                if text != source['text'] or facts != source['facts']:
                    raise ValueError(f'{template["id"]}: template differs from the English sentence or its manual facts')
                if spans != source['predicate_spans']:
                    raise ValueError(f'{template["id"]}: template differs from the manual predicate spans')
        if seen != set(corpus.by_id):
            raise ValueError('Every English sentence must have a template; run build_book_srl.py to synchronize it')

    def template_domains(self, template):
        domains = {}
        for name, identifier in template['slots'].items():
            if identifier not in self.domains:
                raise ValueError(f'{template["id"]}: unknown slot domain {identifier!r}')
            rows = self.domains[identifier]['values']
            if not rows or len({r['id'] for r in rows}) != len(rows):
                raise ValueError(f'{template["id"]}: empty or duplicate slot choices')
            domains[name] = rows
        return domains

    @staticmethod
    def compatible(template, bindings):
        return all([bindings[name]['id'] for name in constraint['slots']] in constraint['pairs']
                   for constraint in template.get('constraints', []))

    def generate(self, unit):
        for template in self.units[unit]['templates']:
            domains = self.template_domains(template)
            names = list(domains)
            for values in product(*(domains[name] for name in names)):
                bindings = dict(zip(names, values))
                if not self.compatible(template, bindings):
                    continue
                text, spans, _ = expand_predicates(template['text'], bindings,
                    template['predicate_spans'], template['id'])
                facts = substitute(template['facts'], bindings)
                validate_predicates(text, facts, spans, template['id'])
                yield template['family'], text, facts


def synchronize_templates(directory=DEFAULT_LIBRARY):
    """Prepare and validate a synchronized library in memory, without writing.

    Existing variable templates remain authoritative for their substitutions.
    New annotated sentences receive fixed templates; removed sentences drop out.
    Manual facts are checked against templates rather than silently overwritten.
    """
    directory = Path(directory)
    corpus = CorpusLibrary(directory)
    path = directory / 'fact_templates.json'
    old = read_json(path) if path.exists() else {'lessons': []}
    templates = {}
    for lesson in old['lessons']:
        for template in lesson['templates']:
            if template['id'] in templates:
                raise ValueError(f'Duplicate template: {template["id"]}')
            templates[template['id']] = template
    lessons = []
    for unit, records in corpus.by_unit.items():
        prepared = []
        for record in records:
            template = deepcopy(templates.get(record['id']))
            if template is None:
                template = dict(id=record['id'], family=record['id'], text=record['text'],
                    slots={}, original={}, constraints=[], facts=deepcopy(record['facts']),
                    predicate_spans=deepcopy(record['predicate_spans']))
            template['source'] = dict(sentence=record['id'], unit=unit, text=record['text'])
            prepared.append(template)
        lessons.append(dict(unit=unit, templates=prepared))
    catalogue = dict(schema_version=1, coverage=dict(sentences=len(corpus.records),
        completed_units=corpus.completed_units,
        pending_units=[u for u in range(1,85) if u not in corpus.completed_units]), lessons=lessons)
    return Library(directory, catalogue=catalogue, corpus=corpus)


def build(library, out=None):
    units = list(range(1, 85))
    for unit in units:
        if unit not in library.units or not library.units[unit]['templates']:
            raise ValueError(f'Unit {unit} has no prepared templates')
        if unit not in library.coverage['completed_units']:
            raise ValueError(f'Unit {unit} is still incomplete')
    examples, parents = {}, {}

    def root(family):
        parents.setdefault(family, family)
        while parents[family] != family:
            parents[family] = parents[parents[family]]
            family = parents[family]
        return family

    for unit in units:
        for family, text, facts in library.generate(unit):
            story = render(text, facts)
            key = ' '.join(text.casefold().split())
            root(family)
            if key in examples:
                previous_family, previous_story, _ = examples[key]
                if previous_story != story:
                    raise ValueError(f'Conflicting facts for {text!r}: {previous_family}, {family}')
                first, second = sorted((root(previous_family), root(family)))
                parents[second] = first
            else:
                examples[key] = (family, story, sum(len(f) for f in facts))
    splits = {name: [] for name in ('train','valid','test')}
    qa = {name: 0 for name in splits}
    for family, story, count in examples.values():
        name = split_for(root(family))
        splits[name].append(story)
        qa[name] += count
    # Everything is checked before touching existing outputs.
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, stories in splits.items():
            (out / f'{name}.txt').write_text(''.join(stories), encoding='utf-8', newline='\n')
    return {'units': units, 'templates': sum(len(library.units[u]['templates']) for u in units),
            'examples': len(examples), 'splits': {name: {'stories': len(stories), 'qa': qa[name]}
                                               for name, stories in splits.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        library = synchronize_templates(args.library)
        for unit in sorted(library.units):
            state = 'ready' if unit in library.coverage['completed_units'] else 'not completed'
            print(f'{unit:02d}: {len(library.units[unit]["templates"])} sentence templates; {state}')
        # build checks every lesson and variant before writing the dataset.
        stats = build(library, args.out)
        path = args.library / 'fact_templates.json'
        content = (json.dumps(library.catalogue, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
        for name, count in stats['splits'].items():
            print(f'{args.out / (name+".txt")}: {count["stories"]} stories, {count["qa"]} QA')
        print(f'{len(stats["units"])}/84 book units; '
              f'{stats["templates"]} source sentence templates; {stats["examples"]} unique generated sentences')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
