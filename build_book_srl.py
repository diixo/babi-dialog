#!/usr/bin/env python3
"""Synchronize book templates and build single-utterance fact-extraction examples.

Text and answers use the same bindings. There is no parser, random sampling,
dialogue generation or automatic semantic annotation in the builder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import re
from typing import Callable, Iterator

from book_srl_corpus import CorpusLibrary, read_json
from book_srl_predicates import (SPAN_FIELDS, expand_predicates, predicate_from_spans,
                                 validate_fact_spans, substitute)

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / 'data' / 'book_srl'
DEFAULT_OUT = ROOT / 'products-book-srl'
CORE_FIELDS = ('actor', 'predicate', 'action', 'intention', 'operation', 'object')
OPERATIONS = {'none', 'add', 'remove', 'replace', 'keep', 'clear', 'complete'}
AUXILIARY_ACTIONS = {
    'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
    'be': 'be', 'been': 'be', 'being': 'be', "'m": 'be', "'re": 'be',
    "'s": 'be', 'do': 'do', 'does': 'do', 'did': 'do', 'done': 'do',
    'have': 'have', 'has': 'have', 'had': 'have', "'ve": 'have',
    'can': 'unknown', 'ca': 'unknown', 'could': 'unknown',
    'will': 'unknown', 'wo': 'unknown', "'ll": 'unknown',
    'would': 'unknown', "'d": 'unknown', 'shall': 'unknown',
    'should': 'unknown', 'must': 'unknown', 'may': 'unknown', 'might': 'unknown',
}
CONTEXT_FIELDS = {'intention', 'operation', 'aspect', 'voice', 'polarity', 'status',
                  'verb_form', 'form', 'participle_form', 'form_voice', 'degree',
                  'syntactic_function', 'modality', 'modal_meaning', 'negation_scope'}


def validate(text, facts):
    if not isinstance(text, str) or not text.strip() or any(c in text for c in '\n\r\t{}'):
        raise ValueError('A template must produce one nonempty utterance')
    if not isinstance(facts, list) or not facts:
        raise ValueError(f'Missing facts for {text!r}')
    for fact in facts:
        if any(field not in fact for field in CORE_FIELDS):
            raise ValueError(f'Missing fact fields for {text!r}')
        if {'fragment', 'verb'} & fact.keys():
            raise ValueError('fragment and verb are derived from source spans, not fact fields')
        for field, value in fact.items():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', field):
                raise ValueError(f'Invalid fact field: {field!r}')
            if not isinstance(value, str) or not value.strip() or any(c in value for c in '\t\r\n{}'):
                raise ValueError(f'Invalid fact value: {field}={value!r}')
        if fact['operation'] not in OPERATIONS:
            raise ValueError(f'Invalid operation: {fact["operation"]!r}')


def render(text, facts, annotations):
    validate(text, facts)
    validate_fact_spans(text, facts, annotations)
    rows = ['User: ' + text]

    def contained(spans, start, end):
        return all(start <= a < b <= end for a, b in spans)

    def present(value, fragment):
        return bool(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', fragment, re.I))

    for index, fact in enumerate(facts):
        start, end = annotations['fragment_spans'][index][0]
        fragment = text[start:end]
        rows.append(f'fragment?\t{fragment}\t1')
        fragment_line = len(rows)
        verb = predicate_from_spans(text, annotations['verb_spans'][index])
        fields = ['actor', 'predicate', 'verb', 'action', 'intention', 'operation', 'object']
        fields += [field for field in fact if field not in fields and field != 'tense']
        for field in fields:
            value = verb if field == 'verb' else fact[field]
            if value == 'none' and field not in ('predicate', 'verb'):
                continue
            # References outside the selected fragment stay attached to the
            # complete input. Labels alone cannot establish local evidence.
            local = (not text[:start].strip(' \t.,;:!?\"')
                     and not text[end:].strip(' \t.,;:!?\"'))
            if value == 'unknown':
                local = True
            elif field == 'predicate':
                local = local or contained(annotations['predicate_spans'][index], start, end)
            elif field == 'verb':
                local = local or contained(annotations['verb_spans'][index], start, end)
            elif field == 'action':
                auxiliary = verb.casefold().replace('’', "'")
                if auxiliary.endswith("n't"):
                    auxiliary = auxiliary[:-3]
                auxiliary_action = AUXILIARY_ACTIONS.get(auxiliary)
                # Elliptical auxiliaries/to can stand for an action expressed
                # elsewhere. Its normalized label needs that complete context.
                local = local or (bool(annotations['verb_spans'][index])
                    and contained(annotations['predicate_spans'][index], start, end)
                    and (auxiliary_action is None or auxiliary_action == value))
            elif field not in CONTEXT_FIELDS:
                surface = {'user': 'I', 'addressee': 'you'}.get(value, value) if field == 'actor' else value
                local = local or present(surface, fragment)
            refs = str(fragment_line) if local else f'1 {fragment_line}'
            rows.append(f'{field.replace("_", " ")}?\t{value}\t{refs}')
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
                text, annotations = self.expand_spans(template, original)
                facts = substitute(template['facts'], original)
                validate(text, facts)
                validate_fact_spans(text, facts, annotations, template['id'])
                if text != template['source']['text']:
                    raise ValueError(f'{template["id"]}: original sentence cannot be reconstructed')
                if template['source']['unit'] != lesson['unit']:
                    raise ValueError(f'{template["id"]}: source lesson mismatch')
                source = corpus.by_id.get(template['source']['sentence'])
                if source is None or source['id'] != template['id'] or source['unit'] != lesson['unit']:
                    raise ValueError(f'{template["id"]}: unknown source sentence')
                if text != source['text'] or facts != source['facts']:
                    raise ValueError(f'{template["id"]}: template differs from the English sentence or its manual facts')
                for field in SPAN_FIELDS:
                    if annotations[field] != source[field]:
                        raise ValueError(f'{template["id"]}: template differs from the manual {field}')
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

    @staticmethod
    def expand_spans(template, bindings):
        annotations = {}
        for field in SPAN_FIELDS:
            text, spans, _ = expand_predicates(template['text'], bindings,
                                               template.get(field), f'{template["id"]}: {field}')
            annotations[field] = spans
        return text, annotations

    def generate(self, unit):
        for template in self.units[unit]['templates']:
            domains = self.template_domains(template)
            names = list(domains)
            for values in product(*(domains[name] for name in names)):
                bindings = dict(zip(names, values))
                if not self.compatible(template, bindings):
                    continue
                text, annotations = self.expand_spans(template, bindings)
                facts = substitute(template['facts'], bindings)
                validate_fact_spans(text, facts, annotations, template['id'])
                yield template['family'], text, facts, annotations


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
                    **{field: deepcopy(record[field]) for field in SPAN_FIELDS})
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
        for family, text, facts, annotations in library.generate(unit):
            story = render(text, facts, annotations)
            key = ' '.join(text.casefold().split())
            root(family)
            if key in examples:
                previous_family, previous_story, _, previous_facts = examples[key]
                if previous_story != story or previous_facts != facts:
                    raise ValueError(f'Conflicting facts for {text!r}: {previous_family}, {family}')
                first, second = sorted((root(previous_family), root(family)))
                parents[second] = first
            else:
                examples[key] = (family, story, validate_story(story), facts)
    splits = {name: [] for name in ('train','valid','test')}
    qa = {name: 0 for name in splits}
    for family, story, count, _ in examples.values():
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


ANNOTATION_FIELDS = ('tense', 'aspect', 'voice', 'polarity', 'form', 'status')


ANNOTATION_VALUES = {
    'tense': {'present', 'past', 'future', 'nonfinite'},
    'aspect': {'simple', 'continuous', 'perfect', 'perfect_continuous'},
    'voice': {'active', 'passive'},
    'polarity': {'positive', 'negative'},
    'form': {'finite', 'infinitive', 'gerund', 'participle'},
    'status': {'asserted', 'habitual', 'intended', 'questioned', 'hypothetical',
               'reported', 'inferred', 'required', 'possible', 'counterfactual'},
}


@dataclass(frozen=True)
class AnnotationExample:
    unit: int
    source_unit: int
    template_id: str
    rule: str
    binding_id: str
    sentences: tuple[dict, ...]

    @property
    def text(self) -> str:
        return ' '.join(s['text'] for s in self.sentences)

    @property
    def family(self) -> str:
        # Reviews and multi-sentence variants keep their originating family.
        return self.template_id


def validate_annotation_example(example: AnnotationExample) -> None:
    if not example.sentences:
        raise ValueError(f'{example.template_id}: no sentences')
    for number, sentence in enumerate(example.sentences, 1):
        text = sentence.get('text', '')
        if not isinstance(text, str) or not text or any(c in text for c in '\t\n\r{}'):
            raise ValueError(f'{example.template_id}: invalid sentence {text!r}')
        if text[-1] not in '.?!':
            raise ValueError(f'{example.template_id}: sentence has no final punctuation')
        predicates = sentence.get('predicates', [])
        if not predicates and sentence.get('features',{}).get('clause_type') not in ('ellipsis','vocative'):
            raise ValueError(f'{example.template_id}: sentence {number} has no predicates')
        for index, event in enumerate(predicates, 1):
            predicate = event.get('text', '')
            boundary = r'(?<!\w)' if isinstance(predicate, str) and predicate[:1].isalnum() else ''
            if not isinstance(predicate, str) or not predicate or not re.search(boundary+re.escape(predicate)+r'(?!\w)', text, re.I):
                raise ValueError(f'{example.template_id}: predicate {predicate!r} absent in {text!r}')
            roles = event.get('roles', {})
            for role in ('subject', 'actor', 'action', 'object'):
                if not isinstance(roles.get(role), str) or not roles[role].strip():
                    raise ValueError(f'{example.template_id}: missing role {role}')
            for field in ANNOTATION_FIELDS:
                if event.get(field) not in ANNOTATION_VALUES[field]:
                    raise ValueError(f'{example.template_id}: invalid {field}: {event.get(field)!r}')
            for ref in event.get('context', []):
                if type(ref) is not int or not 1 <= ref < number:
                    raise ValueError(f'{example.template_id}: context must reference an earlier sentence')
            parent = event.get('parent')
            if parent is not None and (type(parent) is not int or not 1 <= parent < index):
                raise ValueError(f'{example.template_id}: parent must reference an earlier predicate')
            if (parent is None) != ('relation' not in event):
                raise ValueError(f'{example.template_id}: parent and relation must occur together')
            if parent is not None and not re.fullmatch(r'[a-z][a-z0-9_]*', event['relation']):
                raise ValueError(f'{example.template_id}: invalid predicate relation')
        for group in [sentence.get('features', {}), *(e['roles'] for e in predicates)]:
            for label, answer in group.items():
                if not isinstance(answer, str) or not answer or any(c in answer for c in '\t\n\r{}'):
                    raise ValueError(f'{example.template_id}: invalid annotation {label!r}: {answer!r}')
                if not re.fullmatch(r'[a-z][a-z0-9_]*', label):
                    raise ValueError(f'{example.template_id}: invalid label {label!r}')


class AnnotationLibrary:
    def __init__(self, directory: Path = DEFAULT_LIBRARY):
        self.directory = Path(directory)
        self.anchors = {f'unit-{unit}' for unit in range(1,85)}
        nouns = read_json(self.directory / 'nouns.json')
        verbs = read_json(self.directory / 'verbs.json')
        catalogue = read_json(self.directory / 'templates.json')
        self.lexicons = {'noun': nouns['nouns'], 'person': nouns['people'], 'verb': verbs['verbs']}
        for kind, entries in self.lexicons.items():
            if not isinstance(entries, dict) or not entries:
                raise ValueError(f'Empty lexicon: {kind}')
            for identifier, entry in entries.items():
                if not isinstance(entry, dict):
                    raise ValueError(f'Invalid entry: {kind}:{identifier}')
                entry['id'] = identifier
                if kind == 'noun':
                    required = ('singular', 'plural', 'article', 'countable')
                    if any(k not in entry for k in required):
                        raise ValueError(f'Incomplete noun: {identifier}')
                    if type(entry['countable']) is not bool or not entry['singular']:
                        raise ValueError(f'Invalid noun: {identifier}')
                    if entry['countable'] and not entry['plural']:
                        raise ValueError(f'Countable noun needs a plural: {identifier}')
                    if entry['countable'] and entry['article'] not in ('a', 'an'):
                        raise ValueError(f'Countable noun needs a/an: {identifier}')
                    if not entry['countable'] and (entry['article'] or entry['plural']):
                        raise ValueError(f'Uncountable noun has count forms: {identifier}')
                    entry['indefinite'] = (entry['article']+' '+entry['singular']).strip()
                    entry['definite'] = 'the '+entry['singular']
                elif kind == 'verb':
                    if any(not entry.get(k) for k in ('lemma', 'present3', 'past', 'participle', 'gerund')):
                        raise ValueError(f'Incomplete verb: {identifier}')
                    if any(obj not in self.lexicons['noun'] for obj in entry.get('objects', [])):
                        raise ValueError(f'Unknown compatible noun for {identifier}')
        self.units = {}
        self.templates = {}
        for lesson in catalogue.get('lessons', []):
            unit = lesson['unit']
            if type(unit) is not int or unit in self.units or not 1 <= unit <= 84:
                raise ValueError(f'Duplicate or unknown lesson: {unit}')
            if not isinstance(lesson.get('title'), str) or not lesson['title'].strip():
                raise ValueError(f'Missing JSON title for unit {unit}')
            self.units[unit] = lesson
            for template in lesson.get('templates', []):
                tid = template['id']
                if tid in self.templates:
                    raise ValueError(f'Duplicate template: {tid}')
                if not template.get('rule') or template.get('source') not in self.anchors:
                    raise ValueError(f'{tid}: rule and valid source anchor are required')
                if template['source'] != f'unit-{unit}':
                    raise ValueError(f'{tid}: source must match its unit')
                self.templates[tid] = (unit, template)
        if set(self.units) != set(range(1, 85)):
            raise ValueError('The catalogue must define a generator for every unit, 1–84')
        for unit in self.units:
            self.templates_for(unit)  # Reject empty/cyclic/unknown review references now.

    def templates_for(self, unit: int, ancestors: tuple[int, ...] = ()) -> list[tuple[int, dict]]:
        if unit not in self.units or unit in ancestors:
            raise ValueError(f'Unknown or cyclic lesson reference: {ancestors + (unit,)}')
        lesson = self.units[unit]
        result = [(unit, t) for t in lesson.get('templates', [])]
        for previous in lesson.get('review_units', []):
            if previous >= unit:
                raise ValueError(f'Unit {unit} may only review earlier units')
            result.extend(self.templates_for(previous, ancestors + (unit,)))
        result = list({t['id']: (origin, t) for origin, t in result}.values())
        if not result:
            raise ValueError(f'Unit {unit} has no templates')
        return result

    def bindings_for(self, template: dict) -> Iterator[tuple[str, dict]]:
        slots = template.get('slots', {})
        names = sorted(slots)
        domains = []
        for name in names:
            entries = slots[name]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f'{template["id"]}: empty slot {name}')
            values = []
            for entry in entries:
                if isinstance(entry, str) and ':' in entry:
                    kind, identifier = entry.split(':', 1)
                    if kind not in self.lexicons or identifier not in self.lexicons[kind]:
                        raise ValueError(f'{template["id"]}: unknown lexeme {entry}')
                    value = self.lexicons[kind][identifier]
                elif isinstance(entry, dict) and isinstance(entry.get('id'), str):
                    value = entry
                else:
                    raise ValueError(f'{template["id"]}: invalid binding {entry!r}')
                values.append(value)
            if len({v['id'] for v in values}) != len(values):
                raise ValueError(f'{template["id"]}: duplicate values in slot {name}')
            domains.append(values)
        count = 0
        for combination in product(*domains):
            bound = dict(zip(names, combination))
            valid = True
            for constraint in template.get('compatible', []):
                verb_slot, noun_slot = constraint
                if verb_slot not in bound or noun_slot not in bound:
                    raise ValueError(f'{template["id"]}: constraint references unknown slots')
                if bound[noun_slot]['id'] not in bound[verb_slot].get('objects', []):
                    valid = False
            if not valid:
                continue
            count += 1
            yield ','.join(f'{name}={bound[name]["id"]}' for name in names) or 'fixed', bound
        if not count:
            raise ValueError(f'{template["id"]}: no compatible combinations')

    def generate(self, unit: int) -> Iterator[AnnotationExample]:
        for origin, template in self.templates_for(unit):
            if template.get('combine', False) and len(template['sentences']) != 1:
                raise ValueError(f'{template["id"]}: combine requires single-sentence templates')
            instances = []
            for key, bindings in self.bindings_for(template):
                sentences = tuple(substitute(template['sentences'], bindings))
                example = AnnotationExample(unit, origin, template['id'], template['rule'], key, sentences)
                validate_annotation_example(example)
                instances.append(example)
                yield example
            # Pair independent, explicitly named actors from the SAME template.
            # Native multi-sentence templates keep their manually authored context.
            if template.get('combine', False):
                for left in instances:
                    for right in instances:
                        if left.binding_id >= right.binding_id:
                            continue
                        a = left.sentences[0]['predicates'][0]['roles']['actor']
                        b = right.sentences[0]['predicates'][0]['roles']['actor']
                        if a == b or a in ('unknown', 'user') or b in ('unknown', 'user'):
                            continue
                        combined = AnnotationExample(unit, origin, template['id'], template['rule'],
                                           left.binding_id+'+'+right.binding_id,
                                           left.sentences + right.sentences)
                        validate_annotation_example(combined)
                        yield combined

    def generators(self) -> dict[int, Callable[[], Iterator[AnnotationExample]]]:
        def make_generator(unit: int) -> Callable[[], Iterator[AnnotationExample]]:
            def generate() -> Iterator[AnnotationExample]:
                yield from self.generate(unit)
            generate.__name__ = f'generate_unit_{unit:02d}'
            rules = [t['rule'] for _, t in self.templates_for(unit)]
            generate.__doc__ = self.units[unit]['title']+'\n'+'\n'.join(rules)
            return generate
        return {unit: make_generator(unit) for unit in sorted(self.units)}


def render_annotation(example: AnnotationExample) -> str:
    validate_annotation_example(example)
    rows = ['User: '+example.text]
    def qa(label, answer, refs):
        refs = sorted(set(refs))
        if not refs or not all(1 <= n <= len(rows) for n in refs):
            raise ValueError('Invalid annotation references')
        rows.append(f'{label}\t{answer}\t'+ ' '.join(map(str, refs)))
        return len(rows)
    sentence_lines = []
    predicate_lines = []
    for si, sentence in enumerate(example.sentences, 1):
        line = qa(f'sentence {si}?', sentence['text'], [1])
        sentence_lines.append(line)
        for feature, value in sentence.get('features', {}).items():
            qa(f'{feature} of sentence {si}?', value, [line])
        local_predicates = []
        for pi, event in enumerate(sentence['predicates'], 1):
            scope = f'predicate {pi} of sentence {si}'
            pred = qa(scope+'?', event['text'], [line])
            predicate_lines.append(pred)
            refs = [line, pred] + [sentence_lines[n-1] for n in event.get('context', [])]
            if 'parent' in event:
                refs.append(local_predicates[event['parent']-1])
                qa(f'parent of {scope}?', f'predicate {event["parent"]} of sentence {si}', refs)
                qa(f'relation of {scope}?', event['relation'], refs)
            for role, value in event['roles'].items():
                qa(f'{role} of {scope}?', value, refs)
            for field in ANNOTATION_FIELDS:
                qa(f'{field} of {scope}?', event[field], refs)
            local_predicates.append(pred)
    qa('sentence count?', len(example.sentences), sentence_lines)
    qa('predicate count?', len(predicate_lines), predicate_lines or sentence_lines)
    result = '\n'.join(f'{n} {line}' for n, line in enumerate(rows, 1))+'\n\n'
    validate_story(result)
    return result


def validate_story(story: str) -> int:
    rows = story.strip().splitlines()
    if not rows or not rows[0].startswith('1 User: '):
        raise ValueError('Story must start with a user utterance')
    for number, row in enumerate(rows, 1):
        if not row.startswith(f'{number} '):
            raise ValueError(f'Invalid row number: {row}')
        if number == 1:
            if '\t' in row:
                raise ValueError('User utterance contains a tab')
            continue
        pieces = row.partition(' ')[2].split('\t')
        if len(pieces) != 3 or not pieces[0].endswith('?') or not pieces[1] or not pieces[2]:
            raise ValueError(f'Invalid QA row: {row}')
        if not all(r.isdigit() and 1 <= int(r) < number for r in pieces[2].split()):
            raise ValueError(f'Invalid evidence references: {row}')
    return len(rows)-1


def split_for(family: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(family.encode('utf-8')).digest()[:8], 'big') % 100
    return 'valid' if bucket < 10 else 'test' if bucket < 20 else 'train'


def build_annotations(library, out=None):
    """Expand the manually annotated synthetic templates for all 84 lessons."""
    if set(library.units) != set(range(1, 85)):
        raise ValueError('Annotation templates must cover all 84 lessons')
    examples, parents = {}, {}
    by_unit = {}

    def root(family):
        parents.setdefault(family, family)
        while parents[family] != family:
            parents[family] = parents[parents[family]]
            family = parents[family]
        return family

    for unit in range(1, 85):
        by_unit[unit] = set()
        for example in library.generate(unit):
            story = render_annotation(example)
            key = ' '.join(example.text.casefold().split())
            root(example.family)
            if key in examples:
                previous, old_story = examples[key]
                if old_story != story:
                    raise ValueError(f'Conflicting annotations for {example.text!r}: '
                                     f'{previous.template_id}, {example.template_id}')
                first, second = sorted((root(previous.family), root(example.family)))
                parents[second] = first
            else:
                examples[key] = (example, story)
            by_unit[unit].add(story)
        if not by_unit[unit]:
            raise ValueError(f'Unit {unit} has no annotation examples')
    datasets = {name: [] for name in ('train', 'valid', 'test')}
    for example, story in examples.values():
        datasets[split_for(root(example.family))].append(story)
    stats = {name: {'stories': len(stories), 'qa': sum(validate_story(s) for s in stories)}
             for name, stories in datasets.items()}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, stories in datasets.items():
            (out / f'{name}.txt').write_text(''.join(stories), encoding='utf-8', newline='\n')
    return {'splits': stats, 'units': {u: len(stories) for u, stories in by_unit.items()},
            'sentences': sum(len(e.sentences) for e, _ in examples.values()),
            'unique_examples': len(examples), 'families': len({root(e.family) for e, _ in examples.values()})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--annotate', action='store_true',
                        help='Also build the annotated synthetic templates in OUT/annotations')
    args = parser.parse_args(argv)
    try:
        library = synchronize_templates(args.library)
        annotation_library = AnnotationLibrary(args.library) if args.annotate else None
        for unit in sorted(library.units):
            state = 'ready' if unit in library.coverage['completed_units'] else 'not completed'
            annotation_coverage = (f'; {len(annotation_library.templates_for(unit))} annotation templates'
                                   if annotation_library is not None else '')
            print(f'{unit:02d}: {len(library.units[unit]["templates"])} sentence templates; {state}'
                  + annotation_coverage)
        # Check the optional dataset too before either output can be overwritten.
        if annotation_library is not None:
            build_annotations(annotation_library)
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
        if annotation_library is not None:
            annotation_out = args.out / 'annotations'
            annotation_stats = build_annotations(annotation_library, annotation_out)
            for name, count in annotation_stats['splits'].items():
                print(f'{annotation_out / (name+".txt")}: {count["stories"]} stories, {count["qa"]} QA')
            print(f'{len(annotation_stats["units"])}/84 annotation units; '
                  f'{annotation_stats["families"]} template families; '
                  f'{annotation_stats["unique_examples"]} unique examples')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
