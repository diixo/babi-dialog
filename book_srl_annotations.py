#!/usr/bin/env python3
"""Shared JSON utilities and the legacy synthetic SRL library.

Each of the 84 synthetic units has its own callable generator. All inputs are
JSON; English book sentences are handled by build_book_srl.py. Standard library only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import re
from string import Formatter
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / 'data' / 'book_srl'
DEFAULT_OUT = ROOT / 'products-book-srl' / 'annotations'
FIELDS = ('tense', 'aspect', 'voice', 'polarity', 'form', 'status')
VALUES = {
    'tense': {'present', 'past', 'future', 'nonfinite'},
    'aspect': {'simple', 'continuous', 'perfect', 'perfect_continuous'},
    'voice': {'active', 'passive'},
    'polarity': {'positive', 'negative'},
    'form': {'finite', 'infinitive', 'gerund', 'participle'},
    'status': {'asserted', 'habitual', 'intended', 'questioned', 'hypothetical',
               'reported', 'inferred', 'required', 'possible', 'counterfactual'},
}


@dataclass(frozen=True)
class Example:
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


def substitute(value, bindings: dict):
    """Resolve dictionary paths only; no eval, attribute access or format tricks."""
    if isinstance(value, list):
        return [substitute(v, bindings) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, bindings) for k, v in value.items()}
    if not isinstance(value, str):
        return value
    output = []
    for literal, field, spec, conversion in Formatter().parse(value):
        output.append(literal)
        if field is None:
            continue
        if spec or conversion or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*', field):
            raise ValueError(f'Unsupported placeholder: {field!r}')
        resolved = bindings
        for component in field.split('.'):
            if not isinstance(resolved, dict) or component not in resolved:
                raise ValueError(f'Unknown placeholder {{{field}}}')
            resolved = resolved[component]
        if not isinstance(resolved, (str, int)) or isinstance(resolved, bool):
            raise ValueError(f'Placeholder {{{field}}} must resolve to text')
        output.append(str(resolved))
    return ''.join(output)


def validate_example(example: Example) -> None:
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
            for field in FIELDS:
                if event.get(field) not in VALUES[field]:
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


class Library:
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

    def generate(self, unit: int, mode: str = 'both') -> Iterator[Example]:
        if mode not in ('single', 'multi', 'both'):
            raise ValueError(f'Unknown mode: {mode}')
        for origin, template in self.templates_for(unit):
            if template.get('combine', False) and len(template['sentences']) != 1:
                raise ValueError(f'{template["id"]}: combine requires single-sentence templates')
            instances = []
            for key, bindings in self.bindings_for(template):
                sentences = tuple(substitute(template['sentences'], bindings))
                example = Example(unit, origin, template['id'], template['rule'], key, sentences)
                validate_example(example)
                instances.append(example)
                if mode == 'both' or (mode == 'single') == (len(sentences) == 1):
                    yield example
            # Pair independent, explicitly named actors from the SAME template.
            # Native multi-sentence templates keep their manually authored context.
            if mode != 'single' and template.get('combine', False):
                for left in instances:
                    for right in instances:
                        if left.binding_id >= right.binding_id:
                            continue
                        a = left.sentences[0]['predicates'][0]['roles']['actor']
                        b = right.sentences[0]['predicates'][0]['roles']['actor']
                        if a == b or a in ('unknown', 'user') or b in ('unknown', 'user'):
                            continue
                        combined = Example(unit, origin, template['id'], template['rule'],
                                           left.binding_id+'+'+right.binding_id,
                                           left.sentences + right.sentences)
                        validate_example(combined)
                        yield combined

    def generators(self) -> dict[int, Callable[[str], Iterator[Example]]]:
        def make_generator(unit: int) -> Callable[[str], Iterator[Example]]:
            def generate(mode: str = 'both') -> Iterator[Example]:
                yield from self.generate(unit, mode)
            generate.__name__ = f'generate_unit_{unit:02d}'
            rules = [t['rule'] for _, t in self.templates_for(unit)]
            generate.__doc__ = self.units[unit]['title']+'\n'+'\n'.join(rules)
            return generate
        return {unit: make_generator(unit) for unit in sorted(self.units)}


def render(example: Example) -> str:
    validate_example(example)
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
            for field in FIELDS:
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


def build(library: Library, out: Path | None, units=None, mode='both', per_unit=False) -> dict:
    """Validate and deduplicate all examples, then write; out=None is a dry run."""
    selected = sorted(library.units if units is None else set(units))
    if not selected or any(unit not in library.units for unit in selected):
        raise ValueError('Select existing units, 1–84')
    examples = {}
    by_unit = {}
    parents = {}
    def root(family):
        parents.setdefault(family, family)
        while parents[family] != family:
            parents[family] = parents[parents[family]]
            family = parents[family]
        return family
    generators = library.generators()
    for unit in selected:
        by_unit[unit] = []
        for example in generators[unit](mode):
            story = render(example)
            key = ' '.join(example.text.casefold().split())
            root(example.family)
            if key in examples:
                previous, old_story = examples[key]
                if old_story != story:
                    raise ValueError(f'Conflicting annotations for {example.text!r}: '
                                     f'{previous.template_id}, {example.template_id}')
                a, b = sorted((root(previous.family), root(example.family)))
                parents[b] = a
            else:
                examples[key] = (example, story)
            by_unit[unit].append(story)
        if mode == 'both' and not by_unit[unit]:
            raise ValueError(f'Unit {unit} has no examples in mode {mode}')
    datasets = {'train': [], 'valid': [], 'test': []}
    for example, story in examples.values():
        datasets[split_for(root(example.family))].append(story)
    # All templates, annotations and splits are checked before opening outputs.
    stats = {}
    for split, stories in datasets.items():
        stats[split] = {'stories': len(stories), 'qa': sum(validate_story(s) for s in stories)}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for split, stories in datasets.items():
            (out / f'{split}.txt').write_text(''.join(stories), encoding='utf-8', newline='\n')
        if per_unit:
            target = out / 'units'
            target.mkdir(exist_ok=True)
            for unit, stories in by_unit.items():
                (target / f'unit_{unit:02d}.txt').write_text(''.join(dict.fromkeys(stories)), encoding='utf-8', newline='\n')
    return {'splits': stats, 'units': {u: len(set(v)) for u, v in by_unit.items()},
            'sentences': sum(len(e.sentences) for e, _ in examples.values()),
            'unique_examples': len(examples), 'families': len({root(e.family) for e, _ in examples.values()})}


def parse_units(value: str) -> list[int]:
    result = set()
    for part in value.split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?', part):
            raise ValueError('Use unit numbers/ranges, for example 1,16,39-44')
        start, _, end = part.partition('-')
        a, b = int(start), int(end or start)
        if not 1 <= a <= b <= 84:
            raise ValueError('Units must be between 1 and 84, in ascending ranges')
        result.update(range(a, b+1))
    return sorted(result)


def format_units(units) -> str:
    ranges = []
    for unit in sorted(set(units)):
        if ranges and unit == ranges[-1][1] + 1:
            ranges[-1][1] = unit
        else:
            ranges.append([unit, unit])
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in ranges) or 'none'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Build the legacy synthetic JSON template library')
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--source', choices=('templates',), default='templates')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--units', help='For example: 1,16,39-44')
    parser.add_argument('--mode', choices=('single', 'multi', 'both'), default='both')
    parser.add_argument('--per-unit', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        library = Library(args.library)
        selected = parse_units(args.units) if args.units is not None else sorted(library.units)
        if args.list:
            for unit in selected:
                print(f'{unit:02d}: {library.units[unit]["title"]} '
                      f'({len(library.templates_for(unit))} synthetic templates)')
            return 0
        stats = build(library, None if args.validate_only else args.out,
                      selected, args.mode, args.per_unit)
        if args.validate_only:
            print(f'Validated {stats["unique_examples"]} unique examples across '
                  f'{sum(bool(n) for n in stats["units"].values())}/{len(selected)} selected units')
        else:
            for split, counts in stats['splits'].items():
                print(f'{args.out / (split+".txt")}: {counts["stories"]} stories, {counts["qa"]} QA')
            print(f'{sum(bool(n) for n in stats["units"].values())}/{len(stats["units"])} units; '
                  f'{stats["families"]} template families; {stats["unique_examples"]} unique examples')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
