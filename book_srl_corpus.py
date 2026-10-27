"""Book passages and their manually reviewed SRL annotations.

No statistical parser or inferred semantic labels are used here. Synchronization
only inventories the source; every new passage remains pending until reviewed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from book_srl_source import extract_passages


CORPUS_NAME = 'sentences.json'
REVIEW_KINDS = {'sentences', 'alternatives', 'mixed', 'lexical_fragment', 'exercise_gap'}


def review_variants(review):
    """Return explicit annotations, including sentences within mixed passages."""
    if review.get('kind') == 'mixed':
        return [[sentence for part in review['parts']
                 if part['kind'] == 'sentences' for sentence in part['sentences']]]
    return review.get('variants', [])


def synchronize(book: Path, path: Path) -> dict:
    """Inventory every passage, preserving annotations only for identical text."""
    from build_book_srl import read_json
    old = read_json(path) if path.exists() else {'records': []}
    old_by_id = {r['id']: r for r in old['records']}
    records = []
    for passage in extract_passages(book):
        record = passage.to_dict()
        previous = old_by_id.get(record['id'])
        record['review'] = {'state': 'pending'}
        if previous and previous['text'] == record['text']:
            record['review'] = previous.get('review', {'state': 'pending'})
        records.append(record)
    current_ids = {p['id'] for p in records}
    removed = [r for r in old['records'] if r['id'] not in current_ids
               and r.get('review', {}).get('state') == 'checked']
    if removed:
        raise ValueError(f'{len(removed)} reviewed passages changed or disappeared; '
                         'reconcile their source and annotations before synchronization')
    result = {'schema_version': 1, 'source_sha256': hashlib.sha256(book.read_bytes()).hexdigest(),
              'annotation_method': 'manual', 'records': records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


class CorpusLibrary:
    def __init__(self, directory: Path, book: Path):
        from build_book_srl import read_json
        self.book = Path(book)
        data = read_json(Path(directory)/CORPUS_NAME)
        if data.get('annotation_method') != 'manual':
            raise ValueError('The book corpus must contain manual annotations')
        if data.get('source_sha256') != hashlib.sha256(self.book.read_bytes()).hexdigest():
            raise ValueError('The book changed; run --sync-book to reconcile the corpus')
        source = extract_passages(self.book)
        self.records = data['records']
        if len(self.records) != len(source):
            raise ValueError('Corpus does not inventory every English passage; run --sync-book')
        for record, passage in zip(self.records, source):
            if any(record.get(k) != v for k,v in passage.to_dict().items()):
                raise ValueError(f'Stale or altered source record: {passage.id}')
        self.by_id = {r['id']:r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError('Duplicate corpus passage IDs')
        titles = {int(m[1]):m[2] for m in re.finditer(r'^### Урок (\d+)\. (.+)$',
                  self.book.read_text(encoding='utf-8'),re.M)}
        self.units = {u:{'unit':u,'title':title} for u,title in titles.items()}
        self.by_unit = {u:[r for r in self.records if r['unit']==u] for u in self.units}
        for record in self.records:
            self.validate_review(record)

    def validate_review(self, record):
        from build_book_srl import Example, validate_example
        review = record.get('review', {})
        if review.get('state') == 'pending':
            if set(review) != {'state'}:
                raise ValueError(f'{record["id"]}: pending text must not carry training labels')
            return
        if review.get('state') != 'checked' or review.get('kind') not in REVIEW_KINDS:
            raise ValueError(f'{record["id"]}: invalid manual review')
        if not review.get('rule'):
            raise ValueError(f'{record["id"]}: explain the manual classification/rule')
        kind = review['kind']
        if kind == 'mixed':
            parts = review.get('parts', [])
            if 'variants' in review or not isinstance(parts, list) or not parts:
                raise ValueError(f'{record["id"]}: mixed review needs explicit source parts')
            for part in parts:
                if not isinstance(part, dict) or not isinstance(part.get('text'), str) or not part['text']:
                    raise ValueError(f'{record["id"]}: invalid mixed source part')
                if part.get('kind') == 'sentences':
                    sentences = part.get('sentences', [])
                    if not sentences or ' '.join(s['text'] for s in sentences) != part['text'].strip():
                        raise ValueError(f'{record["id"]}: annotations must cover each sentence part exactly')
                elif part.get('kind') == 'fragment':
                    if not part.get('rule') or 'sentences' in part:
                        raise ValueError(f'{record["id"]}: fragment needs a reason and cannot supply annotations')
                else:
                    raise ValueError(f'{record["id"]}: invalid mixed source part kind')
            if {p['kind'] for p in parts} != {'sentences', 'fragment'}:
                raise ValueError(f'{record["id"]}: mixed review requires sentences and fragments')
            if ''.join(p['text'] for p in parts) != record['text']:
                raise ValueError(f'{record["id"]}: parts must cover the entire source passage exactly')
            sentences = review_variants(review)[0]
            validate_example(Example(record['unit'], record['unit'], record['id'],
                                     review['rule'], '0', tuple(sentences)))
        elif kind in ('sentences','alternatives'):
            variants = review.get('variants', [])
            if not variants:
                raise ValueError(f'{record["id"]}: missing manual annotations')
            for index, sentences in enumerate(variants):
                example = Example(record['unit'],record['unit'],record['id'],review['rule'],str(index),tuple(sentences))
                validate_example(example)
                if kind == 'sentences' and example.text != record['text']:
                    raise ValueError(f'{record["id"]}: annotation must cover the entire source passage exactly')
            if kind == 'sentences' and len(variants) != 1:
                raise ValueError(f'{record["id"]}: use alternatives for multiple readings')
        elif kind == 'exercise_gap':
            targets = review.get('solutions', [])
            if not targets:
                raise ValueError(f'{record["id"]}: incomplete exercise needs its completed solution')
            for identifier in targets:
                target = self.by_id.get(identifier)
                if not target or target['section'] != 'solution' or target['unit'] != record['unit']:
                    raise ValueError(f'{record["id"]}: invalid solution link {identifier}')
                if any(target[k] != record[k] for k in ('exercise','part','item')):
                    raise ValueError(f'{record["id"]}: solution belongs to a different exercise item')
                if target['review'].get('state') != 'checked':
                    raise ValueError(f'{record["id"]}: linked solution has not been manually checked')
        elif 'variants' in review:
            raise ValueError(f'{record["id"]}: a lexical fragment cannot supply sentence annotations')

    def coverage(self, units=None):
        result = {}
        for unit in sorted(self.units if units is None else units):
            records = self.by_unit[unit]
            kinds = Counter(r['review'].get('kind','pending') for r in records)
            result[unit] = {'passages':len(records),'pending':kinds['pending'],
                'checked':len(records)-kinds['pending'],
                'sentences':sum(len(v) for r in records for v in review_variants(r['review'])),
                'kinds':dict(kinds)}
        return result

    def require_complete(self, units):
        from build_book_srl import format_units
        incomplete = {u:c['pending'] for u,c in self.coverage(units).items() if c['pending']}
        if incomplete:
            raise ValueError(f'Manual annotation is incomplete: {sum(incomplete.values())} passages '
                             f'in units {format_units(incomplete)}. '
                             'Use --coverage for details, or omit --units to build fully reviewed units. '
                             'No dataset files were written.')

    def generate(self, unit, mode='both'):
        from build_book_srl import Example, validate_example
        if mode not in ('single','multi','both'):
            raise ValueError(f'Unknown mode: {mode}')
        self.require_complete([unit])
        for record in self.by_unit[unit]:
            review = record['review']
            # Gapped tasks are represented by their reviewed completed solutions.
            for index, sentences in enumerate(review_variants(review)):
                if mode=='single' and len(sentences)!=1 or mode=='multi' and len(sentences)==1:
                    continue
                family = ':'.join((record['source'], record['part'] or '', record['item'] or record['id']))
                example = Example(unit,unit,family,review['rule'],str(index),tuple(deepcopy(sentences)))
                validate_example(example)
                yield example

    def generators(self):
        def make(unit):
            def generate(mode='both'):
                yield from self.generate(unit,mode)
            generate.__name__ = f'generate_unit_{unit:02d}'
            generate.__doc__ = self.units[unit]['title']
            return generate
        return {u:make(u) for u in self.units}
