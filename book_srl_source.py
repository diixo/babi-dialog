"""Extract every Latin-script passage in the 84 lessons, with source locations.

Extraction is deliberately separate from annotation. Words, incomplete exercises
and grammatical formulas are retained as candidates, not silently discarded or
mislabelled as complete sentences. The appendix and duplicate answer keys are
outside the lesson corpus.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re


CYRILLIC = re.compile(r'[\u0400-\u04ff]+')
LATIN = re.compile(r'[A-Za-z]')
UNIT = re.compile(r'^### Урок (\d+)\. (.+)$')
EXERCISE = re.compile(r'^#### Упражнение (\d+\.\d+)')
ITEM = re.compile(r'^\s*(\d+)[.)]\s+')
PART = re.compile(r'^\*\*Часть (\d+)\*\*$')


@dataclass(frozen=True)
class Passage:
    id: str
    unit: int
    source: str
    section: str
    exercise: str | None
    part: str | None
    item: str | None
    line: int
    column: int
    text: str
    raw: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_english(value: str) -> str:
    # Strip Markdown emphasis, preserving apostrophes and internal punctuation.
    # Printed line wraps do not mark the end of an English sentence.
    value = re.sub(r'<br\s*/?>', ' ', value, flags=re.I)
    value = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', value)
    value = value.replace('**', '').replace('*', '').replace('`', '')
    value = re.sub(r'\s+', ' ', value).strip()
    value = re.sub(r'^[\s|:;,=—–•()\[\]«»“”"/]+', '', value)
    value = re.sub(r'^\.(?!\.)\s*(?=[A-Za-z])', '', value)
    value = re.sub(r'\s+[—–]\s*$', '', value)
    # A translation can leave an unmatched opening parenthesis at the boundary.
    while value.endswith('('):
        value = value[:-1].rstrip()
    if value.endswith(')') and value.count(')') > value.count('('):
        value = value[:-1].rstrip()
    return value.strip(' |•')


def extract_passages(book: Path) -> list[Passage]:
    """Keep every English-bearing body segment, including those in Russian prose.

    Table cells are independent passages. Ordinary source lines retain their
    internal multi-sentence context. Numbered items are tracked separately from
    the paragraph text. Offsets refer to the unchanged original Markdown.
    """
    lines = Path(book).read_text(encoding='utf-8').splitlines()
    passages = []
    unit, source, section = None, '', 'theory'
    exercise = part = item = None
    titles = {}
    occurrences = Counter()
    for line_number, raw in enumerate(lines, 1):
        heading = UNIT.fullmatch(raw.strip())
        if heading:
            unit = int(heading[1])
            titles[unit] = heading[2]
            source, section = f'unit-{unit}', 'theory'
            exercise = part = item = None
            continue
        if raw.strip() == '<a id="appendix"></a>':
            break
        if unit is None or not raw.strip():
            continue
        if raw.startswith('<a ') or raw.startswith('#'):
            heading = EXERCISE.match(raw)
            if heading:
                exercise = heading[1]
                source = 'exercise-'+exercise.replace('.', '-')
                section, part, item = 'task', None, None
            continue
        if raw.strip() in ('**Задание**', '**Полное решение**'):
            section = 'task' if raw.strip() == '**Задание**' else 'solution'
            part = item = None
            continue
        heading = PART.fullmatch(raw.strip())
        if heading:
            part, item = heading[1], None
            continue
        # Editorial notes are not lesson examples; preserve the book untouched.
        if raw.startswith('>') or re.fullmatch(r'[|\s:\-]+', raw):
            continue
        numbered = ITEM.match(raw)
        start = numbered.end() if numbered else 0
        if numbered and section != 'theory':
            item = numbered[1]
        if raw[start:].startswith('- '):
            start += 2
        # Cyrillic runs and table separators delimit English-bearing spans.
        # Other scripts (e.g. IPA) are kept; classification happens later.
        boundaries = list(re.finditer(r'(?<!°)[\u0400-\u04ff]+|\||<(?!br\s*/?>)[^>]*>',
                                     raw[start:], re.I))
        offset = start
        spans = []
        for boundary in boundaries:
            end = start+boundary.start()
            spans.append((offset, end))
            offset = start+boundary.end()
        spans.append((offset, len(raw)))
        for left, right in spans:
            original = raw[left:right]
            if not LATIN.search(original):
                continue
            text = clean_english(original)
            if not LATIN.search(text):
                continue
            identity = '\0'.join((str(unit),source,section,part or '',item or '',text))
            digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]
            occurrences[digest] += 1
            passages.append(Passage(f'u{unit:02d}-{digest}-{occurrences[digest]}',unit,
                source,section,exercise,part,item,line_number,left+1,text,original))
    if set(titles) != set(range(1,85)):
        raise ValueError('Expected all 84 lesson headings before the appendix')
    if set(p.unit for p in passages) != set(titles):
        raise ValueError('Extraction left a lesson without English passages')
    return passages


def source_summary(passages: list[Passage]) -> dict:
    return {'passages': len(passages), 'units': len({p.unit for p in passages}),
            'sections': dict(Counter(p.section for p in passages))}
