#!/usr/bin/env python3
"""Synchronize English JSON sentences with their manually authored templates.

Existing variable templates and lexical choices are preserved. A new manually
annotated sentence starts as a fixed template until its substitutions are added.
No Markdown/PDF reading, automatic SRL or lexical guessing is performed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from book_srl_annotations import read_json
from book_srl_corpus import CorpusLibrary
from build_book_srl import DEFAULT_LIBRARY, Library, build


def prepare(directory=DEFAULT_LIBRARY):
    directory = Path(directory)
    corpus = CorpusLibrary(directory)
    catalogue_path = directory / 'fact_templates.json'
    old = read_json(catalogue_path) if catalogue_path.exists() else {'lessons': []}
    templates = {t['id']: t for lesson in old['lessons'] for t in lesson['templates']}
    lessons = []
    for unit, records in corpus.by_unit.items():
        prepared = []
        for record in records:
            template = deepcopy(templates.get(record['id']))
            if template is None:
                template = dict(id=record['id'],family=record['id'],text=record['text'],
                    slots={},original={},constraints=[],facts=record['facts'],
                    predicate_spans=record['predicate_spans'])
            template['source'] = dict(sentence=record['id'],unit=unit,text=record['text'])
            prepared.append(template)
        lessons.append(dict(unit=unit,templates=prepared))
    catalogue = dict(schema_version=1,coverage=dict(sentences=len(corpus.records),
        completed_units=corpus.completed_units,
        pending_units=[u for u in range(1,85) if u not in corpus.completed_units]),lessons=lessons)
    content = json.dumps(catalogue,ensure_ascii=False,indent=2)+'\n'
    # Check original texts/facts and all variants before replacing the catalogue.
    with tempfile.TemporaryDirectory() as folder:
        staging = Path(folder)
        (staging / 'fact_templates.json').write_text(content,encoding='utf-8')
        for name in ('sentences.json','slot_values.json'):
            (staging / name).write_bytes((directory / name).read_bytes())
        build(Library(staging))
    catalogue_path.write_text(content,encoding='utf-8',newline='\n')
    return catalogue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,default=DEFAULT_LIBRARY)
    args = parser.parse_args()
    catalogue = prepare(args.library)
    print(f'Prepared {catalogue["coverage"]["sentences"]} English sentence templates from JSON.')
