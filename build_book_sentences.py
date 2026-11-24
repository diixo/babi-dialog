#!/usr/bin/env python3
"""Write all compatible book sentences as plain text, one example per line."""
import argparse
from pathlib import Path

from build_book_srl import DEFAULT_LIBRARY, DEFAULT_OUT, synchronize_templates, validate


def build(library, out=DEFAULT_OUT):
    sentences = {}
    for unit in range(1, 85):
        if unit not in library.units or not library.units[unit]['templates']:
            raise ValueError(f'Unit {unit} has no prepared templates')
        if unit not in library.coverage['completed_units']:
            raise ValueError(f'Unit {unit} is still incomplete')
    for _, text, facts, _ in library.generate_all():
        validate(text, facts)
        key = ' '.join(text.casefold().split())
        sentences.setdefault(key, text)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sentences.txt').write_text('\n'.join(sentences.values())+'\n',
                                      encoding='utf-8', newline='\n')
    return len(sentences)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Directory for sentences.txt')
    args = parser.parse_args(argv)
    try:
        build(synchronize_templates(args.library), args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
