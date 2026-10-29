"""Validate explicit links from SRL predicates to their source text.

Offsets count Python string characters. Each half-open span selects an exact
piece of the sentence; a predicate can have several pieces and facts may share
an auxiliary. This module checks stored annotation, never infers a predicate.
"""
from __future__ import annotations

import re

from string import Formatter


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


def _prefix(label: str) -> str:
    return f'{label}: ' if label else ''


def predicate_from_spans(text: str, spans: list, label: str = '') -> str:
    """Read an ordered predicate from zero-based, half-open character spans.

    An empty list represents a predicate omitted from the sentence. Adjacent
    or separated pieces are joined with one space; characters inside a piece
    are retained exactly, including case and contraction apostrophes.
    """
    prefix = _prefix(label)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(prefix + 'Predicate source text must be nonempty')
    if not isinstance(spans, list):
        raise ValueError(prefix + 'Predicate spans must be a list')
    pieces = []
    previous_end = 0
    for index, span in enumerate(spans, 1):
        if (not isinstance(span, list) or len(span) != 2
                or any(type(offset) is not int for offset in span)):
            raise ValueError(prefix + f'Predicate span {index} must contain two integer offsets')
        start, end = span
        if not 0 <= start < end <= len(text):
            raise ValueError(prefix + f'Predicate span {index} is empty or outside the source text')
        if start < previous_end:
            raise ValueError(prefix + f'Predicate span {index} overlaps or precedes an earlier span')
        piece = text[start:end]
        if not piece.strip() or piece != piece.strip():
            raise ValueError(prefix + f'Predicate span {index} includes boundary whitespace')
        if any(character in piece for character in '\t\r\n'):
            raise ValueError(prefix + f'Predicate span {index} includes a line break or tab')
        pieces.append(piece)
        previous_end = end
    return ' '.join(pieces) if pieces else 'none'


def validate_predicates(text: str, facts: list, spans: list, label: str = '') -> None:
    """Require every fact's predicate to match its explicitly stored spans.

    ``spans[i]`` belongs to ``facts[i]``. An empty inner list is valid only for
    ``predicate: none``; repeated words remain distinguishable by their
    offsets. A shared auxiliary may occur in more than one fact's spans.
    """
    prefix = _prefix(label)
    if not isinstance(facts, list) or not facts:
        raise ValueError(prefix + 'Predicates require a nonempty list of facts')
    if not isinstance(spans, list) or len(spans) != len(facts):
        raise ValueError(prefix + 'Predicate spans must have one entry per fact')
    for index, (fact, selected) in enumerate(zip(facts, spans), 1):
        fact_label = f'{label}: fact {index}' if label else f'fact {index}'
        if (not isinstance(fact, dict) or 'predicate' not in fact
                or not isinstance(fact['predicate'], str) or not fact['predicate'].strip()):
            raise ValueError(_prefix(fact_label) + 'Missing or invalid predicate')
        predicate = predicate_from_spans(text, selected, fact_label)
        if fact['predicate'] != predicate:
            raise ValueError(_prefix(fact_label)
                             + f'Predicate {fact["predicate"]!r} differs from its source spans {predicate!r}')


def expand_predicates(template_text: str, bindings: dict, template_spans: list,
                      label: str = '') -> tuple[str, list, list[str]]:
    """Expand a template and move its stored predicate spans with substitutions.

    Template spans use offsets in the raw template, including placeholder
    characters. Their boundaries may enclose whole placeholders but may not
    fall inside a placeholder. For example, a subject and its contracted
    auxiliary need separate stored slot attributes when their boundary is a
    predicate boundary. A span selecting a whole placeholder may disappear
    when that stored optional field expands to the empty string. No searching
    or grammatical inference is performed.

    Returns the expanded text, per-fact expanded spans, and predicate strings.
    """
    prefix = _prefix(label)
    if not isinstance(template_text, str) or not template_text.strip():
        raise ValueError(prefix + 'Predicate template text must be nonempty')
    if not isinstance(bindings, dict):
        raise ValueError(prefix + 'Predicate template bindings must be a dictionary')
    if not isinstance(template_spans, list):
        raise ValueError(prefix + 'Predicate template spans must be a list')
    # Use the established substitution grammar, including its rejection of
    # attribute access, conversions, formats and unknown fields.
    text = substitute(template_text, bindings)
    boundaries = {0: 0}
    cursor = 0
    expanded_length = 0
    while cursor < len(template_text):
        if template_text[cursor:cursor + 2] in ('{{', '}}'):
            end = cursor + 2
            replacement_length = 1
        elif template_text[cursor] == '{':
            end = template_text.index('}', cursor) + 1
            replacement_length = len(substitute(template_text[cursor:end], bindings))
        else:
            end = cursor + 1
            replacement_length = 1
        expanded_length += replacement_length
        boundaries[end] = expanded_length
        cursor = end
    expanded_spans = []
    predicates = []
    for index, selected in enumerate(template_spans, 1):
        fact_label = f'{label}: fact {index}' if label else f'fact {index}'
        # Validate ordering, bounds, shapes and whitespace before indexing.
        predicate_from_spans(template_text, selected, fact_label)
        mapped = []
        for start, end in selected:
            if start not in boundaries or end not in boundaries:
                raise ValueError(_prefix(fact_label)
                                 + 'Predicate span boundary splits a placeholder or escaped brace')
            if (boundaries[start] == boundaries[end]
                    and re.fullmatch(r'\{[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*\}',
                                     template_text[start:end])):
                continue
            mapped.append([boundaries[start], boundaries[end]])
        predicates.append(predicate_from_spans(text, mapped, fact_label))
        expanded_spans.append(mapped)
    return text, expanded_spans, predicates
