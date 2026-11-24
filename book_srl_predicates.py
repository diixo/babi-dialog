"""Validate explicit links from SRL predicates to their source text.

Offsets count Python string characters. Each half-open span selects an exact
piece of the sentence; a predicate can have several pieces and facts may share
an auxiliary. This module checks stored annotation, never infers a predicate.
"""
from __future__ import annotations

import re

from functools import lru_cache
from string import Formatter

SPAN_FIELDS = ('predicate_spans', 'fragment_spans', 'verb_spans')
WORD = re.compile(r"[^\W_]+(?:['\u2018\u2019\u02bc\u2010\u2011-][^\W_]+)*")

# A retained auxiliary can stand for an action mentioned elsewhere. It must
# remain literal in verb, e.g. "promised to call on him ... failed to do it".
ELLIPTICAL_VERBS = frozenset((
    'none', 'do', 'does', 'did', 'done', 'doing', "don't", "doesn't", "didn't",
    'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', "'m", "'s", "'re",
    'have', 'has', 'had', "'ve", "'d", 'can', "can't", 'cannot', 'could', "couldn't",
    'may', 'might', 'must', "mustn't", 'will', "'ll", "won't", 'would', "wouldn't",
    'shall', 'should', "shouldn't", 'need', "needn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "shan't",
))
MODAL_MARKERS = {
    'have to': ('to',), 'need to': ('to',), 'be to': ('to',),
    'be able to': ('able', 'to'), 'be going to': ('going', 'to'),
    'used to': ('use(?:d)?', 'to'), 'had better': ('better',),
    'would rather': ('rather',), 'be likely to': ('likely', 'to'),
    'be unlikely to': ('unlikely', 'to'), 'be supposed to': ('supposed', 'to'),
}

# These are spelling checks against an explicitly annotated lemma, not a
# lemmatizer: no action or source span is inferred or changed at build time.
IRREGULAR_FORMS = {
    'be': 'am is are was were been being', 'bear': 'bore born borne',
    'beat': 'beat beaten', 'become': 'became become', 'begin': 'began begun',
    'bend': 'bent', 'bet': 'bet betted', 'bite': 'bit bitten', 'bleed': 'bled',
    'blow': 'blew blown', 'break': 'broke broken', 'breed': 'bred',
    'bring': 'brought', 'build': 'built', 'burn': 'burned burnt',
    'burst': 'burst', 'buy': 'bought', 'cast': 'cast', 'catch': 'caught',
    'choose': 'chose chosen', 'come': 'came come', 'cost': 'cost', 'cut': 'cut',
    'deal': 'dealt', 'dig': 'dug', 'do': 'does did done', 'draw': 'drew drawn',
    'dream': 'dreamed dreamt', 'drink': 'drank drunk', 'drive': 'drove driven',
    'eat': 'ate eaten', 'fall': 'fell fallen', 'feed': 'fed', 'feel': 'felt',
    'fight': 'fought', 'find': 'found', 'fit': 'fit fitted', 'fly': 'flew flown',
    'forbid': 'forbade forbidden', 'forget': 'forgot forgotten',
    'forgive': 'forgave forgiven', 'freeze': 'froze frozen',
    'get': 'got gotten', 'give': 'gave given', 'go': 'goes went gone',
    'grow': 'grew grown', 'hang': 'hung hanged', 'have': 'has had',
    'hear': 'heard', 'hide': 'hid hidden', 'hit': 'hit', 'hold': 'held',
    'hurt': 'hurt', 'keep': 'kept', 'kneel': 'knelt kneeled',
    'knit': 'knit knitted', 'know': 'knew known', 'lay': 'laid', 'lead': 'led',
    'lean': 'leaned leant', 'learn': 'learned learnt', 'leave': 'left',
    'lend': 'lent', 'let': 'let', 'lie': 'lay lain lied', 'light': 'lit lighted',
    'lose': 'lost', 'make': 'made', 'mean': 'meant', 'meet': 'met',
    'misunderstand': 'misunderstood', 'overhear': 'overheard', 'pay': 'paid',
    'prove': 'proved proven', 'put': 'put', 'quit': 'quit quitted',
    'read': 'read', 'proofread': 'proofread', 'ride': 'rode ridden',
    'ring': 'rang rung', 'rise': 'rose risen', 'run': 'ran run', 'say': 'said',
    'see': 'saw seen', 'sell': 'sold', 'send': 'sent', 'set': 'set',
    'sew': 'sewed sewn', 'shake': 'shook shaken', 'shave': 'shaved shaven',
    'shine': 'shone shined', 'shoot': 'shot', 'show': 'showed shown',
    'shrink': 'shrank shrunk', 'shut': 'shut', 'sing': 'sang sung', 'sink': 'sank sunk',
    'sit': 'sat', 'sleep': 'slept', 'smell': 'smelled smelt', 'speak': 'spoke spoken',
    'spell': 'spelled spelt', 'spend': 'spent', 'spill': 'spilled spilt',
    'spoil': 'spoiled spoilt', 'spread': 'spread', 'stand': 'stood',
    'steal': 'stole stolen', 'stick': 'stuck', 'sting': 'stung', 'strike': 'struck',
    'swear': 'swore sworn', 'sweep': 'swept', 'swim': 'swam swum', 'swing': 'swung',
    'take': 'took taken', 'teach': 'taught', 'tear': 'tore torn', 'tell': 'told',
    'think': 'thought', 'throw': 'threw thrown', 'understand': 'understood',
    'wake': 'woke woken waked', 'wear': 'wore worn', 'win': 'won',
    'wind': 'wound', 'write': 'wrote written',
}
DOUBLED_ENDINGS = frozenset(('admit begin commit control eavesdrop equip forget occur omit '
                           'permit prefer refer regret submit transfer').split())
BRITISH_DOUBLE_L = frozenset(('cancel dial label model quarrel travel').split())
CONTRACTED_HEADS = {"'m": ('be',), "'re": ('be',), "'s": ('be', 'have'),
                    "'ve": ('have',), "'d": ('have', 'would'), "'ll": ('will',)}
PREDICATE_OPERATORS = frozenset((
    "am is are was were be been being have has had having do does did "
    "can cannot could may might must will would shall should need needs dare "
    "get gets got gotten getting going used use able likely unlikely supposed "
    "better rather to not never and or 'm 're 's 've 'd 'll"
).split())


def _surface_words(value: str) -> str:
    return value.casefold().translate(str.maketrans({'‘': "'", '’': "'", 'ʼ': "'", '‐': '-', '‑': '-'}))


@lru_cache(maxsize=None)
def _head_forms(lemma: str) -> frozenset[str]:
    """Permitted spellings of a supplied lexical head (including UK/US forms)."""
    if lemma == 'be':
        return frozenset(('be ' + IRREGULAR_FORMS['be']).split())
    consonant_y = bool(re.search(r'[^aeiou]y$', lemma))
    third = (lemma[:-1] + 'ies' if consonant_y else
             lemma + 'es' if re.search(r'(?:s|sh|ch|x|z|o)$', lemma) else lemma + 's')
    if lemma == 'have':
        third = 'has'
    if lemma.endswith('ie'):
        ing = lemma[:-2] + 'ying'
    elif lemma.endswith('e') and not lemma.endswith(('ee', 'oe', 'ye')):
        ing = lemma[:-1] + 'ing'
    else:
        ing = lemma + 'ing'
    doubled = lemma in DOUBLED_ENDINGS or (bool(re.search(r'[^aeiou][aeiou][^aeiouwxy]$', lemma))
                                          and len(re.findall('[aeiou]', lemma)) == 1)
    if doubled:
        ing = lemma + lemma[-1] + 'ing'
    forms = {lemma, third, ing}
    if lemma in IRREGULAR_FORMS:
        forms.update(IRREGULAR_FORMS[lemma].split())
    else:
        forms.add(lemma[:-1] + 'ied' if consonant_y else
                  lemma + 'd' if lemma.endswith('e') else
                  lemma + lemma[-1] + 'ed' if doubled else lemma + 'ed')
    if lemma in BRITISH_DOUBLE_L:
        forms.update((lemma + 'led', lemma + 'ling'))
    return frozenset(forms)


def _head_matches(surface: str, lemma: str) -> bool:
    if surface.endswith("n't"):
        surface = surface[:-3]
    if lemma in CONTRACTED_HEADS.get(surface, ()):
        return True
    # A stored American lemma can annotate the British source spelling.
    if lemma.endswith('ize') and surface.startswith(lemma[:-3] + 'is'):
        surface = lemma[:-3] + 'iz' + surface[len(lemma) - 1:]
    return surface in _head_forms(lemma)


def _lexical_form_matches(surface: str, action: str) -> bool:
    if action == 'dry-clean':
        return surface.startswith(('dry ', 'dry-')) and _head_matches(surface[4:], 'clean')
    words = surface.split() if '-' in action else re.split(r'[\s-]+', surface)
    lemma, *tail = action.split()
    if tail:
        if words[-len(tail):] != tail:
            return False
        words = words[:-len(tail)]
    # Explicit coordination/repetition can contain several forms of one head.
    heads = [word for word in words if word not in ('and', 'or')]
    return bool(heads) and all(_head_matches(word, lemma) for word in heads)


def validate_verb_construction(fact: dict, verb: str, label: str = '') -> None:
    """Check stored lexical tails and modal markers, without parsing a sentence.

    This cannot decide whether a preposition belongs to an action. That choice
    is manually authored. Once selected, its tail must survive in the surface
    verb and hence in the predicate, including across template substitutions.
    """
    surface = _surface_words(verb)
    action = fact.get('action', '')
    if ' ' in action and surface not in ELLIPTICAL_VERBS:
        # Lexicalized participles keep their source hyphen: worn-out / wear out.
        words = re.split(r'[\s\-\u2010\u2011]+', surface)
        tail = action.casefold().split()[1:]
        if len(words) <= len(tail) or words[-len(tail):] != tail:
            raise ValueError(_prefix(label) + 'Surface verb loses part of its multiword action')
    if surface not in ELLIPTICAL_VERBS and not _lexical_form_matches(surface, action):
        raise ValueError(_prefix(label) + f'Surface verb {verb!r} does not match action {action!r}')
    for field in ('modality', 'ability'):
        modal = fact.get(field, '')
        if '_' in modal:
            raise ValueError(_prefix(label) + f'Use spaces in {field} construction names')
        markers = MODAL_MARKERS.get(modal, ())
        if any(not re.search(r'\b' + word + r'\b', fact['predicate'], re.I)
               for word in markers):
            raise ValueError(_prefix(label) + f'Predicate loses part of {modal!r}')


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


def validate_fact_spans(text: str, facts: list, annotations: dict, label: str = '') -> None:
    """Check stored fragments and surface verbs without inferring new annotation."""
    if not isinstance(annotations, dict):
        raise ValueError(_prefix(label) + 'Missing fact spans')
    validate_predicates(text, facts, annotations.get('predicate_spans'), label)
    for fact in facts:
        if fact.get('status') == 'questioned':
            raise ValueError(_prefix(label) + 'Omit status questioned; keep the explicit utterance type')
        if 'purpose' in fact or fact.get('syntactic_function') == 'purpose':
            raise ValueError(_prefix(label) + 'purpose is no longer supported; '
                             'preserve the object and annotate any separate action')
    words = [(match.start(), match.end()) for match in WORD.finditer(text)]
    for field in ('fragment_spans', 'verb_spans'):
        spans = annotations.get(field)
        if not isinstance(spans, list) or len(spans) != len(facts):
            raise ValueError(_prefix(label) + f'{field} must have one entry per fact')
        for index, selected in enumerate(spans):
            field_label = f'{label}: fact {index + 1}: {field}'
            predicate_from_spans(text, selected, field_label)
            if field == 'fragment_spans':
                if len(selected) != 1:
                    raise ValueError(_prefix(field_label) + 'A fragment must be one nonempty source span')
                start, end = selected[0]
                if any(a < start < b or a < end < b for a, b in words):
                    raise ValueError(_prefix(field_label) + 'A fragment boundary cannot split a word')
            else:
                predicate = annotations['predicate_spans'][index]
                if (not selected and facts[index]['predicate'].casefold()
                        not in ('none', 'to', 'not to', 'to not')):
                    raise ValueError(_prefix(field_label) + 'A retained verb needs a source span')
                if any(not any(a <= start < end <= b for a, b in predicate)
                       for start, end in selected):
                    raise ValueError(_prefix(field_label) + 'A verb must come from its predicate spans')
                validate_verb_construction(facts[index], predicate_from_spans(text, selected), field_label)
                # The lexical head cannot be replaced by an auxiliary while
                # the actual main verb remains elsewhere in the predicate.
                remainder = ' '.join(''.join(
                    text[pos] if not any(c <= pos < d for c, d in selected) else ' '
                    for pos in range(a, b)) for a, b in predicate)
                for token in _surface_words(remainder).split():
                    bare = token[:-3] if token.endswith("n't") else token
                    bare = {'ca': 'can', 'wo': 'will', 'sha': 'shall'}.get(bare, bare)
                    if bare not in PREDICATE_OPERATORS:
                        raise ValueError(_prefix(field_label)
                                         + f'Predicate leaves lexical material outside verb: {token!r}')
    for index, predicate in enumerate(annotations['predicate_spans']):
        start, end = annotations['fragment_spans'][index][0]
        for a, b in predicate:
            if start <= a < b <= end:
                continue
            # Shared auxiliaries and particles still need a fragment that
            # actually contains their source. A peer's longer predicate span
            # may supply a shorter inherited piece, e.g. "must learn" / "must".
            owned = any(fragment[0][0] <= a < b <= fragment[0][1]
                        and any(p <= a < b <= q for p, q in annotations['predicate_spans'][other])
                        for other, fragment in enumerate(annotations['fragment_spans']))
            if not owned:
                raise ValueError(_prefix(label)
                                 + f'Fact {index + 1}: predicate piece has no fragment owner')
    for index, verb in enumerate(annotations['verb_spans']):
        if not verb:
            continue
        start, end = annotations['fragment_spans'][index][0]
        head_start, head_end = verb[0]
        if start <= head_start < head_end <= end:
            if any(not start <= a < b <= end for a, b in verb):
                raise ValueError(_prefix(label)
                                 + f'Fact {index + 1}: fragment excludes part of its verb')
            continue
        shared = any(i != index and other and other[0] == verb[0]
                     for i, other in enumerate(annotations['verb_spans']))
        if not shared:
            raise ValueError(_prefix(label)
                             + f'Fact {index + 1}: fragment excludes its unshared verb')
    # These are reviewed links, not guesses based on matching verb strings.
    # The full semantic values stay in facts; the compact output omits them
    # only when another annotated fragment already represents that event.
    embedded = annotations.get('embedded_roles', {})
    if not isinstance(embedded, dict):
        raise ValueError(_prefix(label) + 'embedded_roles must be a dictionary')
    for source, roles in embedded.items():
        if (not isinstance(source, str) or not source.isdecimal()
                or str(int(source)) != source or not 0 <= int(source) < len(facts)
                or not isinstance(roles, dict) or not roles):
            raise ValueError(_prefix(label) + 'Invalid embedded_roles source')
        index = int(source)
        for role, targets in roles.items():
            if (role not in ('actor', 'content', 'object') or facts[index].get(role, 'none') in ('none', 'unknown')
                    or not isinstance(targets, list) or not targets
                    or any(type(target) is not int or not 0 <= target < len(facts)
                           or target == index for target in targets)
                    or len(set(targets)) != len(targets)):
                raise ValueError(_prefix(label) + 'Invalid embedded_roles target or role')
            if any(annotations['fragment_spans'][target] == annotations['fragment_spans'][index]
                   for target in targets):
                raise ValueError(_prefix(label) + 'An embedded role needs a separate fragment')


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
