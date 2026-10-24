#!/usr/bin/env python3
"""Build bAbI-style semantic dialogue stories. Text only; each QA is tab-separated.

For each user utterance: actor/action/intention/object extraction questions first,
then context-dependent dialogue questions. Supporting indices refer to prior lines.

Version 2.
"""
import argparse
import random
from pathlib import Path

SEED = 42
ACTIONS = [
    ('drink', 'something to drink', 'What would you like to drink?', ['tea', 'coffee', 'water', 'juice', 'milk']),
    ('eat', 'something to eat', 'What would you like to eat?', ['pizza', 'soup', 'salad', 'pasta']),
    ('read', 'something to read', 'What would you like to read?', ['a novel', 'a magazine', 'a comic', 'a biography']),
    ('watch', 'something to watch', 'What would you like to watch?', ['a movie', 'a documentary', 'a comedy', 'a series']),
    ('listen', 'something to listen to', 'What would you like to listen to?', ['jazz', 'rock music', 'a podcast', 'classical music']),
    ('buy', 'something to buy', 'What would you like to buy?', ['a laptop', 'a notebook', 'a backpack', 'a phone']),
    ('cook', 'something to cook', 'What would you like to cook?', ['rice', 'pasta', 'soup', 'vegetables']),
    ('play', 'something to play', 'What would you like to play?', ['chess', 'a card game', 'football', 'a video game']),
    ('visit', 'a place to visit', 'Where would you like to go?', ['Rome', 'Paris', 'Berlin', 'Prague', 'London']),
    ('learn', 'something to learn', 'What would you like to learn?', ['Python', 'French', 'guitar', 'photography']),
]
UNKNOWN = ["I don't know.", 'Not sure.', "I haven't decided.", 'No idea.', "I can't decide.", 'Maybe.', 'Hard to say.']
DECLINED = ["I'd rather not decide now.", "I don't want to answer.", "Let's leave it for later."]
CONFIRM = ['Yes.', 'Yeah.', 'Sure.', 'That sounds good.', 'Yep', 'Absolutely.', 'Of course.']
REJECT = ['No.', 'Not really.', "I don't think so.", "Nope"]
REVISIONS = ['Actually, {value}.', 'I changed my mind. {value}.', 'On second thought, {value}.']
PROMPTS = ['I want {thing}.', "I'd like {thing}.", "I'm thinking of {thing}."]
ADVANCE = {
    'drink': ['Do you prefer it hot or cold?', 'That sounds refreshing.'],
    'eat': ['Would you like to make it yourself?', 'That sounds delicious.'],
    'read': ['What kind of {object} do you enjoy?', 'Sounds like a good choice for some quiet reading.'],
    'watch': ['Are you planning to watch it tonight?', 'That sounds like a good way to unwind.'],
    'listen': ['Do you usually listen to it while relaxing?', 'That sounds like a nice choice.'],
    'buy': ['Have you decided where to get it?', 'Sounds like you have something specific in mind.'],
    'cook': ['Are you making it for yourself?', 'That sounds tasty.'],
    'play': ['Will you play with friends?', 'That sounds fun.'],
    'visit': ['Have you been there before?', 'That sounds like a lovely place to visit.'],
    'learn': ['Are you just getting started?', 'That sounds like a useful thing to learn.'],
}

TASKS = [
    'roles',
    'resolution',
    'questions',
    'history',
    'yes_context',
    'ambiguous_yes',
    'revision',
    'decline',
    'loop'
    ]


def intro_text(rng, action, value=None):
    verb, thing, _, _ = action
    if value is None:
        return rng.choice(PROMPTS).format(thing=thing)
    if verb == 'visit':
        return rng.choice([f'I want to visit {value}.', f"I'd like to visit {value}."])
    if verb == 'listen':
        return rng.choice([f'I want to listen to {value}.', f"I'd like to listen to {value}."])
    if verb == 'play':
        return rng.choice([f'I want to play {value}.', f"I'd like to play {value}."])
    if verb == 'learn':
        return rng.choice([f'I want to learn {value}.', f"I'd like to learn {value}."])
    if verb == 'cook':
        return rng.choice([f'I want to cook {value}.', f"I'd like to cook {value}."])
    if verb == 'buy':
        return rng.choice([f'I want to buy {value}.', f"I'd like to buy {value}."])
    if verb == 'eat':
        return rng.choice([f'I want to eat {value}.', f"I'd like to eat {value}."])
    if verb == 'watch':
        return rng.choice([f'I want to watch {value}.', f"I'd like to watch {value}."])
    if verb == 'read':
        return rng.choice([f'I want to read {value}.', f"I'd like to read {value}."])
    return rng.choice([f'I want to drink {value}.', f"I'd like to drink {value}."])


def make_dialogue(rng, task, action):
    verb, thing, ask, values = action
    value = rng.choice(values)
    initial_known = task == 'questions' and rng.choice([True, False])
    intro = intro_text(rng, action, value if initial_known else None)
    # Each User turn stores information explicitly supplied by that turn.
    turns = [('User', intro, value if initial_known else None)]
    if task in ('roles', 'questions'):
        return turns
    turns.append(('Assistant', ask, None))
    if task == 'resolution':
        known = rng.choice([True, False])
        turns.append(('User', (value.capitalize() + '.') if known else rng.choice(UNKNOWN), value if known else None))
    elif task == 'history':
        for i in range(rng.randint(1, 4)):
            turns.append(('User', rng.choice(UNKNOWN), None))
            turns.append((
                'Assistant',
                rng.choice([ask, 'Could you choose one option?', 'Do you have a preference yet?']),
                None))
        if rng.choice([True, False]):
            turns.append(('User', value.capitalize() + '.', value))
        else:
            turns.append(('User', rng.choice(UNKNOWN), None))
    elif task == 'yes_context':
        turns[-1] = ('Assistant', f'Would you like {value}?', None)
        accepted = rng.choice([True, False])
        turns.append(('User', rng.choice(CONFIRM if accepted else REJECT), value if accepted else None))
    elif task == 'ambiguous_yes':
        turns.append(('User', rng.choice(CONFIRM), None))
    elif task == 'revision':
        other = rng.choice([v for v in values if v != value])
        turns.append(('User', value.capitalize() + '.', value))
        turns.append(('Assistant', f'Okay, {value}.', None))
        turns.append(('User', rng.choice(REVISIONS).format(value=other.capitalize()), other))
    elif task == 'decline':
        turns.append(('User', rng.choice(DECLINED), 'declined'))
    elif task == 'loop':
        for i in range(rng.randint(2, 5)):
            turns.append(('User', rng.choice(UNKNOWN), None))
            if i < 2:  # visibly repetitive assistant history for diagnosis
                turns.append(('Assistant', ask, None))
    return turns


def render(rng, task, action):
    verb, _, ask, _ = action
    turns = make_dialogue(rng, task, action)
    rows = []
    state = None
    uncertain_streak = 0
    uncertainty_sources = []
    state_source = None
    last_assistant_line = None

    def fact(line):
        rows.append(line)
        return len(rows)

    def question(label, answer, refs):
        refs = sorted(set(refs))
        assert refs and all(1 <= i < len(rows) + 1 for i in refs)
        return fact(f'{label}\t{answer}\t{" ".join(map(str, refs))}')

    for turn_index, (role, utterance, observed) in enumerate(turns):
        utterance_line = fact(f'{role}: {utterance}')
        if role == 'Assistant':
            last_assistant_line = utterance_line
            continue
        if observed == 'declined':
            state = 'declined'
            state_source = utterance_line
            uncertain_streak += 1
            uncertainty_sources.append(utterance_line)
        elif observed is not None:
            state = observed
            state_source = utterance_line
            uncertain_streak = 0
            uncertainty_sources.clear()
        else:
            uncertain_streak += 1

        # All extraction questions about this one utterance FIRST.
        question('actor?', 'user', [utterance_line])
        question('action?', verb, [utterance_line] if last_assistant_line is None else [last_assistant_line, utterance_line])
        question('intention?', verb, [utterance_line] if last_assistant_line is None else [last_assistant_line, utterance_line])
        # An ambiguous Yes/No answer does not itself state a chosen object; resolved
        # value is reported by the separate contextual question below.
        literal_object = observed if (observed is not None and observed != 'declined' and utterance.strip() not in CONFIRM) else 'unknown'
        question('object?', literal_object, [utterance_line])

        # Only after all extractions do we ask the dialogue/state questions.
        effective_object = 'unknown' if state in (None, 'declined') else state
        support = [utterance_line]
        if state_source is not None and state_source != utterance_line:
            support.append(state_source)
        if last_assistant_line is not None:
            support.append(last_assistant_line)
        question('current object?', effective_object, support)
        question('is object known?', 'yes' if effective_object != 'unknown' else 'no', support)
        if state == 'declined' or (task == 'loop' and uncertain_streak >= 2):
            next_action = 'stop_clarify'
            next_question = 'none'
        elif effective_object == 'unknown':
            next_action = 'clarify'
            next_question = ask
        else:
            next_action = 'advance'
            next_question = 'none'
        decision_support = support + (uncertainty_sources if next_action == 'stop_clarify' else [])
        next_action_line = question('next action?', next_action, decision_support)
        next_question_line = question('ask about object?', next_question, decision_support)
        if next_action == 'clarify':
            response = ask
        elif next_action == 'stop_clarify':
            response = ("That's okay. We can talk about something else."
                        if state == 'declined' else
                        "No rush. We can decide later.")
        else:
            response = rng.choice(ADVANCE[verb]).format(object=effective_object)
        # When a scripted assistant turn follows, the QA target must match it.
        if turn_index + 1 < len(turns) and turns[turn_index + 1][0] == 'Assistant':
            response = turns[turn_index + 1][1]
        question('assistant response?', response,
                 [next_action_line, next_question_line] + decision_support)
    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'


def build(out, counts):
    out.mkdir(parents=True, exist_ok=True)
    for split, n in counts.items():
        rng = random.Random(SEED + {
            'train': 0,
            'valid': 10000,
            'test': 20000
            }[split])
        stories = []
        for i in range(n):
            task = TASKS[i % len(TASKS)]
            action = ACTIONS[(i // len(TASKS) + rng.randrange(len(ACTIONS))) % len(ACTIONS)]
            stories.append(render(rng, task, action))
        rng.shuffle(stories)
        for story in stories:
            rows = story.strip().splitlines()
            assert rows[0].startswith('1 User:')
            for j, row in enumerate(rows, 1):
                assert row.startswith(f'{j} '), row
                body = row[len(str(j)) + 1:]
                if '\t' in body:
                    parts = body.split('\t')
                    assert len(parts) == 3 and parts[1]
                    assert all(1 <= int(k) < j for k in parts[2].split())
        path = out / f'{split}.txt'
        path.write_text(''.join(stories), encoding='utf-8')
        print(f'{path}: {n} stories')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'products')
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--valid', type=int, default=200)
    parser.add_argument('--test', type=int, default=200)

    args = parser.parse_args()
    build(args.out, {
        'train': args.train,
        'valid': args.valid,
        'test': args.test
        })
