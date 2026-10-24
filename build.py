#!/usr/bin/env python3
"""Generate reproducible bAbI-style semantic dialogue QA tasks (English).

Each independent story starts at 1 and ends with a tab-separated QA line:
  N <question>\t<answer>\t<supporting line numbers>
No JSON is written. Semantic fact lines are explicit training context for some tasks.
"""
import argparse
import random
from pathlib import Path

SEED = 42

ACTIONS = [
    ('drink','something to drink','drink','What would you like to drink?', ['tea', 'coffee', 'water', 'juice', 'milk']),
    ('eat','something to eat','eat','What would you like to eat?', ['pizza', 'soup', 'salad', 'pasta']),
    ('read','something to read','read','What would you like to read?', ['a novel', 'a magazine', 'a comic', 'a biography']),
    ('watch','something to watch','watch','What would you like to watch?', ['a movie', 'a documentary', 'a comedy', 'a series']),
    ('listen','something to listen to','listen','What would you like to listen to?', ['jazz', 'rock music', 'a podcast', 'classical music']),
    ('buy','something to buy','buy','What would you like to buy?', ['a laptop', 'a notebook', 'a backpack', 'a phone']),
    ('cook','something to cook','cook','What would you like to cook?', ['rice', 'pasta', 'soup', 'vegetables']),
    ('play','something to play','play','What would you like to play?', ['chess', 'a card game', 'football', 'a video game']),
    ('visit','a place to visit','visit','Where would you like to visit?', ['Rome', 'Paris', 'Berlin', 'Prague', 'London']),
    ('learn','something to learn','learn','What would you like to learn?', ['Python', 'French', 'guitar', 'photography']),
]

UNKNOWN = [
    'I don\'t know.',
    'Not sure.',
    'I haven\'t decided.',
    'No idea.',
    'I can\'t decide.',
    'Maybe.',
    'Hard to say.'
    ]

DECLINED = ["I'd rather not decide now.", "I don't want to answer.", "Let's leave it for later."]
CONFIRM = ['Yes.', 'Yeah.', 'Sure.', 'That sounds good.']
REJECT = ['No.', 'Not really.', 'I don\'t think so.']
REVISIONS = ['Actually, {value}.', 'I changed my mind. {value}.', 'On second thought, {value}.']
PROMPTS = ['I want {thing}.', "I'd like {thing}.", "I'm thinking of {thing}."]


def qa(lines, question, answer, refs):
    assert refs and all(1 <= r <= len(lines) for r in refs)
    return '\n'.join(f'{i} {line}' for i, line in enumerate(lines,1)) + f'\n{len(lines)+1} {question}\t{answer}\t{" ".join(map(str,sorted(set(refs))))}\n\n'


def example(rng, task, action):
    verb, thing, intention, ask, values = action
    intro = rng.choice(PROMPTS).format(thing=thing)
    value = rng.choice(values)
    u = rng.choice(UNKNOWN)

    # Each task ends in a supervised QA; alternate explicitly-stated semantic facts and semantic queries.
    if task == 'roles':
        lines = [f'User: {intro}']
        key = rng.choice(['actor', 'action', 'intention', 'object', 'certainty'])
        answers = {
            'actor':'user',
            'action':verb,
            'intention':intention,
            'object':'unknown',
            'certainty':'unknown'
            }
        return qa(lines, key + '?', answers[key], [1])

    if task == 'resolve':
        known = rng.choice([True,False])
        answer = value.capitalize()+'.' if known else u
        lines = [f'User: {intro}',f'Assistant: {ask}',f'User: {answer}']
        key = rng.choice(['object','certainty','missing'])
        answers = {
            'object':value if known else 'unknown',
            'certainty':'known' if known else 'unknown',
            'missing':'none' if known else 'object'
            }
        return qa(lines,key+'?', answers[key], [2,3])

    if task == 'questions':
        known = rng.choice([True,False])
        lines = [f'User: {intro}',
                 'actor user',
                 f'action {verb}',
                 f'intention {intention}',
                 f'object {value if known else "unknown"}',
                 f'missing {"none" if known else "object"}'
                 ]
        return qa(lines,'ask about object?', 'none' if known else ask, [3, 5, 6])

    if task == 'history':
        n = rng.randint(1,4)
        lines = [f'User: {intro}', f'Assistant: {ask}']
        for i in range(n):
            lines.append('User: '+ rng.choice(UNKNOWN))
            if i < n-1:
                lines.append('Assistant: '+rng.choice([ask, 'Could you choose one option?', 'Do you have a preference yet?']))
        if rng.choice([True,False]):
            lines.append('User: '+value.capitalize()+'.')
            known = True
        else: known = False
        return history_qa(rng, lines, known, value, ask)

    if task == 'yes_context':
        yes = rng.choice([True,False])
        lines = [f'User: {intro}', f'Assistant: Would you like {value}?', f'User: {rng.choice(CONFIRM if yes else REJECT)}']
        return qa(lines, 'object?', value if yes else 'unknown', [2, 3])

    if task == 'ambiguous_yes':
        lines = [f'User: {intro}',f'Assistant: {ask}',f'User: {rng.choice(CONFIRM)}']
        return qa(lines, 'object?', 'unknown', [2, 3])

    if task == 'revision':
        other = rng.choice([v for v in values if v != value])
        lines = [f'User: {intro}',f'Assistant: {ask}',f'User: {value.capitalize()}.',
                 f'Assistant: Okay, {value}.', 'User: '+ rng.choice(REVISIONS).format(value=other.capitalize())]
        return qa(lines, 'object?', other, [3, 5])

    if task == 'decline':
        lines = [f'User: {intro}',f'Assistant: {ask}',f'User: {rng.choice(DECLINED)}']
        return qa(lines, 'next action?', 'stop_clarifying', [2, 3])

    if task == 'loop':
        n = rng.randint(2,5)
        lines = [f'User: {intro}', f'Assistant: {ask}']
        for i in range(n):
            lines.append('User: '+ rng.choice(UNKNOWN))
            if i < n-1:
                lines.append('Assistant: ' + ask)
        return qa(lines, 'next action?', 'stop_clarifying', [len(lines)-2,len(lines)])
    raise ValueError(task)


def history_qa(rng, lines, known, value, ask):
    q = rng.choice(['object?', 'certainty?', 'ask about object?'])
    a = {
        'object?': value if known else 'unknown',
        'certainty?': 'known' if known else 'unknown',
        'ask about object?':'none' if known else ask
        }[q]
    return qa(lines, q, a, [2, len(lines)])

TASKS = [
    'roles',
    'resolve',
    'questions',
    'history',
    'yes_context',
    'ambiguous_yes',
    'revision',
    'decline',
    'loop'
    ]

def build(out, counts):
    out.mkdir(parents=True, exist_ok=True)
    for split, n in counts.items():
        rng = random.Random(SEED + {
            'train': 0,
            'valid': 10_000,
            'test': 20_000
            }[split])
        data=[]
        for i in range(n):
            task=TASKS[i % len(TASKS)]
            action=ACTIONS[(i//len(TASKS) + rng.randrange(len(ACTIONS)))%len(ACTIONS)]
            data.append(example(rng, task, action))
        rng.shuffle(data)
        path=out/f'{split}.txt'
        path.write_text(''.join(data),encoding='utf-8')
        assert len(data) == n
        for story in data:
            rows=story.strip().splitlines()
            assert rows[0].startswith('1 ')
            assert len(rows[-1].split('\t'))==3
            for j,row in enumerate(rows,1): assert row.startswith(f'{j} '), row
        print(f'{path}: {n} examples')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'products')

    p.add_argument('--train', type=int, default=900)
    p.add_argument('--valid', type=int, default=180)
    p.add_argument('--test', type=int, default=180)
    a = p.parse_args()

    build(a.out,
        {
        'train': a.train,
        'valid': a.valid,
        'test': a.test
        })
