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
    ('have_breakfast', 'something for breakfast', 'What would you like for breakfast?', ['eggs', 'oatmeal', 'toast', 'pancakes']),
    ('bake', 'something to bake', 'What would you like to bake?', ['bread', 'cookies', 'a cake', 'muffins']),
    ('order_food', 'something to order', 'What would you like to order?', ['pizza', 'sushi', 'a salad', 'noodles']),
    ('draw', 'something to draw', 'What would you like to draw?', ['a portrait', 'a landscape', 'a cat', 'a city street']),
    ('photograph', 'something to photograph', 'What would you like to photograph?', ['a sunset', 'a garden', 'the sea', 'the city']),
    ('write', 'something to write', 'What would you like to write?', ['a story', 'a poem', 'an email', 'a journal entry']),
    ('practice', 'something to practice', 'What would you like to practice?', ['the piano', 'the guitar', 'English', 'drawing']),
    ('garden', 'something to grow', 'What would you like to grow?', ['tomatoes', 'flowers', 'herbs', 'strawberries']),
    ('walk', 'somewhere to walk', 'Where would you like to walk?', ['in the park', 'along the river', 'through the city', 'by the sea']),
    ('walk_seaside', 'to take a walk by the sea', 'Where by the sea would you like to walk?', ['along the beach', 'on the promenade', 'near the harbor', 'by the pier']),
    ('cycle', 'somewhere to cycle', 'Where would you like to cycle?', ['in the park', 'along the coast', 'on a quiet trail', 'around the lake']),
    ('run', 'somewhere to run', 'Where would you like to run?', ['in the park', 'along the beach', 'on a trail', 'around the neighborhood']),
    ('swim', 'somewhere to swim', 'Where would you like to swim?', ['at the indoor pool', 'at the outdoor pool', 'in the sea', 'at the community pool']),
    ('visit_pool', 'to go to a swimming pool', 'Which swimming pool would you like to visit?', ['the indoor pool', 'the outdoor pool', 'the community pool', 'the sports center pool']),
    ('workout', 'to work out at the gym', 'What would you like to do at the gym?', ['strength training', 'cardio', 'yoga', 'a treadmill session']),
    ('visit_gym', 'to go to the gym', 'Which gym would you like to visit?', ['the neighborhood gym', 'the fitness center', 'the sports club', 'the university gym']),
    ('go_cinema', 'to go to the cinema', 'What would you like to see at the cinema?', ['a comedy', 'a thriller', 'a science fiction film', 'an animated film']),
    ('shop', 'to go shopping', 'What are you shopping for?', ['a jacket', 'new shoes', 'groceries', 'a birthday present']),
    ('visit_store', 'a store to visit', 'What kind of store would you like to visit?', ['a bookstore', 'a clothing store', 'an electronics store', 'a grocery store']),
    ('visit_mall', 'to visit a shopping mall', 'Which shopping mall would you like to visit?', ['the city mall', 'the mall near the station', 'the shopping center downtown', 'the outlet mall']),
    ('choose_gift', 'a gift to choose', 'What kind of gift would you like to choose?', ['a book', 'a board game', 'a scarf', 'a pair of headphones']),
    ('plan', 'something to plan', 'What would you like to plan?', ['the weekend', 'a holiday', 'a birthday party', 'a day trip']),
    ('travel', 'somewhere to travel', 'Where would you like to travel?', ['to France', 'to Italy', 'to the mountains', 'to the seaside']),
    ('pack', 'something to pack', 'What would you like to pack?', ['a suitcase', 'a backpack', 'a gym bag', 'a travel bag']),
    ('clean', 'something to clean', 'What would you like to clean?', ['the kitchen', 'my desk', 'the windows', 'my room']),
    ('repair', 'something to repair', 'What would you like to repair?', ['a lamp', 'my bicycle', 'a chair', 'my laptop']),
    ('work_on', 'something to work on', 'What would you like to work on?', ['a project', 'a presentation', 'a report', 'a new feature']),
    ('code', 'something to code', 'What would you like to code?', ['a Python script', 'a chatbot', 'a website', 'a small game']),
    ('debug', 'something to debug', 'What would you like to debug?', ['a Python script', 'an API', 'a training loop', 'a web application']),
    ('train_model', 'something to train', 'What would you like to train?', ['a language model', 'an image classifier', 'a small chatbot', 'a recommendation model']),
    ('prepare_dataset', 'a dataset to prepare', 'What kind of dataset would you like to prepare?', ['a dialogue dataset', 'a sentiment dataset', 'a question-answering dataset', 'a text corpus']),
    ('play_video_games', 'a video game to play', 'What video game would you like to play?', ['a racing game', 'a role-playing game', 'a strategy game', 'a puzzle game']),
    ('play_board_games', 'a board game to play', 'What board game would you like to play?', ['chess', 'Scrabble', 'Catan', 'a card game']),
    ('meet', 'someone to meet', 'Who would you like to meet?', ['a friend', 'my family', 'a colleague', 'an old classmate']),
    ('discuss', 'something to talk about', 'What would you like to talk about?', ['the future', 'an idea', 'a difficult decision', 'a memory']),
    ('relax', 'something to do to relax', 'How would you like to relax?', ['with music', 'with a book', 'on a walk', 'with a cup of tea']),
    ('work', 'some work to do', 'What are you working on?', ['a report', 'a project proposal', 'a presentation', 'a client task']),
    ('program', 'something to program', 'What are you programming?', ['a Python application', 'a website', 'an AI assistant', 'a game']),
    ('work_office', 'some work to do at the office', 'What are you working on at the office?', ['a project', 'a report', 'a meeting agenda', 'a presentation']),
    ('program_office', 'something to program at the office', 'What are you programming at the office?', ['a Python script', 'a web service', 'a chatbot', 'an internal tool']),
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
    'have_breakfast': ['Are you making breakfast at home?', 'That sounds like a nice start to the day.'],
    'bake': ['Have you made it before?', 'The kitchen will smell wonderful.'],
    'order_food': ['Would you like delivery or takeaway?', 'That sounds like a good meal.'],
    'draw': ['Will you use a pencil or paints?', 'That could be a fun creative project.'],
    'photograph': ['Are you taking pictures today?', 'That sounds like a lovely subject.'],
    'write': ['Have you started a draft?', 'That sounds worth writing about.'],
    'practice': ['How long have you been practicing?', 'A little practice each day can help.'],
    'garden': ['Will you grow it indoors or outside?', 'That should make your garden more lively.'],
    'walk': ['Are you heading out now?', 'That sounds like a pleasant walk.'],
    'walk_seaside': ['Do you prefer walking there in the morning or evening?', 'The sea can be a peaceful place for a walk.'],
    'cycle': ['Are you going alone or with someone?', 'That sounds like a nice cycling route.'],
    'run': ['Do you prefer a short run or a longer one?', 'That sounds like a good way to clear your head.'],
    'swim': ['Do you usually swim laps or just relax in the water?', 'Swimming sounds refreshing.'],
    'visit_pool': ['Would you like to swim laps or just relax?', 'That sounds like a good place for a swim.'],
    'workout': ['Do you have a workout planned?', 'That sounds like a good session.'],
    'visit_gym': ['Are you planning to work out today?', 'That sounds like a convenient place to exercise.'],
    'go_cinema': ['Would you prefer an afternoon or evening showing?', 'That sounds like a fun cinema trip.'],
    'shop': ['Do you have a budget in mind?', 'Would you rather browse online or in a store?', 'That sounds like a useful shopping trip.'],
    'visit_store': ['Are you going there to browse or buy something?', 'That sounds like a good place to look around.'],
    'visit_mall': ['Are you going to shop or just look around?', 'That sounds like a nice place to spend an afternoon.'],
    'choose_gift': ['Who is the gift for?', 'That sounds like a thoughtful idea.'],
    'plan': ['Have you picked a date yet?', 'That sounds worth planning ahead.'],
    'travel': ['How long would you like to stay?', 'That sounds like an exciting destination.'],
    'pack': ['When are you leaving?', 'It helps to make a packing list.'],
    'clean': ['Will you do it today?', 'It will feel good to have it done.'],
    'repair': ['Do you have the tools you need?', 'Hopefully it is a simple fix.'],
    'work_on': ['What is the next step?', 'That sounds like a productive project.'],
    'code': ['Are you starting from scratch?', 'That sounds like an interesting coding task.'],
    'debug': ['Have you found where it fails?', 'Hope you track down the bug.'],
    'train_model': ['Are you using a new dataset?', 'That sounds like an interesting ML experiment.'],
    'prepare_dataset': ['Are you collecting data or cleaning it?', 'A good dataset can make a big difference.'],
    'play_video_games': ['Do you prefer playing solo or with friends?', 'That sounds like a fun game to try.'],
    'play_board_games': ['Who will you play with?', 'That sounds like a fun game night.'],
    'meet': ['Have you arranged a time?', 'That sounds like a nice chance to catch up.'],
    'discuss': ['Would you like to start with what is on your mind?', 'We can talk it through.'],
    'relax': ['Would you rather be alone or have some company?', 'That sounds like a relaxing choice.'],
    'work': ['Are you on a tight deadline?', 'That sounds like an important task.'],
    'program': ['Which programming language are you using?', 'That sounds like an interesting coding project.'],
    'work_office': ['Is anyone helping you with it?', 'Hope the office work goes smoothly.'],
    'program_office': ['Are you working with a team?', 'That sounds like a productive day at the office.'],
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
    'loop',
    'advice'
    ]


def intro_text(rng, action, value=None):
    verb, thing, _, _ = action
    if value is None:
        if verb == 'work_office':
            return rng.choice(['I have some work to do at the office.', "I'd like to work at the office."])
        if verb == 'program_office':
            return rng.choice(['I want to do some programming at the office.', "I'd like to code at the office."])
        return rng.choice(PROMPTS).format(thing=thing)
    KNOWN_INTROS = {
        'have_breakfast': ['I want {value} for breakfast.', "I'd like {value} for breakfast."],
        'bake': ['I want to bake {value}.', "I'd like to bake {value}."],
        'order_food': ['I want to order {value}.', "I'd like to order {value}."],
        'draw': ['I want to draw {value}.', "I'd like to draw {value}."],
        'photograph': ['I want to photograph {value}.', "I'd like to photograph {value}."],
        'write': ['I want to write {value}.', "I'd like to write {value}."],
        'practice': ['I want to practice {value}.', "I'd like to practice {value}."],
        'garden': ['I want to grow {value}.', "I'd like to grow {value}."],
        'walk': ['I want to walk {value}.', "I'd like to walk {value}."],
        'walk_seaside': ['I want to walk {value}.', "I'd like to walk {value}."],
        'cycle': ['I want to cycle {value}.', "I'd like to cycle {value}."],
        'run': ['I want to run {value}.', "I'd like to run {value}."],
        'swim': ['I want to swim {value}.', "I'd like to swim {value}."],
        'visit_pool': ['I want to visit {value}.', "I'd like to visit {value}."],
        'workout': ['I want to do {value} at the gym.', "I'd like to do {value} at the gym."],
        'visit_gym': ['I want to go to {value}.', "I'd like to go to {value}."],
        'go_cinema': ['I want to see {value} at the cinema.', "I'd like to see {value} at the cinema."],
        'shop': ["I'm shopping for {value}.", "I'd like to shop for {value}."],
        'visit_store': ['I want to visit {value}.', "I'd like to visit {value}."],
        'visit_mall': ['I want to go to {value}.', "I'd like to go to {value}."],
        'choose_gift': ['I want to choose {value} as a gift.', "I'd like to choose {value} as a gift."],
        'plan': ['I want to plan {value}.', "I'd like to plan {value}."],
        'travel': ['I want to travel {value}.', "I'd like to travel {value}."],
        'pack': ['I want to pack {value}.', "I'd like to pack {value}."],
        'clean': ['I want to clean {value}.', "I'd like to clean {value}."],
        'repair': ['I want to repair {value}.', "I'd like to repair {value}."],
        'work_on': ['I want to work on {value}.', "I'd like to work on {value}."],
        'code': ['I want to code {value}.', "I'd like to code {value}."],
        'debug': ['I want to debug {value}.', "I'd like to debug {value}."],
        'train_model': ['I want to train {value}.', "I'd like to train {value}."],
        'prepare_dataset': ['I want to prepare {value}.', "I'd like to prepare {value}."],
        'play_video_games': ['I want to play {value}.', "I'd like to play {value}."],
        'play_board_games': ['I want to play {value}.', "I'd like to play {value}."],
        'meet': ['I want to meet {value}.', "I'd like to meet {value}."],
        'discuss': ['I want to talk about {value}.', "I'd like to talk about {value}."],
        'relax': ['I want to relax {value}.', "I'd like to relax {value}."],
        'work': ['I want to work on {value}.', "I'd like to work on {value}."],
        'program': ['I want to program {value}.', "I'd like to program {value}."],
        'work_office': ['I want to work on {value} at the office.', "I'd like to work on {value} at the office."],
        'program_office': ['I want to program {value} at the office.', "I'd like to program {value} at the office."],
    }
    if verb in KNOWN_INTROS:
        return rng.choice(KNOWN_INTROS[verb]).format(value=value)
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
    if task == 'advice':
        return rng.choice([render_weather, render_tired, render_sleepy])(rng)
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
        if verb in ('work_office', 'program_office'):
            question('location?', 'office', [utterance_line] if turn_index == 0 else [1, utterance_line])

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



WEATHER = ['sunny', 'rainy', 'cloudy', 'windy', 'snowy']
TIMES_OF_DAY = ['morning', 'afternoon', 'evening', 'night']
SEASONS = ['spring', 'summer', 'autumn', 'winter']
WEATHER_QUESTIONS = {
    'weather': ["What's the weather like?", 'Is it sunny, rainy, or cloudy?', 'Would you describe the weather as windy or calm?'],
    'time_of_day': ['What time of day is it?', 'Is it morning, afternoon, evening, or night?', 'Would you say it is daytime or nighttime?'],
    'season': ['What season is it?', 'Is it spring, summer, autumn, or winter?', 'Which season are you in right now?']
}


def clothing_advice(weather, season, time_of_day):
    if weather == 'rainy':
        return 'A waterproof jacket and shoes would be useful.'
    if weather == 'snowy':
        return 'Wear a warm coat, insulated shoes, gloves, and a hat.'
    if season == 'winter':
        return 'A warm coat and layers would be a good idea.'
    if weather == 'windy':
        return 'A windproof jacket and a comfortable extra layer would help.'
    if weather == 'sunny' and season == 'summer':
        return 'Light clothes, a hat, and sunscreen would be sensible.'
    if time_of_day == 'night':
        return 'Bring an extra layer in case it gets cool at night.'
    return 'Comfortable layers should work; check the temperature before heading out.'


def render_weather(rng):
    # Weather, time of day, and season are separate semantic slots.
    # Each missing answer is followed by a relevant question; once known,
    # the model receives a contextual clothing suggestion.
    values = {
        'weather': rng.choice(WEATHER),
        'time_of_day': rng.choice(TIMES_OF_DAY),
        'season': rng.choice(SEASONS)
    }
    known = {}
    refs = {}
    rows = []

    def fact(line):
        rows.append(line)
        return len(rows)

    def question(label, answer, supports):
        supports = sorted(set(supports))
        assert supports and all(1 <= r <= len(rows) for r in supports)
        return fact(f'{label}\t{answer}\t{" ".join(map(str, supports))}')

    intro = fact('User: I am thinking of going outside.')
    question('actor?', 'user', [intro])
    question('action?', 'go_outside', [intro])
    question('intention?', 'go_outside', [intro])
    question('location?', 'outside', [intro])
    keys = ['weather', 'time_of_day', 'season']
    rng.shuffle(keys)
    for key in keys:
        question(f'current {key}?', 'unknown', [intro])
        question('next action?', 'clarify', [intro])
        question('assistant response?', WEATHER_QUESTIONS[key][0], [intro])
        assistant_line = fact('Assistant: ' + WEATHER_QUESTIONS[key][0])
        uncertain = rng.randrange(3)
        for attempt in range(uncertain):
            user_line = fact('User: ' + rng.choice(UNKNOWN))
            question(f'{key}?', 'unknown', [assistant_line, user_line])
            question('next action?', 'clarify', [assistant_line, user_line])
            follow_up = WEATHER_QUESTIONS[key][attempt + 1]
            question('assistant response?', follow_up, [assistant_line, user_line])
            assistant_line = fact('Assistant: ' + follow_up)
        answer = values[key]
        user_line = fact('User: It is ' + answer + '.' if key != 'season' else 'User: It is ' + answer + '.')
        refs[key] = user_line
        known[key] = answer
        question(f'{key}?', answer, [assistant_line, user_line])
        question(f'current {key}?', answer, [user_line])
        question(f'is {key} known?', 'yes', [user_line])
    question('current weather?', known['weather'], [refs['weather']])
    question('current time_of_day?', known['time_of_day'], [refs['time_of_day']])
    question('current season?', known['season'], [refs['season']])
    question('next action?', 'advance', list(refs.values()))
    question('assistant response?', clothing_advice(known['weather'], known['season'], known['time_of_day']), list(refs.values()))
    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'



REST_HOURS = [2, 3, 4, 5, 6, 8, 10, 12]
USUAL_REST_HOURS = [1, 2, 3, 4]
SLEEP_HOURS = [4, 5, 6, 7, 8, 9]
AWAKE_HOURS = [8, 12, 16, 20, 24, 30]
WELLBEING_QUESTIONS = {
    'tired': {
        'since': ['How many hours has it been since you last rested?', 'Has it been a few hours or most of the day since your last break?', 'About how long have you gone without a proper rest?'],
        'last': ['How many hours did you rest last time?', 'Was your last rest closer to one hour or several?', 'Roughly how many hours did your last break last?'],
        'usual': ['How many hours do you usually rest?', 'Do you usually take a short rest or a few hours?', 'About how many hours of rest is normal for you?']
    },
    'sleepy': {
        'since': ['How many hours have you been awake?', 'Have you been awake for just a few hours or much longer?', 'Roughly how long has it been since you last slept?'],
        'last': ['How many hours did you sleep last time?', 'Was your last sleep short or close to a full night?', 'Roughly how many hours of sleep did you get last time?'],
        'usual': ['How many hours do you usually sleep?', 'Is your usual sleep closer to six or eight hours?', 'About how much sleep do you normally get?']
    }
}


def wellbeing_advice(kind, values):
    if kind == 'sleepy':
        if values['since'] >= 20:
            return 'You have been awake a long time. Try to get some sleep soon, and avoid driving while sleepy.'
        if values['last'] < values['usual']:
            return 'You slept less than usual. A chance to rest and an earlier bedtime might help.'
        return 'If you are sleepy, consider taking a break and making time for sleep.'
    if values['since'] >= 8:
        return 'It sounds like you have gone a while without a break. Consider resting and having some water.'
    if values['last'] < values['usual']:
        return 'Your last rest was shorter than usual. A quiet break may help you recharge.'
    return 'You sound tired. A comfortable break and a little downtime may help.'


def render_wellbeing(rng, kind):
    values = {
        'since': rng.choice(AWAKE_HOURS if kind == 'sleepy' else REST_HOURS),
        'last': rng.choice(SLEEP_HOURS if kind == 'sleepy' else USUAL_REST_HOURS),
        'usual': rng.choice(SLEEP_HOURS if kind == 'sleepy' else USUAL_REST_HOURS)
    }
    rows = []
    refs = {}
    label = 'sleep' if kind == 'sleepy' else 'rest'

    def fact(line):
        rows.append(line)
        return len(rows)

    def question(name, answer, supports):
        supports = sorted(set(supports))
        assert supports and all(1 <= r <= len(rows) for r in supports)
        return fact(f'{name}\t{answer}\t{" ".join(map(str, supports))}')

    opening = rng.choice(["I'm tired.", 'I feel exhausted.', 'I need some rest.'] if kind == 'tired' else ["I'm sleepy.", 'I really want to sleep.', 'I can barely keep my eyes open.'])
    intro = fact('User: ' + opening)
    question('actor?', 'user', [intro])
    question('action?', label, [intro])
    question('intention?', label, [intro])
    question('object?', 'unknown', [intro])
    keys = ['since', 'last', 'usual']
    rng.shuffle(keys)
    for key in keys:
        question(f'current {key}_{label}_hours?', 'unknown', [intro])
        question('next action?', 'clarify', [intro])
        initial_question = WELLBEING_QUESTIONS[kind][key][0]
        question('assistant response?', initial_question, [intro])
        assistant = fact('Assistant: ' + initial_question)
        for attempt in range(rng.randrange(3)):
            user = fact('User: ' + rng.choice(UNKNOWN))
            question(f'{key}_{label}_hours?', 'unknown', [assistant, user])
            question('next action?', 'clarify', [assistant, user])
            follow_up = WELLBEING_QUESTIONS[kind][key][attempt + 1]
            question('assistant response?', follow_up, [assistant, user])
            assistant = fact('Assistant: ' + follow_up)
        answer = values[key]
        user = fact(f'User: {answer} hours.')
        refs[key] = user
        question(f'{key}_{label}_hours?', str(answer), [assistant, user])
        question(f'current {key}_{label}_hours?', str(answer), [user])
        question(f'is {key}_{label}_hours known?', 'yes', [user])
    for key in keys:
        question(f'current {key}_{label}_hours?', str(values[key]), [refs[key]])
    question('next action?', 'advance', list(refs.values()))
    question('assistant response?', wellbeing_advice(kind, values), list(refs.values()))
    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'


def render_tired(rng):
    return render_wellbeing(rng, 'tired')


def render_sleepy(rng):
    return render_wellbeing(rng, 'sleepy')

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
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--valid', type=int, default=300)
    parser.add_argument('--test', type=int, default=300)

    args = parser.parse_args()
    build(args.out, {
        'train': args.train,
        'valid': args.valid,
        'test': args.test
        })
