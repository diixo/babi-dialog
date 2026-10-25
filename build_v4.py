#!/usr/bin/env python3
"""Build bAbI-style semantic dialogue stories. Text only; each QA is tab-separated.

For each user utterance: actor/action/intention/object extraction questions first,
then context-dependent dialogue questions. Supporting indices refer to prior lines.

Version 4: independent object collections, consistent clarification decisions,
and balanced, reproducible datasets with distinct dialogue inputs across splits.
"""
import argparse
import random
from pathlib import Path

SEED = 42
LOOP_MIN_REPETITIONS = 2
MAX_GENERATION_ATTEMPTS = 1000
DEFAULT_COUNTS = {'train': 13000, 'valid': 1300, 'test': 1300}
STOP_DECLINED = "That's okay. We can talk about something else."
CLARIFICATION_PREFIXES = ['', 'Take your time. ', 'You can give a short answer. ']
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
INTRO_FRAMES = [
    'I want to {activity}.', "I'd like to {activity}.",
    'I would like to {activity}.', 'I plan to {activity}.',
    "I'm planning to {activity}.", 'I intend to {activity}.',
    "I'm hoping to {activity}.", 'My plan is to {activity}.',
    "I've decided to {activity}.", 'I would really like to {activity}.',
    'I feel ready to {activity}.', 'I would prefer to {activity}.',
]
INTRO_TIMES = ['', ' today', ' tomorrow', ' this weekend', ' soon', ' when I have time']
# Explicit activity phrases keep the action grounded even when the object is unknown.
ACTIVITY_PHRASES = {
    'drink': ('drink something', 'drink {value}'),
    'eat': ('eat something', 'eat {value}'),
    'read': ('read something', 'read {value}'),
    'watch': ('watch something', 'watch {value}'),
    'listen': ('listen to something', 'listen to {value}'),
    'buy': ('buy something', 'buy {value}'),
    'cook': ('cook something', 'cook {value}'),
    'play': ('play something', 'play {value}'),
    'visit': ('visit somewhere', 'visit {value}'),
    'learn': ('learn something', 'learn {value}'),
    'have_breakfast': ('have something for breakfast', 'have {value} for breakfast'),
    'bake': ('bake something', 'bake {value}'),
    'order_food': ('order some food', 'order {value}'),
    'draw': ('draw something', 'draw {value}'),
    'photograph': ('photograph something', 'photograph {value}'),
    'write': ('write something', 'write {value}'),
    'practice': ('practice something', 'practice {value}'),
    'garden': ('grow something', 'grow {value}'),
    'walk': ('go for a walk somewhere', 'walk {value}'),
    'walk_seaside': ('take a walk by the sea', 'take a seaside walk {value}'),
    'cycle': ('cycle somewhere', 'cycle {value}'),
    'run': ('run somewhere', 'run {value}'),
    'swim': ('swim somewhere', 'swim {value}'),
    'visit_pool': ('visit a swimming pool', 'visit {value}'),
    'workout': ('work out at the gym', 'do {value} at the gym'),
    'visit_gym': ('go to a gym', 'go to {value}'),
    'go_cinema': ('see something at the cinema', 'see {value} at the cinema'),
    'shop': ('go shopping for something', 'shop for {value}'),
    'visit_store': ('visit a store', 'visit {value}'),
    'visit_mall': ('visit a shopping mall', 'visit {value}'),
    'choose_gift': ('choose a gift', 'choose {value} as a gift'),
    'plan': ('plan something', 'plan {value}'),
    'travel': ('travel somewhere', 'travel {value}'),
    'pack': ('pack something', 'pack {value}'),
    'clean': ('clean something', 'clean {value}'),
    'repair': ('repair something', 'repair {value}'),
    'work_on': ('work on something', 'work on {value}'),
    'code': ('code something', 'code {value}'),
    'debug': ('debug something', 'debug {value}'),
    'train_model': ('train a model', 'train {value}'),
    'prepare_dataset': ('prepare a dataset', 'prepare {value}'),
    'play_video_games': ('play a video game', 'play {value}'),
    'play_board_games': ('play a board game', 'play {value}'),
    'meet': ('meet someone', 'meet {value}'),
    'discuss': ('talk about something', 'talk about {value}'),
    'relax': ('do something to relax', 'relax {value}'),
    'work': ('do some work', 'do some work on {value}'),
    'program': ('program something', 'program {value}'),
    'work_office': ('work on something at the office', 'work on {value} at the office'),
    'program_office': ('program something at the office', 'program {value} at the office'),
}
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
    'advice',
    'emotion',
    'object_tracking',
    'multi_action'
    ]


def intro_text(rng, action, value=None):
    unknown, known = ACTIVITY_PHRASES[action[0]]
    activity = unknown if value is None else known.format(value=value)
    return rng.choice(INTRO_FRAMES).format(activity=activity + rng.choice(INTRO_TIMES))


def clarification_question(rng, question, attempt):
    return (rng.choice(CLARIFICATION_PREFIXES) if attempt else '') + question


def make_dialogue(rng, task, action):
    verb, thing, ask, values = action
    value = rng.choice(values)
    initial_known = task in ('roles', 'questions') and rng.choice([True, False])
    intro = intro_text(rng, action, value if initial_known else None)
    # Each User turn stores information explicitly supplied by that turn.
    turns = [('User', intro, value if initial_known else None)]
    if task in ('roles', 'questions'):
        return turns
    turns.append(('Assistant', ask, None))
    if task == 'resolution':
        known = rng.choice([True, False])
        turns.append(('User', (value.capitalize() + '.') if known else rng.choice(UNKNOWN), value if known else None))
    elif task in ('history', 'loop'):
        attempts = rng.randint(1, 4) if task == 'history' else rng.randint(2, 5)
        for i in range(attempts):
            turns.append(('User', rng.choice(UNKNOWN), None))
            if i + 1 < attempts:
                turns.append(('Assistant', clarification_question(rng, ask, i + 1), None))
        # A finite training history may end at a clarification target. Complete
        # loop stories continue until the user supplies a value or declines.
        if task == 'loop' or rng.choice([True, False]):
            turns.append(('Assistant', clarification_question(rng, ask, attempts), None))
            if rng.randrange(4) == 0:
                turns.append(('User', rng.choice(DECLINED), 'declined'))
            else:
                turns.append(('User', value.capitalize() + '.', value))
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
    return turns


def render(rng, task, action):
    if task == 'advice':
        return rng.choice([render_weather, render_tired, render_sleepy])(rng)
    if task == 'emotion':
        return render_emotion(rng, action)
    if task == 'object_tracking':
        return render_object_tracking(rng)
    if task == 'multi_action':
        return render_multi_action(rng)
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
        elif observed is not None:
            state = observed
            state_source = utterance_line
            uncertain_streak = 0
            uncertainty_sources.clear()
        elif last_assistant_line is not None and state is None:
            # The opening request is not a failed clarification answer.
            uncertain_streak += 1
            uncertainty_sources.extend([last_assistant_line, utterance_line])

        # All extraction questions about this one utterance FIRST.
        question('actor?', 'user', [utterance_line])
        context_support = [1, utterance_line]
        if last_assistant_line is not None:
            context_support.append(last_assistant_line)
        question('action?', verb, context_support)
        question('intention?', verb, context_support)
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
        question('missing?', 'object' if effective_object == 'unknown' else 'none', support)
        question('declined?', 'object' if state == 'declined' else 'none', support)
        repeated = state is None and uncertain_streak >= LOOP_MIN_REPETITIONS
        question('loop?', 'yes' if repeated else 'no', support + uncertainty_sources)
        # The same visible history must lead to the same decision in every task.
        # Repeated uncertainty does not waive a required fact.
        if state == 'declined':
            next_action = 'stop_clarify'
            next_question = 'none'
        elif effective_object == 'unknown':
            next_action = 'clarify'
            next_question = ask
        else:
            next_action = 'advance'
            next_question = 'none'
        if next_action == 'clarify':
            response = clarification_question(rng, ask, uncertain_streak)
        elif next_action == 'stop_clarify':
            response = STOP_DECLINED
        else:
            response = rng.choice(ADVANCE[verb]).format(object=effective_object)
        # A script may rephrase a question, but cannot override a stop decision.
        if turn_index + 1 < len(turns) and turns[turn_index + 1][0] == 'Assistant':
            if next_action != 'stop_clarify':
                response = turns[turn_index + 1][1]
            turns[turn_index + 1] = ('Assistant', response, None)
        if next_action == 'clarify':
            next_question = response
        decision_support = context_support + support + uncertainty_sources
        next_action_line = question('next action?', next_action, decision_support)
        next_question_line = question('ask about object?', next_question, decision_support)
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


def render_fact_collection(rng, opening, roles, slots, advice):
    """Ask every required slot until answered or explicitly declined.

    Unknown answers keep the current slot active. A refusal waives only that
    slot, leaving its value unknown, and questioning continues for other slots.
    """
    rows = []
    keys = list(slots)
    rng.shuffle(keys)
    initial = rng.sample(keys, rng.randrange(len(keys) + 1))
    known = {key: slots[key]['value'] for key in initial}
    declined = set()
    attempts = dict.fromkeys(keys, 0)
    uncertainty_sources = {key: [] for key in keys}

    def fact(text):
        rows.append(text)
        return len(rows)

    def question(label, answer, refs):
        refs = sorted(set(refs))
        assert refs and all(1 <= ref <= len(rows) for ref in refs)
        return fact(f'{label}\t{answer}\t{" ".join(map(str, refs))}')

    details = [slots[key]['statement'].format(value=known[key]) for key in initial]
    intro = fact('User: ' + ' '.join([opening] + details))
    sources = dict.fromkeys(keys, intro)
    for role, value in roles.items():
        question(role + '?', value, [intro])

    def snapshot(user, refused_now=False):
        state_refs = []
        for key in keys:
            current = question(f'current {key}?', str(known.get(key, 'unknown')), [sources[key]])
            state_refs.append(current)
            question(f'is {key} known?', 'yes' if key in known else 'no', [current])
        missing = [key for key in keys if key not in known]
        pending = [key for key in missing if key not in declined]
        missing_line = question('missing?', ', '.join(missing) or 'none', state_refs)
        declined_line = question('declined?', ', '.join(key for key in keys if key in declined) or 'none',
                                 [intro] + [sources[key] for key in declined])
        support = [user, missing_line, declined_line]
        key = pending[0] if pending else None
        question('next fact?', key or 'none', support)
        repeated = key is not None and attempts[key] >= LOOP_MIN_REPETITIONS
        question('loop?', 'yes' if repeated else 'no',
                 support + (uncertainty_sources[key] if key else []))
        if key is not None:
            variants = slots[key]['questions']
            ask = variants[min(attempts[key], len(variants) - 1)]
            response = clarification_question(rng, ask, attempts[key])
            if refused_now:
                response = 'Okay, we can leave that aside. ' + response
            next_action = 'clarify'
        elif missing:
            next_action = 'stop_clarify'
            response = ('Okay, we can leave those questions aside. '
                        'I do not have all the details needed for a specific suggestion.')
        else:
            next_action = 'advance'
            response = advice(known)
        decision = question('next action?', next_action, support)
        ask_line = question('ask about fact?', response if key is not None else 'none', [decision] + support)
        question('assistant response?', response, [decision, ask_line] + state_refs + support)
        return key, response

    key, response = snapshot(intro)
    while key is not None:
        current_key = key
        uncertain = rng.randrange(5)
        for attempt in range(uncertain + 1):
            assistant = fact('Assistant: ' + response)
            refused = False
            if attempt < uncertain:
                user = fact('User: ' + rng.choice(UNKNOWN))
                answer = 'unknown'
                attempts[current_key] += 1
                uncertainty_sources[current_key].extend([assistant, user])
            elif rng.randrange(5) == 0:
                user = fact('User: ' + rng.choice(DECLINED))
                answer = 'declined'
                declined.add(current_key)
                refused = True
            else:
                value = slots[current_key]['value']
                user = fact('User: ' + slots[current_key]['answer'].format(value=value))
                answer = str(value)
                known[current_key] = value
            sources[current_key] = user
            question(f'{current_key}?', answer, [assistant, user])
            key, response = snapshot(user, refused)
    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'


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
    values = {
        'weather': rng.choice(WEATHER),
        'time_of_day': rng.choice(TIMES_OF_DAY),
        'season': rng.choice(SEASONS)
    }
    statements = {'weather': 'The weather is {value}.',
                  'time_of_day': 'It is {value}.', 'season': 'The season is {value}.'}
    slots = {key: {'value': value, 'questions': WEATHER_QUESTIONS[key],
                   'statement': statements[key], 'answer': 'It is {value}.'}
             for key, value in values.items()}
    return render_fact_collection(
        rng, 'I am thinking of going outside.',
        {'actor': 'user', 'action': 'go_outside', 'intention': 'go_outside', 'location': 'outside'},
        slots, lambda known: clothing_advice(known['weather'], known['season'], known['time_of_day']))



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
    label = 'sleep' if kind == 'sleepy' else 'rest'
    opening = rng.choice(["I'm tired.", 'I feel exhausted.', 'I need some rest.'] if kind == 'tired' else ["I'm sleepy.", 'I really want to sleep.', 'I can barely keep my eyes open.'])
    statements = {
        'since': ('I have been awake for {value} hours.' if kind == 'sleepy'
                  else 'It has been {value} hours since my last rest.'),
        'last': ('I slept for {value} hours last time.' if kind == 'sleepy'
                 else 'I rested for {value} hours last time.'),
        'usual': ('I usually sleep for {value} hours.' if kind == 'sleepy'
                  else 'I usually rest for {value} hours.'),
    }
    slots = {f'{key}_{label}_hours': {
        'value': value, 'questions': WELLBEING_QUESTIONS[kind][key],
        'statement': statements[key], 'answer': '{value} hours.'}
        for key, value in values.items()}
    return render_fact_collection(
        rng, opening, {'actor': 'user', 'action': label, 'intention': label}, slots,
        lambda known: wellbeing_advice(kind, {key: known[f'{key}_{label}_hours'] for key in values}))


def render_tired(rng):
    return render_wellbeing(rng, 'tired')


def render_sleepy(rng):
    return render_wellbeing(rng, 'sleepy')

EMOTIONS = {
    'neutral': {
        'answers': ['Neutral.', 'I feel okay.', 'Nothing special.', 'Just normal.'],
        'questions': ['Anything on your mind today?', 'What would you like to talk about?', 'How has your day been?']
        },
    'joy': {
        'answers': ['Happy.', 'I feel great.', 'Really cheerful.', 'Pretty happy today.'],
        'questions': ["What's making you happy today?", 'Did something good happen?', 'What has been the best part of your day?']
        },
    'surprise': {
        'answers': ['Surprised.', 'Something caught me off guard.', "I'm amazed.", 'Really surprised.'],
        'questions': ['What surprised you?', 'Was it a pleasant surprise?', 'What happened that you did not expect?']
        },
    'sadness': {
        'answers': ['Sad.', 'I feel down.', 'A bit sad.', 'Feeling gloomy.'],
        'questions': ['Would you like to talk about it?', 'Has something been bothering you?', 'Would some company help right now?']
        },
    'anger': {
        'answers': ['Angry.', 'I feel frustrated.', 'Really annoyed.', 'I am upset.'],
        'questions': ['Would you like to tell me what upset you?', 'Has something frustrating happened?', 'Would you rather vent or take your mind off it?']
        },
    'disgust': {
        'answers': ['Disgusted.', 'That was gross.', 'I feel disgusted.', 'Something really put me off.'],
        'questions': ['Did something unpleasant happen?', 'Was there something that put you off?', 'Would you rather change the subject?']
        },
    'fear': {
        'answers': ['Afraid.', 'I feel scared.', 'A little frightened.', "I'm worried and scared."],
        'questions': ['Is something worrying you?', 'Would you like to tell me what happened?', 'Would you prefer to talk it through or switch topics?']
        }
    }
EMOTION_PROMPTS = [
    'How are you feeling today: neutral, happy, surprised, sad, angry, disgusted, or afraid?',
    'Would you say you feel neutral, joyful, surprised, sad, angry, disgusted, or afraid?',
    'Which fits your mood best: neutral, happy, surprised, sad, angry, disgusted, or scared?'
    ]
EMOTION_FOLLOW_UP = [
    'No pressure to label it. Does it feel more positive, negative, or neutral?',
    'Take your time. How would you describe your mood?',
    'How are you feeling at the moment?'
    ]
EMOTION_CONTEXTS = {
    'work': ["I have been working all day.", 'Work has been on my mind.'],
    'program': ["I've been programming today.", 'I have been coding for a while.'],
    'play_video_games': ["I've been playing video games.", 'I spent some time gaming.'],
    'walk_seaside': ["I just came back from a walk by the sea.", 'I was walking along the coast.'],
    'relax': ["I am trying to relax.", 'I want to take it easy for a while.']
    }
EMOTION_CONTEXT_QUESTIONS = {
    'work': ['Has work affected your mood today?', 'Did something happen at work?'],
    'program': ['Did something happen while you were coding?', 'Has your project affected your mood?'],
    'play_video_games': ['Was it something that happened in the game?', 'Did gaming put you in that mood?'],
    'walk_seaside': ['Did the walk affect how you feel?', 'Did something happen on your walk?'],
    'relax': ['Has it been hard to unwind?', 'Is something on your mind while you rest?']
    }


def render_emotion(rng, action):
    rows = []

    def fact(line):
        rows.append(line)
        return len(rows)

    def question(label, answer, refs):
        refs = sorted(set(refs))
        assert refs and all(1 <= r <= len(rows) for r in refs)
        return fact(f'{label}\t{answer}\t{" ".join(map(str, refs))}')

    context = rng.choice([None, None, 'work', 'program', 'play_video_games', 'walk_seaside', 'relax'])
    opening = rng.choice(['I just want to chat.', 'I feel like talking.', 'Can we talk for a bit?']) if context is None else rng.choice(EMOTION_CONTEXTS[context])
    intro = fact('User: ' + opening)
    question('actor?', 'user', [intro])
    question('action?', 'chat' if context is None else context, [intro])
    question('intention?', 'chat', [intro])
    question('emotion?', 'unknown', [intro])
    question('current emotion?', 'unknown', [intro])
    question('missing?', 'emotion', [intro])
    question('declined?', 'none', [intro])
    question('next action?', 'clarify', [intro])
    opening_question = rng.choice(EMOTION_PROMPTS)
    question('assistant response?', opening_question, [intro])
    assistant = fact('Assistant: ' + opening_question)

    # Unknown mood remains pending until an answer or an explicit refusal.
    if rng.randrange(4) == 0:
        for attempt in range(rng.randint(1, 5)):
            user = fact('User: ' + rng.choice(UNKNOWN))
            question('actor?', 'user', [user])
            question('emotion?', 'unknown', [user])
            question('current emotion?', 'unknown', [intro, user])
            question('missing?', 'emotion', [intro, user])
            question('declined?', 'none', [user])
            question('next action?', 'clarify', [assistant, user])
            followup = rng.choice(EMOTION_FOLLOW_UP)
            question('assistant response?', followup, [assistant, user])
            assistant = fact('Assistant: ' + followup)

    if rng.randrange(5) == 0:
        user = fact('User: ' + rng.choice(DECLINED))
        question('actor?', 'user', [user])
        question('current emotion?', 'unknown', [intro, user])
        question('missing?', 'emotion', [intro, user])
        question('declined?', 'emotion', [assistant, user])
        question('next action?', 'stop_clarify', [assistant, user])
        question('assistant response?', 'Of course. We can talk about something else.', [assistant, user])
        return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'

    emotion = rng.choice(list(EMOTIONS))
    user = fact('User: ' + rng.choice(EMOTIONS[emotion]['answers']))
    emotion_source = user
    question('actor?', 'user', [user])
    question('emotion?', emotion, [user])
    question('current emotion?', emotion, [user])
    question('is emotion known?', 'yes', [user])
    question('missing?', 'none', [user])
    question('declined?', 'none', [user])
    question('next action?', 'advance', [assistant, user])
    next_question = rng.choice(EMOTIONS[emotion]['questions'])
    question('assistant response?', next_question, [user])
    assistant = fact('Assistant: ' + next_question)

    # Respect user-initiated topic changes or a refusal to discuss feelings.
    if rng.randrange(5) == 0:
        user = fact('User: ' + rng.choice(["I'd rather not talk about my mood.", "Let's talk about something else.", "I don't want to explain."]))
        question('actor?', 'user', [user])
        question('current emotion?', emotion, [emotion_source, user])
        question('next action?', 'stop_clarify', [assistant, user])
        question('assistant response?', 'Of course. We can talk about something else.', [assistant, user])
    elif context is not None:
        question('context action?', context, [intro])
        question('assistant response?', rng.choice(EMOTION_CONTEXT_QUESTIONS[context]), [intro, user])
    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'



# Object tracking exercises use explicit semantic operations, not a general
# dialogue-state machine. Each action has its own independent object collection.
TRACKING_OBJECTS = {
    'buy': ['apples', 'pears', 'milk', 'bread', 'a book', 'a phone', 'a jacket'],
    'cook': ['pasta', 'rice', 'soup', 'vegetables', 'bread'],
    'read': ['a book', 'a magazine', 'a novel', 'a comic'],
    'play_video_games': ['a racing game', 'a puzzle game', 'a strategy game', 'a role-playing game'],
    'pack': ['a jacket', 'a book', 'a phone', 'a charger'],
    'watch': ['a movie', 'a documentary', 'a comedy', 'a series']
    }
TRACKING_VERBS = {
    'buy': 'buy',
    'cook': 'cook',
    'read': 'read',
    'play_video_games': 'play',
    'pack': 'pack',
    'watch': 'watch'
    }


def tracking_story(rng, multiple):
    actions = rng.sample(list(TRACKING_OBJECTS), 3) if multiple else [rng.choice(list(TRACKING_OBJECTS))]
    if multiple and rng.randrange(3) == 0:
        actions = ['buy', 'read', rng.choice(['cook', 'pack', 'watch'])]
    objects = {action: [] for action in actions}
    sources = {action: [] for action in actions}
    rows = []

    def fact(line):
        rows.append(line)
        return len(rows)

    def question(label, answer, refs):
        refs = sorted(set(refs))
        assert refs and all(1 <= r <= len(rows) for r in refs)
        return fact(f'{label}\t{answer}\t{" ".join(map(str, refs))}')

    def state_answer(action):
        return ', '.join(objects[action]) if objects[action] else 'none'

    def turn(action, utterance, operation, changed, new_objects=None, old_objects=None):
        new_objects = new_objects or []
        old_objects = old_objects or []
        user = fact('User: ' + utterance)
        support = [user] + sources[action][-1:]
        question('actor?', 'user', [user])
        question('action?', action, support)
        question('intention?', action, support)
        question('objects?', ', '.join(changed) if changed else 'none', [user])
        question('operation?', operation, support)
        if operation == 'add':
            for obj in new_objects:
                if obj not in objects[action]:
                    objects[action].append(obj)
        elif operation == 'remove':
            objects[action] = [obj for obj in objects[action] if obj not in old_objects]
        elif operation == 'replace':
            index = objects[action].index(old_objects[0])
            objects[action][index] = new_objects[0]
        elif operation == 'clear':
            objects[action] = []
        # keep and complete preserve the collection unchanged.
        sources[action].append(user)
        current = question(f'current objects for {action}?', state_answer(action), sources[action])
        if operation == 'complete':
            next_action = 'advance'
            response = 'All right, that is everything for now.'
        elif operation == 'clear':
            next_action = 'advance'
            response = 'Okay, I have cleared those plans.'
        elif objects[action]:
            next_action = 'advance'
            response = 'Got it. For ' + action.replace('_', ' ') + ', I have: ' + state_answer(action) + '.'
        else:
            next_action = 'clarify'
            response = 'What would you like to ' + TRACKING_VERBS[action] + '?'
        decision = question('next action?', next_action, [current, user])
        question('assistant response?', response, [current, decision])
        return response

    def emit(action, utterance, operation, changed, new_objects=None, old_objects=None):
        response = turn(action, utterance, operation, changed, new_objects, old_objects)
        fact('Assistant: ' + response)

    # The first user message is always a concrete, grounded addition.
    for index, action in enumerate(actions):
        value = rng.choice(TRACKING_OBJECTS[action])
        if multiple and action in ('buy', 'read') and 'buy' in actions and 'read' in actions:
            value = 'a book'
        utterance = rng.choice([
            f'I want to {TRACKING_VERBS[action]} {value}.',
            f"I'd like to {TRACKING_VERBS[action]} {value}."
            ])
        emit(action, utterance, 'add', [value], [value])

    # Adding two objects in one sentence and then removing one is the core test.
    action = rng.choice(actions)
    additions = rng.sample([value for value in TRACKING_OBJECTS[action] if value not in objects[action]], 2)
    add_text = (f'I also want to {TRACKING_VERBS[action]} {additions[0]} and {additions[1]}.'
                if multiple else f'Also {additions[0]} and {additions[1]}.')
    emit(action, add_text, 'add', additions, additions)
    to_remove = rng.choice(additions)
    remove_text = (f'Actually, remove {to_remove} from my {action.replace("_", " ")} plans.'
                   if multiple else f'Actually, forget {to_remove}.')
    emit(action, remove_text, 'remove', [to_remove], old_objects=[to_remove])

    # Replacements are not removals of the entire set.
    replace_actions = [a for a in actions if objects[a]]
    if multiple and 'buy' in actions and 'read' in actions:
        replace_actions = [a for a in replace_actions if a != 'buy']
    action = rng.choice(replace_actions)
    original = rng.choice(objects[action])
    replacement = rng.choice([value for value in TRACKING_OBJECTS[action] if value not in objects[action]])
    emit(action, f'Change {original} to {replacement}.', 'replace', [original, replacement], [replacement], [original])

    action = rng.choice(actions)
    if rng.choice([True, False]):
        kept = rng.choice(objects[action])
        keep_text = (f'I still want to {TRACKING_VERBS[action]} {kept}.'
                     if multiple else f'I still want {kept}.')
        emit(action, keep_text, 'keep', [kept])
    else:
        emit(action, 'That is everything for now.', 'complete', [])

    if multiple:
        if 'buy' in actions and 'read' in actions:
            emit('buy', 'Actually, I no longer want to buy the book.', 'remove', ['a book'], old_objects=['a book'])
        action = rng.choice(actions)
        emit(action, f'I have cancelled all my {action.replace("_", " ")} plans.', 'clear', [])
        # Query all independent actions after the change, not only the last one.
        for action in actions:
            question(f'final objects for {action}?', state_answer(action), sources[action])
        overview = '; '.join(f'{action}: {state_answer(action)}' for action in actions)
        question('assistant response?', 'Current plans: ' + overview + '.', [sources[a][-1] for a in actions])
    else:
        action = actions[0]
        question(f'final objects for {action}?', state_answer(action), sources[action])

    return '\n'.join(f'{i} {line}' for i, line in enumerate(rows, 1)) + '\n\n'


def render_object_tracking(rng):
    return tracking_story(rng, False)


def render_multi_action(rng):
    return tracking_story(rng, True)


def dialogue_key(story):
    """Deduplicate dialogue inputs even when QA targets or final replies differ."""
    turns = [row.partition(' ')[2] for row in story.strip().splitlines()
             if row.partition(' ')[2].startswith(('User: ', 'Assistant: '))]
    while turns and turns[-1].startswith('Assistant: '):
        turns.pop()
    return '\n'.join(' '.join(turn.split()).casefold() for turn in turns)


def validate_story(story):
    """Check references and the agreement between decisions and assistant turns."""
    rows = story.strip().splitlines()
    if not rows or not rows[0].startswith('1 User: '):
        raise ValueError('A story must start with a user utterance numbered 1')
    next_action = None
    next_question = None
    response = None
    missing = None
    declined = set()
    qa_count = 0
    for j, row in enumerate(rows, 1):
        if not row.startswith(f'{j} '):
            raise ValueError(f'Invalid line numbering: {row}')
        body = row.partition(' ')[2]
        if '\t' not in body:
            if body.startswith('User: '):
                next_action = next_question = response = None
                missing = None
                declined = set()
            elif body.startswith('Assistant: '):
                if response is None or body != 'Assistant: ' + response:
                    raise ValueError(f'Assistant turn disagrees with its QA target: {row}')
            else:
                raise ValueError(f'Unexpected dialogue line: {row}')
            continue
        parts = body.split('\t')
        if len(parts) != 3 or not parts[1] or not parts[2].split():
            raise ValueError(f'Invalid QA line: {row}')
        label, answer, refs = parts
        if not all(k.isdigit() and 1 <= int(k) < j for k in refs.split()):
            raise ValueError(f'Invalid supporting references: {row}')
        qa_count += 1
        if label == 'missing?':
            missing = set() if answer == 'none' else set(answer.split(', '))
        elif label == 'declined?':
            declined = set() if answer == 'none' else set(answer.split(', '))
        elif label == 'next action?':
            if answer not in ('clarify', 'advance', 'stop_clarify'):
                raise ValueError(f'Unknown next action: {row}')
            next_action = answer
            next_question = None
            if missing is not None:
                if answer == 'advance' and missing:
                    raise ValueError(f'Cannot advance with unresolved facts: {row}')
                if answer == 'stop_clarify' and missing - declined:
                    raise ValueError(f'Cannot stop while facts are pending without refusal: {row}')
        elif label in ('ask about object?', 'ask about fact?'):
            next_question = answer
            if next_action == 'clarify' and answer == 'none':
                raise ValueError(f'Clarification requires a question: {row}')
            if next_action in ('advance', 'stop_clarify') and answer != 'none':
                raise ValueError(f'Unexpected object clarification: {row}')
        elif label == 'assistant response?':
            response = answer
            if next_action == 'stop_clarify' and '?' in answer:
                raise ValueError(f'Clarification continues after stop_clarify: {row}')
            if next_action == 'clarify' and next_question is not None and answer != next_question:
                raise ValueError(f'Assistant response disagrees with clarification question: {row}')
    return qa_count


def build(out, counts):
    offsets = {'train': 0, 'valid': 10000, 'test': 20000}
    if any(split not in offsets or not isinstance(n, int) or n < 0
           for split, n in counts.items()):
        raise ValueError('Use non-negative integer counts for train, valid and test')
    seen = set()
    datasets = {}
    stats = {}
    # Reserve evaluation inputs first: changing only --train keeps valid/test fixed.
    for split in ('valid', 'test', 'train'):
        if split not in counts:
            continue
        rng = random.Random(SEED + offsets[split])
        stories = []
        task_counts = dict.fromkeys(TASKS, 0)
        qa_count = 0
        for i in range(counts[split]):
            task = TASKS[i % len(TASKS)]
            for attempt in range(MAX_GENERATION_ATTEMPTS):
                action = rng.choice(ACTIONS)
                story = render(rng, task, action)
                key = dialogue_key(story)
                if key not in seen:
                    qa_count += validate_story(story)
                    seen.add(key)
                    stories.append(story)
                    task_counts[task] += 1
                    break
            else:
                raise ValueError(
                    f'Cannot generate enough distinct {task} dialogues for {split}; '
                    'expand the templates or reduce the requested count')
        rng.shuffle(stories)
        datasets[split] = stories
        stats[split] = {'stories': len(stories), 'qa': qa_count, 'tasks': task_counts}
    # Validate every split before replacing any existing dataset files.
    out.mkdir(parents=True, exist_ok=True)
    for split in offsets:
        if split in datasets:
            path = out / f'{split}.txt'
            path.write_text(''.join(datasets[split]), encoding='utf-8')
            print(f'{path}: {stats[split]["stories"]} unique dialogues, {stats[split]["qa"]} QA pairs')
    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'products-v4')
    parser.add_argument('--train', type=int, default=DEFAULT_COUNTS['train'])
    parser.add_argument('--valid', type=int, default=DEFAULT_COUNTS['valid'])
    parser.add_argument('--test', type=int, default=DEFAULT_COUNTS['test'])

    args = parser.parse_args()
    try:
        build(args.out, {'train': args.train, 'valid': args.valid, 'test': args.test})
    except ValueError as error:
        parser.error(str(error))
