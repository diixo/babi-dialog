# babi-dialog

## Генерация данных: версия 4

Актуальный генератор — `build_v4.py`: 50 действий и 13 типов задач.
`build.py`, `build_v2.py` и `build_v3.py` сохраняют предыдущие версии.

```shell
python build_v4.py
```

По умолчанию файлы записываются в `products-v4/`:

| Выборка | Уникальных диалогов | На каждый тип задачи |
| --- | ---: | ---: |
| train | 13 000 | 1 000 |
| valid | 1 300 | 100 |
| test | 1 300 | 100 |

Размеры задаются в **историях**, каждая из которых содержит несколько QA-пар.
Для сравнения с большим обучающим набором:

```shell
python build_v4.py --train 26000 --out products-v4-26k
```

При неизменных параметрах генерация воспроизводима. `valid` и `test` создаются
первыми: изменение только `--train` сохраняет эти две выборки без изменений.
Диалоги распределяются по задачам поровну; при некратном 13 размере разница
составляет не более одной истории.

В `roles` и `questions` есть начальные реплики как с известным, так и с неизвестным
объектом. Используются 12 формулировок намерения, варианты времени и явное
упоминание действия. Известное значение всегда присутствует в тексте пользователя.

Повторы исключаются внутри выборок и между ними по тексту диалога: QA-разметка,
регистр, пробелы и завершающий ответ ассистента не создают новый входной пример.
Общие шаблоны, префиксы диалогов и отдельные QA-пары при этом могут встречаться
в разных выборках. Для проверки обобщения на новые формулировки нужен отдельный тест.

Ассистент уточняет **каждый необходимый для текущего сценария факт до ответа
или явного отказа**. Ответ «не знаю» оставляет факт неизвестным: ассистент продолжает
уточнять его, при необходимости переформулируя вопрос. Число попыток само по себе
не вызывает `stop_clarify`. Начальная реплика без конкретного объекта не считается
ответом на уточняющий вопрос.

`loop? = yes` отмечает повторное уточнение после двух неопределённых ответов,
сохраняя `next action? = clarify`. Задача `loop` включает 2–5 таких ответов, после
которых пользователь сообщает значение или явно отказывается. Решение о следующем
действии зависит от фактов диалога, а не от скрытого имени задачи.

В сценариях одежды, отдыха и сна собираются все три необходимых факта:
погода/время суток/сезон либо время с последнего отдыха или сна/его последняя
продолжительность/обычная продолжительность. Часть значений может присутствовать
уже в первой реплике; известные факты повторно не запрашиваются. В разметке:

- `missing?` перечисляет все неизвестные факты, включая те, от которых отказались;
- `declined?` отмечает факты с явным отказом;
- `next fact?` выбирает неизвестный факт, от которого пользователь не отказался;
- `ask about fact?` задаёт соответствующий вопрос.

Отказ прекращает уточнение конкретного факта, после чего ассистент спрашивает
о других недостающих фактах. Когда все факты известны, выдаётся `advance` и совет.
Если оставшиеся неизвестные факты явно отклонены пользователем, выдаётся
`stop_clarify` с объяснением неполноты данных. Отказ не превращается в известное
значение. В задачах об объекте или эмоции единственный недостающий факт также
уточняется до ответа или отказа.

Перед записью всех выборок проверяются нумерация, опорные ссылки, согласованность
решений и ответов ассистента. Если разнообразия недостаточно для запрошенного
размера, генерация завершается с ошибкой до перезаписи файлов.

Регрессионные проверки:

```shell
python -B -m unittest discover -s tests -v
```


## 1. Что умеет генератор

Реализовал девять типов упражнений, охватывающих обсуждавшиеся задачи.
| Тип        | Обучаемая способность |
| ---------- | --------------------- |
| Roles	     | Определение `actor`, `action`, `intention`, `object` |
| Resolution | Распознавание определённости ответа |
| Question Linking | Связывание действия и недостающего аргумента с вопросом |
| History | Накопление и разрешение информации |
| Yes Context | Понимание `Yes` в контексте вопроса |
| Ambiguous Yes | Распознавание бессмысленного `Yes` |
| Revision | Обновление ранее полученного значения |
| Decline | Прекращение уточнений при отказе |
| Loop | Распознавание повторяющихся безуспешных уточнений |


## 2. Что особенно важно для эксперимента

Я намеренно разделил два вида истории:

* **Обычные истории** — пользователь даёт неопределённые ответы, а модель должна определить, появилась ли необходимая информация.

* **Истории с зацикливанием** — в контексте присутствуют многократные уточнения.
  В версии 4 они отмечаются `loop? = yes`; ассистент продолжает уточнение до
  конкретного ответа или явного отказа. Повторное «не знаю» не является отказом.

Так мы можем проверить, способна ли модель распознавать проблему в уже накопленной истории.

Однако текущий генератор обучает прежде всего **извлечению семантических значений и коротким решениям**, а не свободной генерации диалогов.

Моя рекомендация для следующей версии — добавить специальную задачу **Semantic State Continuation**. В ней модель должна будет продолжать последовательность:
```text
actor → action → intention → object → missing → question → answer → updated object
```

Причём каждая новая реплика пользователя должна изменять семантические факты только там, где это действительно необходимо.

Именно это позволит экспериментально проверить твою основную гипотезу: **может ли модель усвоить связь между действием, недостающей информацией и соответствующим вопросом, чтобы не зацикливаться в долгих разговорах.**


### Пример

Я бы предложил **сохранить наш bAbI-формат, но ввести понятие** `sentence`. Тогда одна реплика пользователя (`User`) может содержать несколько предложений, и каждое предложение получает собственный набор семантических ролей.

При этом важно различать три уровня:
- **Utterance** — сообщение пользователя целиком.
- **Sentence** — отдельное предложение внутри сообщения.
- **Semantic roles** — `actor`, `action`, `intention`, `object`, `location`, `emotion` и другие, относящиеся к конкретному предложению.

Самое интересное — после извлечения ролей из каждого предложения можно сформировать общее состояние всей реплики, а затем обучать модель выбирать следующий вопрос.


#### 1. Несколько предложений в одной реплике:

```text
1 User: I am tired. I've been programming all day. I want to relax.

2 sentence?	I am tired.	1
3 actor?	user	2
4 state?	tired	2
5 emotion?	unknown	2

6 sentence?	I've been programming all day.	1
7 actor?	user	6
8 action?	program	6
9 duration?	all day	6
10 location?	unknown	6

11 sentence?	I want to relax.	1
12 actor?	user	11
13 action?	relax	11
14 intention?	relax	11
15 object?	unknown	11

16 current state?	tired	2 4
17 current action?	program	6 8
18 current intention?	relax	11 14
19 next action?	clarify	16 18
20 assistant response?	You've been programming all day. When did you last take a break?	4 8 9 18
```

Здесь сохраняется твой принцип: **каждое извлечение имеет ссылку на соответствующее предложение**, а вопросы об общем состоянии опираются на несколько фактов.

При этом `tired` — физическое состояние, а не обязательно эмоция `sadness`. Это важно для наших семи категорий настроения.


#### 2. А если несколько предложений содержат разные действия?

```text
1 User: I went to the office this morning. I finished programming. Now I want to play video games.

2 sentence?	I went to the office this morning.	1
3 actor?	user	2
4 action?	go	2
5 location?	office	2
6 time_of_day?	morning	2
7 tense?	past	2

8 sentence?	I finished programming.	1
9 actor?	user	8
10 action?	program	8
11 aspect?	completed	8

12 sentence?	Now I want to play video games.	1
13 actor?	user	12
14 action?	play_video_games	12
15 intention?	play_video_games	12
16 time?	now	12

17 current intention?	play_video_games	12 15
18 next action?	advance	15 17
19 assistant response?	Sounds like you've finished your work. What game are you thinking of playing?	8 11 12 15
```

В этом примере модель не должна спрашивать, что пользователь делал утром. Его текущее намерение — играть в видеоигры.

Но эта связь не должна быть жёстким правилом «последнее предложение всегда важнее». Иногда последнее предложение просто уточняет первое.


#### 3. Если одно предложение содержит несколько действий

```text
1 User: I want to buy a laptop and a phone.
2 sentence?	I want to buy a laptop and a phone.	1
3 actor?	user	2
4 action?	buy	2
5 intention?	buy	2
6 object_1?	a laptop	2
7 object_2?	a phone	2
8 object count?	2	6 7
9 next action?	advance	4 6 7
10 assistant response?	Are you planning to buy both at the same store?	4 6 7
```

Это простейшая форма поддержки множественных объектов.

Позднее можно перейти к структуре `event_1`, `event_2`, если в одном предложении встречается несколько независимых действий.


#### 4. Как реализовать в генераторе

Я бы добавил **новую задачу** `multi_sentence`, не меняя существующие 11 задач.

Алгоритм генерации:
```text
Generate utterance
Создать реплику из 2–4 связанных предложений

Extract roles per sentence
Каждое предложение получает свои QA-строки

Aggregate semantic context
Определить текущее состояние и намерение

Generate dialogue decision
clarify / advance / stop_clarify

Generate assistant response
Вопрос или продолжение, учитывающее всю реплику
```

Важно, что при программной генерации не обязательно сначала создавать текст, а потом разбирать его настоящим SRL-парсером. Мы можем строить предложение из заранее известных смысловых компонентов и сразу знать правильные роли. Это даст точную разметку для синтетического датасета.


#### 5. Я бы предусмотрел два варианта обучения
Вариант	Что делает модель
```text
Sentence-level	Извлекает роли из каждого предложения
Utterance-level	Объединяет извлечённые факты и выбирает ответ для всей реплики
```

**Не стоит делать отдельный ответ ассистента на каждое предложение.** Пользователь отправил одно сообщение, значит ассистент должен сформировать одну ответную реплику с учётом всех предложений.

И ещё я бы добавил специальный случай, который особенно важен для предотвращения зацикливания:
```
User: I wanted coffee. Actually, I changed my mind. I'll have tea.
```


Здесь три предложения, но текущее выбранное значение — tea. Модель должна понимать исправления внутри одного сообщения, не требуя дополнительного подтверждения.

##### Моя рекомендация
Для первого эксперимента я бы добавил `multi_sentence` с 2–3 предложениями, используя уже существующие действия и роли.

Сначала простые комбинации:
- Состояние + действие: *I'm tired. I've been working all day.*
- Действие + намерение: *I finished work. I want to relax.*
- Намерение + уточнение: *I want tea. Make it strong.*
- Намерение + исправление: *I want coffee. Actually, tea.*
- Настроение + действие: *I'm happy today. I finally fixed my code.*

После этого можно проверить, способна ли GPT-2 правильно извлекать роли из нескольких предложений, объединять их и формировать **один связный ответ без повторных уточнений уже известных фактов**.

Это будет естественным развитием нашего bAbI-подхода и не потребует перехода на JSON или полноценную машину состояний.


## Semantic Dialogue Learning Chains (build_v4)

The dataset uses a bAbI-style question-answer format to teach a language model how to extract semantic information, track dialogue context, resolve uncertainty, update facts, and generate context-aware responses.

Each question-answer pair includes supporting line references. Semantic information is represented through structured textual questions rather than JSON.

### 1. Semantic Role Extraction

`actor → action → intention → object`

Extract the semantic components of a user's utterance, including who is involved, what action is described, what the user intends, and which objects are associated with the action.

### 2. Missing Information and Clarification

`action → object → missing → next action → question`

Determine whether the information required to continue the dialogue is known. If relevant information is missing, generate an appropriate clarification question.

### 3. Uncertainty Resolution

`question → user answer → certainty → clarify / advance`

Interpret a user's answer in the context of the preceding question. Distinguish known, unknown, and ambiguous information, and determine whether clarification is still necessary.

### 4. Contextual State Updating

`previous object → user answer → operation → updated object`

Update previously established information when the user provides new values, confirms a choice, corrects an earlier statement, or changes a preference.

### 5. Dialogue Response Generation

`semantic context → next action → assistant response`

Generate a relevant assistant response based on extracted semantic roles, known information, missing arguments, and the current dialogue context.

### 6. Object Accumulation and Tracking

`action → object → add → current objects`

Associate multiple objects with a single action and accumulate them across successive user utterances.

### 7. Object Removal

`action → current objects → remove → updated objects`

Remove a specific object from an action without affecting other objects associated with the same action.

### 8. Object Replacement

`action → previous object → replace → new object → updated objects`

Replace previously selected objects when the user changes their mind while preserving unrelated information.

### 9. Object Preservation

`action → current objects → keep → current objects`

Preserve previously established objects when new utterances do not modify them or explicitly confirm that they remain relevant.

### 10. Action Cancellation and Completion

`action → current objects → clear / complete → updated state`

Distinguish between cancelling an action and completing the collection of information associated with that action.

- `clear`: Remove all objects associated with the specified action.
- `complete`: Mark the selection as finished without removing its objects.

### 11. Multiple Action Tracking

`actor → actions → objects → operations → updated action objects`

Track multiple independent actions and their associated objects within the same dialogue.

Updates to one action must not unintentionally modify objects belonging to another action.

### 12. Cross-Action Object Independence

`shared object → action A + action B → cancel action A → preserve action B`

Allow the same object to participate in several independent actions.

Cancelling or modifying one action does not automatically cancel or modify other actions involving that object.

### 13. Contextual Answer Interpretation

`previous question → user answer → resolved argument → updated state`

Interpret short responses such as "Yes", "No", "Maybe", or "Actually" using the preceding dialogue context.

The same answer may carry different meanings depending on the question it addresses.

### 14. Dialogue History and Memory

`dialogue history → extracted facts → current semantic state → next action`

Accumulate semantic information throughout a conversation and use the current state to avoid asking questions that have already been answered.

### 15. Loop Prevention and Clarification Recovery

`missing → clarify → repeated uncertainty → rephrase → answer / explicit refusal`

Recognize repeated unsuccessful clarification attempts and rephrase questions while
keeping the unresolved fact pending. The `loop?` annotation does not stop clarification.

Continue until the user supplies a value or explicitly declines that fact. A refusal
waives only that fact; collect other required facts before ending clarification.
Never ask again for facts that are already known or explicitly declined.

### 16. Emotion Recognition and Dialogue Adaptation

`user utterance → emotion → current emotion → next action → assistant response`

Identify one of seven emotion categories:

`neutral / joy / surprise / sadness / anger / disgust / fear`

When the emotion is unknown, ask for clarification. When it is known, generate a contextually appropriate follow-up question or response.

### 17. Contextual Emotion Tracking

`action → dialogue context → emotion → context-aware question`

Connect the user's emotional state with previously discussed activities, such as work, programming, gaming, walking, or relaxing.

Use emotional context to guide the conversation without unnecessarily repeating questions about the user's mood.

### 18. Advice Generation

`user state → missing information → clarification → collected facts → advice`

Gather relevant information before generating advice.

Supported scenarios include:

- Weather, time of day, and season → clothing advice.
- Tiredness and rest duration → rest suggestions.
- Sleepiness and sleep history → sleep-related suggestions.

Advice is generated from the collected information rather than from an isolated user utterance.

### 19. Semantic State Continuation

`actor → action → intention → object → missing → question → user answer → operation → updated objects → next action → assistant response`

Combine semantic role extraction, clarification, contextual answer interpretation, object tracking, state updates, and response generation into a continuous dialogue-learning process.

### 20. Implicit Semantic Behavior Learning

`structured semantic examples → learned relationships → dialogue behavior`

The training objective is to encourage the model to learn relationships between semantic roles, missing information, dialogue history, user responses, and appropriate assistant behavior.

Explicit semantic questions provide supervision during training. Whether these relationships transfer to natural dialogue without explicit semantic annotations must be evaluated separately.

---

## Core Learning Objective

Teach a language model to maintain coherent semantic relationships throughout a conversation:

**Understand → Extract → Clarify → Resolve → Accumulate → Update → Preserve → Respond**

The model should learn to:

- Recognize user intentions and actions.
- Associate multiple objects with their corresponding actions.
- Identify missing or uncertain information.
- Ask relevant clarification questions.
- Resolve short and ambiguous answers using dialogue context.
- Accumulate, remove, replace, and preserve objects.
- Track several independent actions simultaneously.
- Respect refusals and user-initiated changes.
- Track repeated uncertainty and clarify each missing fact until an answer or explicit refusal.
- Adapt responses to emotions, activities, and contextual information.
- Generate natural responses consistent with the current semantic state.

The ultimate goal is **stable, context-aware dialogue behavior without requiring an explicit finite-state machine during inference**.
