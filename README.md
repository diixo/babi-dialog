# babi-dialog

## SRL по учебнику

`build_book_srl.py` читает английские предложения и их SRL из
[sentences.json](data/book_srl/sentences.json), автоматически синхронизирует
каталог шаблонов, проверяет варианты и собирает датасет за один запуск.
Markdown и PDF книги для сборки не нужны.
Одна story содержит **одно исходное предложение или его допустимую вариацию**,
затем блоки `fragment?` с вопросами `actor?`, `predicate?`, `verb?`, `action?`
и дополнительными ролями. Вопросы идут без индексов и относятся к текущему
фрагменту. Реплик ассистента и управления диалогом нет.

```shell
python build_book_srl.py
```

Нужен Python 3.10+, внешних зависимостей нет. Результат записывается в
`products-book-srl/{train,valid,test}.txt`. Параметры `--library` и `--out`
позволяют указать другие каталоги исходных JSON и готового датасета.
Флаг `--annotate` дополнительно собирает прежний набор размеченных шаблонов
в `products-book-srl/annotations/{train,valid,test}.txt` тем же запуском.

В `sentences.json` находятся **8 046 английских предложений из уроков 1–84**.
Каждая запись содержит `id`, `unit`, `text`, `facts`, `predicate_spans`,
`fragment_spans` и `verb_spans`.
Все **11 966 исходных фактов** имеют `predicate` и проверяемую привязку к тексту.
Русских пояснений, заголовков, списков слов, заданий с пропусками, исходной
Markdown-разметки и незавершённых фрагментов в этом файле нет. Отдельные
обращения и именные фразы вроде `Waiter!` и `A ceiling.` исключены.
Короткие грамматические ответы вроде `Yes, she does.` сохраняются с неизвестным
действием, если оно не названо в самом предложении.

С учётом разрешённых подстановок сборка даёт **12 231 уникальный пример**.
Все 84 урока подготовлены: английские предложения и ручная SRL находятся в JSON.

Каждый запуск обрабатывает все 84 урока, показывает число шаблонов и готовность
каждого урока, проверяет разметку и все допустимые варианты. Если хотя бы один
урок не подготовлен или проверка выявила ошибку, сборка останавливается до записи файлов.
При сборке автоматически выводятся количества однофрагментных и многофрагментных
примеров для `train`, `valid`, `test`, а также общие количества и проценты.
Статистика учитывает уникальные примеры после удаления дублей.

| Файл | Назначение |
| --- | --- |
| [sentences.json](data/book_srl/sentences.json) | Английские предложения, номера уроков и готовые ответы SRL |
| [fact_templates.json](data/book_srl/fact_templates.json) | Шаблон каждого размеченного предложения, исходный текст, слоты, совместимость и ответы SRL |
| [slot_values.json](data/book_srl/slot_values.json) | Общие наборы допустимых значений с указанием места в книге |

Текст и ответы используют одни и те же подстановки. Например:

```text
{actor_frame.word_subject}{actor_frame.word_auxiliary} {verb.word} an interesting {noun_1.word}.
actor: {actor_frame.actor}
predicate: {actor_frame.word_auxiliary} {verb.word}
action: {verb.lemma}
object: interesting {noun_1.word}
```

Из исходного `I'm reading an interesting book.` получаются, в частности,
`She's writing an interesting article.` и `You're translating an interesting letter.`
Варианты подлежащего связаны с формой вспомогательного или смыслового глагола;
`I have` меняется на `He has`. Сочетание артикля с существительным также связано:
`a book` меняется на `an article`. Ограничения `constraints` разрешают только
перечисленные пары значений слотов; например, `write` не комбинируется с `magazine`.

Варианты и нужные формы слов уже записаны в `slot_values.json`. Поле `evidence`
сохраняет написание и источник: у старых значений — координаты в книге,
у новых — ID предложения и позиции слова в `sentences.json`.
Проверка новых ссылок выполняется по JSON, без чтения книги.
Совместимость зависит от всего предложения: `The cow gives milk.` не допускает
замены на coffee, а `strong tea` — на strong water. Непроверенные альтернативы
не добавляются. Если безопасная замена ещё не задана, слот содержит только
исходное значение. Это ограниченная проверенная библиотека, а не обещание
перебора всех лингвистически возможных сочетаний.

В уроках 13–16 отдельно ограничены сочетания количества с исчисляемыми и
неисчисляемыми существительными, общее и раздельное владение, формы местоимений.
Например, `little money` и `few friends` сохраняют разные наборы подстановок,
а `She took her bag and left.` допускает `He took his umbrella and left.`
Местоимение раскрывается только при наличии опоры в том же предложении:
в `My bag is black, hers is brown.` — `her bag`, а в `Can I use yours?` — `yours`.

В уроке 17 возвратное местоимение связывает участника с ним самим:
`I hurt myself.` → `actor: user`, `action: hurt`, `object: user`, `reflexive: yes`.
В `He made himself a cup of tea.` совпадают `actor` и `beneficiary`,
в `Jane looked at herself in the mirror.` — `actor` и `target`.
Формы `I/myself`, `he/himself`, `we/ourselves` меняются одной связанной подстановкой.
Усилительное `You yourself told us the story.` получает `emphasis: actor`
и `reflexive: no`; объектом остаётся `story`. Предложение с `by himself`
сохраняет значение самостоятельности (`manner: without help`).

В уроках 18–20 подстановки сохраняют формы `some/any/no`, единственное число
сказуемого при `everybody`, смысл отрицания при `nobody/nothing` и число после
`another`. Взаимное действие с `each other` или `one another` получает
`reciprocal: yes`, `reflexive: no`; его участники образуют группу.

Уроки 21–25 добавляют выбор и совместное действие (`coordination: or/and`,
`coordination_role`), признаки после связочных глаголов (`complement`) и способ
действия (`manner`). Сравнения сохраняют `comparison` и степень `degree`:
`comparative`, `equative` или `superlative`. Конструкция `the ... the ...`
размечается как зависимость с `status: conditional` и `condition`.
Согласование с ближайшим подлежащим в `either ... or` и `neither ... nor`
меняется вместе с самим подлежащим. `farther/further` взаимозаменяются в
проверенных шаблонах расстояния; `further information` сохраняет своё значение.
В упражнениях на исправление ошибок используются исправленные предложения.

Уроки 26–30 продолжают сравнения наречий и разделяют способ действия, место,
направление, время и длительность. `have got` при владении получает `action: have`
и `aspect: simple`; в `was having a shower` сохраняется `aspect: continuous`.
Замена `have` на `have got` разрешена только в проверенных шаблонах владения.
Вопросы получают `status: questioned`; отрицательный вопрос сохраняет собственную
полярность, а отдельное `Yes, he does.` — неизвестное действие.
Альтернативы с `or` сохраняются в соответствующей роли одного факта, включая
подлежащее, объект, место, время и владельца. `won't have to` обозначает отсутствие
необходимости (`status: not required`, `negation_scope: modality`).

Уроки 31–35 добавляют разделительные, специальные и косвенные вопросы.
Вопросительный «хвостик» сохраняет полярность основной части; просьбы
`Close the window, will you?` остаются `required`, а предложения с `Let's` —
`intended`. Подстановки учитывают особую форму `I am …, aren't I?`.
В специальных вопросах `unknown` стоит в запрашиваемой роли: `actor`,
`object`, `recipient`, `possessor`, `location`, `duration` и других.
`What is he?`, `What is he like?` и `What does he look like?` различаются
полем `attribute`: профессия, характер и внешность.
Вложенный вопрос размечается отдельно от `know`, `remember`, `wonder` или
`think`. Отрицание в `I don't know why they are late.` относится к `know`;
вложенная часть сохраняет положительную полярность и `status: questioned`.
Варианты `if/whether` сохраняют прямой порядок слов. Вопросы о длительности
выделяют время и само действие; маршруты и транспорт сочетаются только
по перечисленным ограничениям.

В уроках 36–40 `There is a book on the table.` получает `actor: none`,
`action: exist`, `object: book`, а `The book is on the table.` — `actor: book`
и `action: be`. При перечислении предметов сохраняются отдельные количества,
а форма `is/are` связана с первым существительным. Восклицательные предложения
с `how/what` получают `clause_type: exclamative`, известный признак или способ
действия и `status: asserted`. Подстановки сохраняют артикли и исчисляемость.

Present Simple сохраняет согласование и каждое действие в последовательности.
В упражнении 39.3 сохранены исходные грамматически правильные утверждения и их
содержательные исправления: SRL отражает текст, а не проверяет истинность сообщения.
Present Continuous получает `aspect: continuous`; явные будущие договорённости —
`status: planned`, повторяющиеся действия с раздражением — `attitude: annoyance`.
В `She usually makes breakfast herself.` усиление относится к исполнителю,
а в `We are washing our hands.` объектом остаются руки и сохраняется их владелец.

В уроках 41–45 различаются состояния и действия: `The meat smells wonderful!`
содержит признак мяса, а `Why are you smelling the meat?` — действие человека
над объектом. Расписания получают `status: scheduled`, договорённости — `planned`.
Прогноз `It's going to rain.` не создаёт намерения. В условных и временных
придаточных сохраняется Present Simple, а во вложенных вопросах о будущем —
Future Simple; условное событие не становится утверждённым фактом.

Past Simple сохраняет неправильные формы и `did + infinitive`, а `used to`
и привычное `would` — прошлое повторяющееся действие. Present Perfect получает
`tense: present`, `aspect: perfect`; `just/already/yet`, число повторений и
текущие периоды времени остаются отдельными признаками. В `It's not the first
time the cat has torn my curtains!` отрицание относится к «первому разу».
Подстановки связывают `have/has` с подлежащим, сохраняют артикли названий мест
и не переносят `yesterday/last night` в шаблоны Present Perfect.

В уроках 46–50 длительность с `for` записывается в `duration`, точка отсчёта
с `since` — в `time`. Событие после `since` сохраняет Past Simple и собственную
полярность, даже если основная часть отрицательна. Present Perfect Continuous
получает `aspect: perfect_continuous`: процесс не означает завершённого результата.
Вопросы `how many` сохраняют количество, а `how long` — длительность.
Глаголы состояния (`know`, `own`) остаются в Present Perfect.

У завершённых ситуаций длительность не меняет Past Simple: `She worked at school
for three years and then quit.` В Past Continuous фоновое действие и отдельное
событие получают разные факты и аспекты. Планы из прошлого остаются `planned`
или `intended` с `operation: none`; они не добавляют текущее намерение.
Подстановки связывают `have/has`, `was/were`, притяжательные формы и участников
взаимного действия. Списки слов, короткие ответы без сказуемого и ошибочные
варианты из упражнений на исправление в корпус не включаются.

В уроках 51–55 Past Perfect получает `tense: past`, `aspect: perfect`,
Past Perfect Continuous — `aspect: perfect_continuous`. В `had dinner`
сохраняется Past Simple, в `had bought` — Past Perfect. Длительность процесса,
число завершённых действий и состояния размечаются отдельно. Конструкции
`hardly ... when` и `no sooner ... than` сохраняют положительную полярность
и связь `temporal_relation: immediately before`.
В литературных примерах реплики с авторскими словами сохраняют `status: reported`;
местоимения раскрываются только при наличии опоры в том же предложении.
Самостоятельные `But I shall.` и `Perhaps I can here.` сохраняют `action: unknown`.

В уроках 56–59 отдельно размечены слова автора и содержание речи. Сообщения
сохраняют `reported`, вопросы — `questioned`, команды — `required`; совет,
обещание и притворство не становятся совершёнными действиями. В прямой речи
`I/me` относятся к названному говорящему, а `you` — к указанному адресату;
если адресат отсутствует, используется `unknown`. Согласование времён сохраняет
формы из книги, включая настоящее время общих фактов и прошлое время
временных придаточных. Future-in-the-Past получает `tense: future`, `modality: would`.

В уроке 60 пассив сохраняет смысловые роли: `The letter was written by a boy
with a pencil.` → `actor: boy`, `action: write`, `object: letter`,
`instrument: pencil`, `voice: passive`. Без указанного исполнителя `actor`
остаётся `unknown`, даже если исполнитель назван в соседнем активном варианте.
Подстановки связывают объект, причастие и формы `is/are`, `was/were`, `has/have`.

В уроках 61–62 получатель в пассиве остаётся `recipient`: `I was given really
good advice.` → `actor: unknown`, `object: really good advice`, `recipient: user`.
Предложный пассив сохраняет `topic` или `target`: `He is being waited for by
everybody.` → `actor: everybody`, `target: he`. Длительность сама по себе не
делает пассивную форму `perfect_continuous`; сохраняется аспект из предложения.

В уроках 63–65 нереальные условия и результаты получают `mood: subjunctive`,
`status: counterfactual` и связь `condition`. `tense` здесь обозначает смысловое
время: `If I knew ... I would phone ...` — настоящее, `If I had known ...
I would have phoned ...` — прошедшее с `aspect: perfect`. В смешанных условных
предложениях время каждой части размечается отдельно. Вопрос сохраняет
`status: questioned` вместе с `mood: subjunctive`.
В `I wish ...` утверждается само желание; желаемое состояние получает
`counterfactual`, собственное время и полярность. Обратный факт не добавляется.
`wish`, условные советы и нереальные действия сохраняют `operation: none`.
Подстановки связывают формы `wish/wishes`, местоимения, владельцев и ссылки
на предметы внутри предложения; `it` без опоры в той же story остаётся `it`.

Уроки 66–70 различают значения модальных глаголов через `modal_meaning` и
`status`: умение и возможность — `possible`, разрешение — `permitted`,
обязанность и запрет — `required`, предположение — `inferred`.
Вежливая просьба с `could` относится к настоящему; способность в прошлом —
к прошедшему. `don't have to` получает `not required` и `negation_scope: modality`.
`be to` при договорённости получает `planned`, без утверждения, что план выполнен.

`must have done` — предположение о прошлом (`past`, `perfect`), а
`had to do` — прошлая необходимость (`past`, `simple`). У продолжающегося
`must/can have been doing` сохраняется `present`, `perfect_continuous`.
В `can't have failed to ...` отрицание относится к модальному предположению
о неудаче; вложенное действие остаётся `hypothetical`, не объявляется совершённым.
`be (un)likely to` сохраняет вероятность будущего события, без утверждения
его отсутствия. Упрёк с `might` получает `attitude: reproach`: нынешний совет —
`advised`, упущенная возможность в прошлом — `counterfactual`, `operation: none`.

В уроках 71–74 `may/might` в предположениях получает `possible`; в `may not`
отрицается возможное действие (`negation_scope: predicate`), а запрет имеет
`modal_meaning: prohibition`. `needn't have done` отмечает выполненное ненужное
действие (`outcome: completed`); `didn't have to` не сообщает, выполнено ли оно.
В советах о прошлом `should have done` получает `outcome: not completed`,
`shouldn't have done` — `completed`. Вопросы с `should have` исход не утверждают.

Герундий в уроке 75 получает `verb_form: gerund`, `tense: nonfinite` и
`syntactic_function`. Формы `doing` и `being done` имеют `aspect: simple`,
`having done` и `having been done` — `perfect`. Пассив сохраняет смысловые роли:
в `I remember being asked for help.` исполнитель неизвестен, `recipient: user`.
В `My shoes need repairing.` форма активная (`form_voice: active`), значение
пассивное; ремонт требуется, но не объявляется выполненным. Владелец герундия
задаёт его исполнителя: `I don't mind his opening the window.` → `actor: he`.
Подстановки меняют владельца и исполнителя вместе, сохраняя говорящего отдельно.

В уроке 76 герундий после `deny` получает `denied`, после `give up` — `ceased`.
`without doing` отрицает вложенное действие; совершённость основного действия
сохраняется. В `can't help doing` действие совершается непроизвольно
(`control: involuntary`), а `need doing` сохраняет неизвестного исполнителя.

В уроке 77 `verb_form: infinitive` отделяет инфинитив от личной формы глагола.
`pretend` получает вложенное `pretended`, а `claim/seem` — `reported`.
`temporal_relation` различает одновременность и предшествование; перфектный
инфинитив остаётся `tense: nonfinite`. После `make/let` активный инфинитив имеет
`marker: bare`; в пассиве исполнитель `make/allow` неизвестен, а исполнитель
вложенного действия указан отдельно.

В уроке 78 `for somebody to do` задаёт исполнителя инфинитива; `too … to` и
`not … enough to` получают вложенное `impossible`. `have something done`
имеет `causative: yes`: инициатор организует действие, неизвестный исполнитель
выполняет его. `I must have my watch repaired.` выражает необходимость сейчас,
не предположение о прошлом. Вопросы о выполнении сохраняют `questioned`.

В уроке 79 `stop doing` прекращает действие, а `stop to do` задаёт цель остановки.
`remember doing` относится к прошлому событию; `remember to do` — к требуемому
будущему действию. `try` различает `method: effort` и `experiment`.
`be used to doing` описывает привычность, `used to do` — привычное действие
в прошлом. В `having a rest` слово `having` не образует перфект.

Причастия урока 80 имеют `verb_form: participle`, `tense: nonfinite`,
`participle_form` и `syntactic_function`. `doing/being done` сохраняют
`aspect: simple`, `having (been) done` — `perfect`; `temporal_relation` задаёт
отношение к основному действию. У пассивного причастия `by …` задаёт исполнителя,
а подлежащее предложения может быть объектом или получателем. Форма на `-ed`
сама по себе не делает залог пассивным: в `worn-out clothes` одежда износилась.

В уроках 81–82 наблюдатель и исполнитель вложенного действия разделены:
`I saw him buy a newspaper.` содержит `see` от `user` и `buy` от `he`.
Инфинитив после глагола восприятия получает `event_view: whole`, причастие —
`process`; форма `buying` сама по себе не задаёт `aspect: continuous`.
`I hear you speak English.` передаёт сведения (`evidence: hearsay`), а
`I heard them speaking English.` — непосредственное восприятие.
В `I want you to close the window.` закрыть окно должен `addressee`;
ожидание, просьба или попытка не превращаются в совершившееся действие.

Урок 83 сохраняет `continuation: still` и `completion: yet/already` отдельно
от времени и аспекта. `He still hasn't taken a decision.` отрицает действие
(`negation_scope: predicate`), а `He isn't still waiting.` — его продолжение
(`negation_scope: continuation`). Вопросы остаются `questioned`.

В уроке 84 `also/too/either` получают `addition`, а краткие подтверждения —
`addition: so/neither` и свой `auxiliary`. У отдельного `So were we.`
действие, объект, аспект и залог остаются `unknown`: предыдущее предложение
не входит в эту story. Подстановки связывают местоимение с формой вспомогательного
глагола (`Neither am I.` → `Neither are they.`).

Пример формата (между вопросом, ответом и номером опорной строки — табуляции):

```text
1 User: She's writing an interesting article.
2 fragment?	She's writing an interesting article	1
3 actor?	she	2
4 predicate?	's writing	2
5 verb?	writing	2
6 action?	write	2
7 object?	interesting article	2
8 aspect?	continuous	2
9 voice?	active	2
10 polarity?	positive	2
11 status?	asserted	2
```

Каждый факт начинает новый блок `fragment?`; вопросы `actor?`, `predicate?`,
`verb?`, `action?` повторяются без индексов. Ответы ссылаются на строку текущего
фрагмента, а сведения за его пределами — дополнительно на исходную строку `1`.
Например, в `he is taller` сравнение с `Paul` требует полного исходного предложения.
Внутри story нет пустых строк: всё предложение и его разбор образуют один
обучающий пример. Пустая строка отделяет только следующую story.
Вопросы `tense?` и роли со значением `none` не выводятся. Исключение —
`predicate?` и `verb?`: `none` явно обозначает отсутствие глагола в тексте.
Неизвестное значение сохраняется как `unknown`. `I` обозначается как
`user`, `you` — `addressee`.
Пассив сохраняет семантического исполнителя, а не выдаёт подлежащее за actor.
У отдельного `Have you seen them?` объектом остаётся `them`: сведения из
предыдущей story не подставляются. Связи внутри того же предложения сохраняются.

`predicate` сохраняет глагольную группу из текста: исходное написание,
вспомогательные глаголы, отрицание `not/n't/never`, инфинитивное `to` и частицы
фразового глагола. Сокращения не раскрываются: `She's been sleeping` получает
`'s been sleeping`. Подлежащее, объект и обычные наречия в группу не входят:
`Have you seen them?` → `Have seen`, `has already written` → `has written`,
`turned it off` → `turned off`. У `was sitting and watching` оба факта
используют общий `was`: `was sitting` и `was watching`.

`predicate_spans[i]` указывает фрагменты для `facts[i]`: пары `[start, end]`
считают символы Python-строки с нуля, правая граница не включается.
Фрагменты идут в порядке текста, соединяются одним пробелом и могут повторно
использоваться разными фактами. В `Yes, he is, but he is taller…` первый `is`
имеет границы `[8, 10]`, второй — `[19, 21]`. Привязки находятся вне `facts`
и не превращаются в обучающие вопросы.

`fragment_spans[i]` задаёт один непрерывный фрагмент исходного текста для
`facts[i]`; `verb_spans[i]` — форму основного глагола и его частицы.
Все три набора границ хранятся и в предложениях, и в шаблонах. При вложенных
конструкциях фрагменты могут пересекаться; общий вспомогательный глагол или
подлежащее могут находиться за пределами текущего фрагмента.
В конструкции `take … to …` время и участник остаются в первом фрагменте,
а действие начинается с инфинитивного `to`: `How much time does it take you`
и `to prepare everything necessary`. Цепочки инфинитивов тоже разделяются:
`How long does it take` / `to study` / `to be a doctor`.
Последовательные вложенные действия также получают отдельные фрагменты:
`Do you know` / `how long it takes` / `to get there`. В `I don't mind` /
`your phoning him` притяжательное местоимение остаётся при герундии.
Пересечение сохраняется, когда действие само является подлежащим:
`Going there by car is very convenient` / `Going there by car`.
Проверка при сборке запрещает разрезать слово и оставлять собственный глагол
или его частицу за границей фрагмента; общий глагол может использоваться
несколькими фактами с опорой на исходное предложение.

У шаблона границы относятся к его строке `text`, включая placeholders.
Граница не может разрезать placeholder; формы подлежащего и вспомогательного
глагола при необходимости хранятся отдельными атрибутами слота. Сборщик
пересчитывает позиции при подстановках и проверяет совпадение с `predicate`.
Явно заданная необязательная часть может исчезнуть вместе с пустым значением
слота. Поиск первого подходящего слова и автоматический разбор грамматики
во время сборки не используются.

При полностью пропущенном глаголе `predicate` равен `none`, список его
фрагментов пуст: `The longer the flight…` не содержит `is`. Таких исходных
фактов 31. Оставшийся вспомогательный глагол или маркер сохраняется:
`So were we.` → `were` / `unknown`, `…I'd like her to.` → `to` / `treat`,
если `treat` назван ранее в том же предложении. Сведения из соседней story
не восстанавливаются.

`verb` связывает форму в тексте с `action`: `has been reading` →
`verb: reading`, `action: read`; `turned it off` → `verb: turned off`,
`action: turn off`. Частицы фразового глагола сохраняются в обеих формах,
обычные обстоятельства в `verb` не входят. Границы отделяют основной глагол
от вспомогательных и инфинитивного `to`; при пропуске основного глагола
`verb` равен `none`. Сохранившаяся глагольная форма в кратком ответе
используется буквально: `Yes, she does.` → `verb: does`, `action: unknown`.

`action` — нормализованное действие, обычно начальная форма глагола.
Фразовые глаголы сохраняют частицу: `turned off` → `turn off`,
`hadn't phoned back` → `phone back`. Семантическая нормализация также допускает
`There is…` → `exist` и `has got` → `have`.
Текущее явно выраженное положительное
намерение вроде `I want to give him a book.` сворачивается в `action=give`,
`intention=give`, `operation=add`, `predicate=to give`: предикат относится
к целевому действию. Обычное сообщение о действии имеет
`intention=none`, `operation=none`. `operation` сохраняет смысл изменения
набора намерений из `build_v4.py`, а не обозначает время действия.
`aspect`, `voice`, `polarity`, `status` описывают грамматику и статус
события независимо. `tense` хранится в исходной разметке, но не выводится
в обучающие вопросы. Дополнительные поля включают `recipient`, `location`,
`time`, `duration`, `modality` и другие роли исходной разметки.

Случайного выбора и добора до заданного размера нет: сборщик перебирает
все разрешённые комбинации. SHA-256 стабильного семейства распределяет примеры
между train/valid/test примерно в пропорции 80/10/10 по семействам.
Все вариации одного исходного примера остаются вместе. Совпавшие входные тексты
удаляются, их семейства объединяются; противоречащие ответы вызывают ошибку
до записи файлов. Повторный запуск с той же библиотекой даёт те же байты.

Предложения и ответы редактируются в `sentences.json`, допустимые шаблоны —
в `fact_templates.json`, варианты значений — в `slot_values.json`.
При загрузке сборщик восстанавливает исходный вариант каждого шаблона и
сравнивает текст и все ответы с записью в `sentences.json`. Расхождение вызывает
ошибку до записи датасета.

```shell
# Обычная сборка, в том числе после добавления или удаления предложений:
python build_book_srl.py
```

Синхронизация встроена в `build_book_srl.py` и работает с тремя JSON-файлами.
Она сохраняет существующие переменные шаблоны, создаёт фиксированный шаблон
для нового размеченного предложения и убирает шаблоны удалённых предложений.
Если каталог шаблонов отсутствует, он создаётся из размеченных предложений.
Разрешённые замены задаются явно.
Изменение текста или ответов существующего предложения требует согласованного
изменения его шаблона: синхронизация не подменяет ручную разметку догадками.
Проверка всего корпуса выполняется при каждой сборке до записи каталога и
датасета; ошибка разметки сохраняет прежние файлы. Неизменившийся каталог не
перезаписывается. `--library` задаёт каталог JSON.

Для сборки книги вместе с прежним синтетическим набором разметки:

```shell
python build_book_srl.py --annotate
```

Этот флаг дополнительно читает `templates.json`, `nouns.json` и `verbs.json`
из каталога `--library` и разворачивает их готовую ручную разметку. Он не
определяет SRL новых предложений автоматически. Набор содержит одно- и
многопредложные примеры с вопросами по предложениям и предикатам.
Результат записывается в подкаталог `annotations` выбранного `--out`.
Оба набора проверяются до записи файлов; всегда используются все 84 урока
и выводится покрытие. Без флага эти три дополнительных JSON не требуются.

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
