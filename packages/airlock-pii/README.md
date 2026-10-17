# airlock-pii

Built by [Airlock](https://example.com/airlock).

Detects Kazakhstan personal data in text: IIN and BIN, KZ IBAN, payment cards, +7 phone numbers,
e-mail addresses and names of people, in Russian, Kazakh, mixed and English text. And secrets:
API keys, tokens, private keys, database addresses with a password in them, passwords.

- Pure Python, no dependencies. Python 3.11 or later. Apache-2.0.
- An IIN, a BIN, an IBAN and a card are held to their check digits (mod 11, mod-97, Luhn),
  and a check that is weak on its own is never the only evidence.
- Kazakhstan first. Recognizers are grouped by country, so that others can be added.
- **Alpha.** The first public release. The word lists for names have not been checked by a
  native speaker of Kazakh, and the benchmark is synthetic: see [Benchmark](#benchmark).

Кратко по-русски: [ниже](#кратко-по-русски).

## Quickstart

```
pip install airlock-pii
```

```python
>>> from airlock_pii import detect, mask_preview
>>> text = (
...     "Клиент с ИИН 009900000008 (ТОО, БИН 700140000019) просит перевести деньги "
...     "на счёт KZ73 9990 0000 0001 2345 с карты 4111 1111 1111 1111. "
...     "Телефон +7 (000) 123-45-67, почта client@example.com."
... )
>>> print(mask_preview(text))
Клиент с ИИН [IIN] (ТОО, БИН [BIN]) просит перевести деньги на счёт [KZ_IBAN] с карты [CARD]. Телефон [PHONE], почта [EMAIL].
>>> for entity in detect(text):
...     print(entity.type, entity.start, entity.end, entity.confidence, entity.source)
IIN 13 25 0.8 kz.iin_bin
BIN 36 48 0.95 kz.iin_bin
KZ_IBAN 82 106 0.99 kz.iban
CARD 115 134 0.95 common.card
PHONE 144 162 0.85 kz.phone
EMAIL 170 188 0.95 common.email
>>> text[13:25]
'009900000008'

```

`detect()` returns `Entity` objects sorted by position, no two of them overlapping. An `Entity`
holds positions in the text and never the text itself, so logging one cannot leak a value:

```python
>>> detect(text)[0]
Entity(type=<EntityType.IIN: 'IIN'>, start=13, end=25, confidence=0.8, source='kz.iin_bin', subtype=None)

```

`mask_preview()` only shows what was found. It is not masking: the markers are not numbered and
nothing can be restored from them.

Every value in this README is made up, and none can be a real one: the IIN has month 99, the
BIN registration year 70, the IBAN bank code 999 and the phone number the unused code 000; the
card is a published test number and the address is on a domain reserved for examples. The names
further down are common given names put together with common surnames, and no one in
particular.

## What is detected

| Type | What counts | Confidence |
|---|---|---|
| `IIN` | 12 digits with the right check digit and a birth date in the first six | 0.95 |
| `BIN` | 12 digits with the right check digit and the structure of a BIN | 0.95 |
| `IIN`, `BIN` | the right check digit and neither structure, after an IIN word (`ИИН`, `ЖСН`, `IIN`, and also `ИНН`, `РНН`, `СТН`) or a BIN word (`БИН`, `БСН`, `BIN`) | 0.8 |
| `IIN`, `BIN` | a wrong check digit after one of those words: most likely a mistyped number | 0.6 |
| `KZ_IBAN` | a Kazakhstan IBAN that passes the mod-97 check, with or without spaces, in any case | 0.99 |
| `KZ_IBAN` | the same shape with a wrong check, after `IBAN`, `ИИК`, `счёт`, `есепшот` and the like | 0.6 |
| `CARD` | 13 to 19 digits that pass the Luhn check and begin with 2 to 6, in a row or grouped the way cards are printed | 0.9 |
| `CARD` | after `карта`, `card`, `Visa` and the like: any first digit and any grouping | 0.95 |
| `PHONE` | `+7` or `8` and ten digits, with brackets, spaces or dashes; subtype `kz` when the national number starts with 6 or 7 | 0.85 |
| `PHONE` | a bare `7` and ten digits that start with 6, 7 or 9, as in `wa.me/77xxxxxxxxx` | 0.75, and 0.85 after a phone word or `wa.me/` |
| `EMAIL` | `local-part@domain`, a pragmatic subset of RFC 5322 | 0.95 |
| `PERSON` | a given name with a surname or a patronymic, in either order; a surname with initials | 0.9 |
| `PERSON` | one of those words next to another capitalized word; a name alone next to a form of address (`уважаемый`, `мырза`); the capitalized words after the field of a form that holds a name (`Фамилия: Шмидт`, `ФИО:`, `Surname:`) | 0.8 |
| `PERSON` | a surname or a patronymic alone, right after a word for a person (`клиент`, `гражданин`), also with a colon between (`Ответственный: Петров`), or before a dash and such a word (`Иванов — председатель`) | 0.7 |
| `PERSON` | a surname alone in the middle of a sentence or in a list of names, in Cyrillic letters (`заявление от Петровой`, `Участники: Иванов, Петров`); subtype `surname_only` | 0.7 |
| `PERSON` | a given name alone | 0.6 |
| `PERSON` | with the NER switched on: what the model tags and the word lists do not rule out | 0.7 or 0.8 |
| `SECRET` | a key with a shape of its own: AWS key id, OpenAI, Anthropic, Google, GitHub, GitLab, Slack, a JWT, a private-key block, a URI with `user:password@` | 0.99 |
| `SECRET` | a token of 32 characters or more after `Bearer`; forty characters of base64 under an AWS secret name | 0.9 |
| `SECRET` | what follows a word for a password and `=` or `:` (`password`, `пароль`, `құпиясөз`), if it is no word, no number and no stand-in for a value | 0.8 |
| `SECRET` | the same right after the word, without a separator: eight characters or more with a letter and a digit | 0.7 |
| `SECRET` | sixteen characters or more with a letter and a digit under a name for a key (`api_key`, `client_secret`, `token` ...) | 0.8 |
| `SECRET` | three or four digits after `CVV`, `CVC`, `PIN` or `ПИН` | 0.9 |
| `SECRET` | with internal domains named: a host or a URL under one of them | 0.9 |

About one random 12-digit number in ten has a valid check digit, so the check digit alone is
never enough for an IIN or a BIN: the number also needs a birth date, a BIN structure, or a word
in front of it that says what it is. The Luhn check of a card is as weak, which is why a card
without a card word has to begin like a payment card and be written like one.

When the digits around a card can be read as a card in two ways, one `CARD` span covers both
readings. A few digits too many are masked, and none of the card's are left out.

The rules of each recognizer, and their sources, are in its module under
`airlock_pii/recognizers/`.

## Secrets

A `SECRET` says in its `subtype` what kind it is: `openai_api_key`, `github_token`, `jwt`,
`private_key`, `connection_string`, `password`, `generic_assignment`, `card_code` and so on.
A caller will rather refuse a text with a secret in it than mask it, so the rules are written
to report what a reader would call a credential and to leave the rest alone: a word after
`пароль:`, a date, `${DB_PASSWORD}`, `os.getenv("DB_PASSWORD")`, a UUID, a git SHA and the
payload of a data URI are no secrets.

```python
>>> import random
>>> from airlock_pii import synthetic
>>> token = synthetic.secret(random.Random(7), kind="github_token")
>>> database = synthetic.secret(random.Random(7), kind="connection_string")
>>> report = f"В логе токен {token}, строка подключения {database}, пароль: Qwerty123."
>>> print(mask_preview(report))
В логе токен [SECRET], строка подключения [SECRET], пароль: [SECRET].
>>> [entity.subtype for entity in detect(report)]
['github_token', 'connection_string', 'password']

```

The key and the address above are made up while the example runs: no README should hold
something a secret scanner would report. `synthetic.secret()` makes one of each kind.

Hosts inside your network are secrets only if you say which they are. List the domains in
`AIRLOCK_PII_INTERNAL_DOMAINS` (`corp.local,bank.internal`), and a host name or a URL under one
of them is a `SECRET` of kind `internal_host`.

The patterns for keys with a prefix are adapted from the rules of
[gitleaks](https://github.com/gitleaks/gitleaks) (MIT); see the `NOTICE` file.

## Names

```python
>>> names = (
...     "Заёмщик — Айгерім Ахметова, поручитель Нурлан Сакенулы. "
...     "Хатты Айгерімге жіберіңіз, көшірмесін Ивановой Марии. Кеңсе Абай көшесінде."
... )
>>> print(mask_preview(names))
Заёмщик — [PERSON], поручитель [PERSON]. Хатты [PERSON]ге жіберіңіз, көшірмесін [PERSON]. Кеңсе Абай көшесінде.

```

Names are found by rules, with no model (`recognizers/kz/person.py`). A name is told from the
other capitalized words by what its words are: a given name of the word lists, a word with the
ending of a surname or a patronymic (`-ов`, `-ева`, `-ұлы`, `-овна`), an initial.

- **The word lists are unverified.** About 510 Kazakh and 190 Russian given names, written
  from general knowledge and not yet checked by a native speaker. They are plain text files in
  `airlock_pii/data/`, with a README.
- **Kazakh letters or not.** `Айгерім`, `Айгерим` and `Айгерiм` with a Latin `i` are one name:
  words are looked up by `airlock_pii.validators.name_key`, which writes the nine Kazakh
  letters as the Russian letters typed in their place. Names in Latin letters are found too
  (`Aigerim Akhmetova`, `Ysqaq`), and next to a given name a surname in the Kazakh style is
  known by its ending there (`Arman Zhumabay`, `Dana Seitzhan`). A given name that is also
  an ordinary word needs more than its capital in Latin letters: `Qairat`, which is also
  energy, is found after a greeting (`Dear Qairat`) or with a surname, and not alone.
- **Case forms.** A Russian case form is reported whole (`Ивановой Марии`). A Kazakh case suffix
  is left outside the span: the name in `Айгерімге` is `Айгерім`. Which suffixes can follow a
  name is a public function, for a caller that knows a name and looks for it again:
  `airlock_pii.validators.kazakh_case_suffixes(name)` (see below).
- **Surnames with no ending.** A surname in the Kazakh style is the name of an ancestor as it
  is (`Сейтқали`, `Жұмабай`, `Оспан`), with nothing of `-ов` or `-ев` to know it by. It is
  found next to a given name, in either order: `Айгүл Сейтқали`, `Сейтқали Айгүл`. At the
  start of a sentence, where every word has a capital, the word before a given name is the
  surname only if it looks like a name: a given name of the lists, which most such surnames
  are, or a word that ends as Kazakh given names do (`-бай`, `-бек`, `-қали`). `Дочь Айгерим`
  and `Водитель Айгүл` are a daughter and a driver, and the name is found without them.
- **A surname alone** in a form with an ending (`Петровой`, `Ахметову`, `Ахметовқа`) is
  reported in the middle of a sentence, where its capital says that a name is meant: after a
  word in lower case, after a comma or a dash, or after the word that opened the sentence.
  Its entity has the subtype `surname_only`, because the claim rests on an ending and a
  capital and on nothing else. A caller that hides every repeat of a name may want to look
  for this one with its capital only. Towns that end like a surname are a list (`в Ростов`,
  `из Пушкина`), and so are ordinary words that people are called as well (`Калина`,
  `Рябина`, `Родина`): alone such a word is the word, and next to a given name it is a
  surname (`Анна Калина`). A word that nobody is called (`Телятина`, `Адреналин`) is on the
  list of what is no name. Neither list can be complete.
- **The base form of a surname** (`Петров`, `Сергеев`) is also the genitive plural of a noun,
  which a contract writes with a capital (`от Застройщиков`, `по заявлению Дольщиков`). Alone
  it is a person only where a subject can stand: after a dash, after a word such as `что`,
  `вчера` or `кеше`, in a Kazakh sentence, or in a list of names (`Айгерим, Петров и
  Сидоров`; at the start of a sentence too, `Иванов, Петров и Сидоров приглашены`). After a
  preposition or a noun it is not reported: `наш сосед Петров` is missed, and `клиент
  Петров` and `подписал Петров` are found, because a word for a person or a verb of the
  lists stands before the name.
- **Not a person:** a name next to a street or place word, on either side (`улица Абая`, `Абай
  көшесі`), next to a form of organization (`ТОО «Нұрлан»`, `«Нұрлан» ЖШС`), and a city.
  After `ИП` and `ЖК` comes the owner's own name, which is a person.
- **Names that are also words.** `Арман` is a dream and `Вера` is faith. Alone at the start of a
  sentence such a name is not reported; with a surname, a greeting or a form of address it is,
  and so it is next to a capitalized word of which nothing is known (`Мақсат Сейтқали жазды`,
  but not `Мақсат Банк ашты`).
- **A request opens with a verb.** In `Напиши Ахметову письмо` the verb is no part of the name,
  and the surname after it is a person. The verbs are a list too (`data/context_words.txt`).

```python
>>> more = (
...     "Мақсат Сейтқали жазды. Сейтқали Айгүл ертең келеді. "
...     "Заявление от Петровой принято, а в Ростов летим завтра."
... )
>>> print(mask_preview(more))
[PERSON] жазды. [PERSON] ертең келеді. Заявление от [PERSON] принято, а в Ростов летим завтра.
>>> [entity.subtype for entity in detect(more)]
[None, None, 'surname_only']
>>> from airlock_pii.validators import kazakh_case_suffixes, name_key
>>> name_key("ге") in kazakh_case_suffixes("Айгерім")    # Айгерімге, Айгеримге
True
>>> name_key("ты") in kazakh_case_suffixes("Алма")       # Алматы is no Алма in a case
False

```

- **Minutes and rosters.** A list of surnames is its own evidence where it can be seen to
  begin: after a colon (`Участники: Иванов, Петров, Сидоров`), at the start of a sentence, as
  items of a list on lines that follow each other (`- Иванов`, `- Петров`), or as a column
  under a line that ends in a colon. One word after a colon stays what it was (`Статус:
  Одобрен`). A town that ends like a surname is a surname in a list that also holds a
  surname which is no town; `Филиалы: Ростов, Саратов` stays two towns, and `Ингредиенты:
  Калина, Рябина, Осина` three berries.
- **Labels.** A word for a person announces a surname before a colon (`Ответственный:
  Петров`, `Секретарь собрания: Ахметова`) and after a dash (`Иванов — председатель`). After
  the field of a form that holds a name (`Фамилия:`, `ФИО:`, `Тегі:`, `Аты-жөні:`,
  `Surname:`, `Full name:`) the capitalized words are a name whatever they are: `Фамилия:
  Шмидт`. Not after `Имя:`, `Аты:` or `Name:`, which also name files and products, and not
  where the form points to the name with an abbreviation and its full stop (`ФИО: См.
  приложение`, `Фамилия: Стр. 2`).

What the rules do not find: a surname alone at the start of a sentence (`Петрова подала
заявление`), in its base form after a noun (`наш сосед Петров`), or with an ending that many
words share (`письмо для Ковалевского`), a name of
which the lists know nothing and which has no telling ending (`Ханс Шмидт`), a name in Latin
letters whose given name is also a word, next to a surname that neither ends as names do nor
stands on a list (`Arman Schmidt`; `Arman Zhumabay` is found, by the ending of the surname),
and names written in lower case.

## The Russian NER (optional)

A model finds the first two of those. The `ner-ru` extra runs slovnet's NER for Russian (natasha
project, MIT; numpy only, no torch) next to the rules, as the recognizer `common.ner_ru`:

```
pip install "airlock-pii[ner-ru]"
python -m airlock_pii models download --dir /opt/airlock/models
export AIRLOCK_PII_NER=ru AIRLOCK_PII_MODELS_DIR=/opt/airlock/models
```

In a checkout of the repository `make models` is the second command, for the folder `models/`.

- **The model files** (29 MB) are fetched once by the second command, which is the only thing in
  the library that uses the network. Their SHA-256 is in the package and is checked after the
  download and again before every load. `python -m airlock_pii models verify` checks a
  directory, and `download --from-wheel` takes the files out of a `natasha` wheel on disk, for
  a site with a package mirror and no other way out.
- **On means on.** With `AIRLOCK_PII_NER=ru`, a missing file, a changed file or a missing extra
  raises `airlock_pii.models.ModelError`. `detect()` never answers without the model it was
  asked to use. A service calls `airlock_pii.load_models()` when it starts, so that it fails
  there and not on the first text.
- **Cost.** About 120 MB of memory for the model, and about 4 ms for every kilobyte of text on
  one core, against 0.3 ms without it. Text of any length is read in chunks, so the memory does
  not grow with it. Set `OPENBLAS_NUM_THREADS=1`: with numpy's default of one thread per core
  the model is several times slower on Linux.
- **Kazakh.** The model was trained on Russian news. It reads a copy of the text with the Kazakh
  letters replaced by their Russian look-alikes, and what it tags is checked against the word
  lists, because on its own it takes many words of a Kazakh sentence for names.

## Options

```python
>>> from airlock_pii import EntityType
>>> [str(entity.type) for entity in detect(text, types=[EntityType.PHONE, EntityType.EMAIL])]
['PHONE', 'EMAIL']
>>> [str(entity.type) for entity in detect(text, min_confidence=0.9)]
['BIN', 'KZ_IBAN', 'CARD', 'EMAIL']
>>> [str(entity.type) for entity in detect(text, countries=[])]
['CARD', 'EMAIL']

```

- `types` limits the entity types. Recognizers that cannot emit them do not run.
- `countries` limits the country-specific recognizers (ISO codes such as `"kz"`). Country-neutral
  ones, cards, e-mail and secrets, always run.
- `min_confidence` drops what is less certain.

An argument that could only make detection find nothing raises `ValueError`: an unknown type, a
country no recognizer has, a `min_confidence` outside 0 to 1.

`detect()` returns one entity for each character at most. `candidates()` takes the same
arguments and returns every claim of every recognizer, before overlaps are settled; `detect()`
is `airlock_pii.engine.resolve_overlaps()` of exactly those. Use the candidates to hide every
character any recognizer pointed at, or to drop claims by a rule of your own first:

```python
>>> from airlock_pii import candidates
>>> from airlock_pii.engine import resolve_overlaps
>>> resolve_overlaps(candidates(text)) == detect(text)
True

```


## Adding a recognizer

A recognizer is any object with a `name`, a `country` (an ISO code such as `"kz"`, or `None`
for one that belongs to no country), the `entity_types` it can emit, and `analyze(text)`,
which returns entities with positions in the text. Give yours to an engine next to the
built-in ones:

```python
>>> import re
>>> from airlock_pii import Entity, EntityType
>>> from airlock_pii.engine import Engine
>>> from airlock_pii.recognizers import default_recognizers
>>> class SpelledOutEmail:
...     """An address written for people and not for harvesters: ``name (at) example.com``."""
...
...     name = "my.email_spelled_out"
...     country = None
...     entity_types = frozenset({EntityType.EMAIL})
...     pattern = re.compile(r"[\w.+-]+ ?\(at\) ?[\w-]+(?:\.[\w-]+)+")
...
...     def analyze(self, text):
...         return [
...             Entity(EntityType.EMAIL, match.start(), match.end(), 0.8, self.name)
...             for match in self.pattern.finditer(text)
...         ]
>>> engine = Engine([*default_recognizers(), SpelledOutEmail()])
>>> note = "Пишите на client (at) example.com, ИИН 009900000008."
>>> [(str(entity.type), entity.source) for entity in engine.detect(note)]
[('EMAIL', 'my.email_spelled_out'), ('IIN', 'kz.iin_bin')]

```

The engine settles overlaps between your claims and the built-in ones by confidence, then by
length. It refuses a recognizer that returns a span outside the text or a type it did not
declare. An entity never holds text, and neither should a `source` or a `subtype`.

To add one to the library itself, a country's numbers for instance, see
[CONTRIBUTING.md](https://github.com/airlockteam/airlock-pii/blob/main/CONTRIBUTING.md):
country-specific recognizers live in `airlock_pii/recognizers/<country>/`, and each comes with
its generator in `airlock_pii.synthetic`, its tests and its templates in the benchmark.

## With Presidio

For users of [Presidio](https://github.com/data-privacy-stack/presidio): the `presidio` extra
brings `airlock_pii.presidio_adapter.EngineRecognizer`, a Presidio recognizer that finds what
`detect()` finds.

```
pip install "airlock-pii[presidio]"
```

```python
>>> from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
>>> from presidio_analyzer.nlp_engine import NoOpNlpEngine
>>> from airlock_pii.presidio_adapter import EngineRecognizer
>>> registry = RecognizerRegistry(supported_languages=["ru"])
>>> registry.add_recognizer(EngineRecognizer(supported_language="ru"))
>>> analyzer = AnalyzerEngine(
...     registry=registry,
...     nlp_engine=NoOpNlpEngine(models=[{"lang_code": "ru", "model_name": "none"}]),
...     supported_languages=["ru"],
... )
>>> message = "Клиент с ИИН 009900000008, телефон +7 (000) 123-45-67, почта client@example.com."
>>> for result in sorted(analyzer.analyze(text=message, language="ru"), key=lambda r: r.start):
...     print(result.entity_type, result.start, result.end, result.score)
KZ_IIN 13 25 0.8
PHONE_NUMBER 35 53 0.85
EMAIL_ADDRESS 61 79 0.95

```

- This analyzer has the library's recognizers and no others, so it needs no language model
  of spaCy. In an analyzer of your own, with Presidio's recognizers and a model, the one line
  is `analyzer.registry.add_recognizer(EngineRecognizer(supported_language="ru"))`.
- **Names of entities.** What Presidio has a name for keeps it: `PERSON`, `EMAIL_ADDRESS`,
  `PHONE_NUMBER`, `CREDIT_CARD`, `IBAN_CODE`. An IIN is `KZ_IIN`, a BIN `KZ_BIN`, a secret
  `SECRET`. `entity_names=` changes any of them.
- **Languages.** A Presidio recognizer has one language, and the library reads Russian, Kazakh
  and English alike: register one `EngineRecognizer` for each language you ask Presidio for.
- **Overlaps.** The one recognizer returns what `detect()` returns, with the library's own
  overlaps settled. Next to Presidio's recognizers a value that both find under the same name
  is one finding.
- Tested with presidio-analyzer 2.2.364. Presidio asks for a numpy below 2.5, so with both
  extras installed the Russian NER runs on that numpy; its tests run on it too.

## Benchmark

Five detectors on 2,000 synthetic samples: the library as it is installed (`airlock_pii`) and
with the Russian NER (`airlock_pii_ner`), plain regular expressions (`naive_regex`), Presidio
as its documentation sets it up (`presidio_default`) and Presidio with spaCy's largest Russian
model (`presidio_ru`). **The notes under the tables are part of the result**: most of the
figures grade the library on data that its own generator made.

<!-- benchmark:begin -->
From [`bench/results/2026-10-09_compare.md`](https://github.com/airlockteam/airlock-pii/blob/main/bench/results/2026-10-09_compare.md). Data: `v1.jsonl`, 2,000 samples of which 400 are negatives, SHA-256 `3e003c8ec12a2b42`. Run on 2026-10-09.
Of the samples, 1,623 are as typed, 167 in capitals and 210 with OCR-like noise.

### Recall / precision / F1 by type

A found span matches a labeled one of its type if the two share half or more of what they cover together. A cell is recall / precision / F1; `–` where there is nothing to compute. Every row is over all the samples, those in capitals and those with noise among them; the last table takes them apart.

| Type | Values | `airlock_pii` | `airlock_pii_ner` | `naive_regex` | `presidio_default` | `presidio_ru` |
|---|---:|---:|---:|---:|---:|---:|
| PERSON | 1,304 | 0.883 / 0.992 / 0.934 | 0.940 / 0.984 / 0.962 | 0.000 / – / 0.000 | 0.516 / 0.330 / 0.403 | 0.597 / 0.525 / 0.559 |
| IIN | 528 | 0.930 / 0.994 / 0.961 | 0.930 / 0.994 / 0.961 | 0.930 / 0.649 / 0.765 | 0.000 / – / 0.000 | 0.000 / – / 0.000 |
| BIN | 145 | 0.938 / 1.000 / 0.968 | 0.938 / 1.000 / 0.968 | 0.000 / – / 0.000 | 0.000 / – / 0.000 | 0.000 / – / 0.000 |
| KZ_IBAN | 348 | 0.963 / 1.000 / 0.981 | 0.963 / 1.000 / 0.981 | 0.557 / 1.000 / 0.716 | 0.902 / 1.000 / 0.949 | 0.902 / 1.000 / 0.949 |
| CARD | 248 | 0.903 / 1.000 / 0.949 | 0.903 / 1.000 / 0.949 | 0.895 / 0.860 / 0.877 | 0.794 / 1.000 / 0.885 | 0.794 / 1.000 / 0.885 |
| PHONE | 424 | 0.965 / 0.998 / 0.981 | 0.965 / 0.998 / 0.981 | 0.965 / 0.958 / 0.961 | 0.604 / 0.496 / 0.545 | 0.604 / 0.496 / 0.545 |
| EMAIL | 290 | 0.993 / 0.997 / 0.995 | 0.993 / 0.997 / 0.995 | 0.997 / 0.923 / 0.959 | 0.634 / 0.979 / 0.770 | 0.634 / 0.979 / 0.770 |
| SECRET | 164 | 0.970 / 0.994 / 0.981 | 0.970 / 0.994 / 0.981 | 0.000 / – / 0.000 | 0.000 / – / 0.000 | 0.000 / – / 0.000 |
| **All types** | 3,451 | 0.925 / 0.995 / 0.959 | 0.947 / 0.992 / 0.969 | 0.465 / 0.824 / 0.595 | 0.471 / 0.499 / 0.485 | 0.501 / 0.641 / 0.562 |
| Negatives with a false positive | 400 | 1.2% | 2.2% | 42.0% | 60.5% | 38.5% |
| Negatives with a span of another type | 400 | 0.0% | 0.0% | 0.0% | 92.5% | 57.2% |
| Latency per sample, p95 |  | 0.19 ms | 1.54 ms | 0.03 ms | 9.55 ms | 11.35 ms |

### Would the value be masked?

The share of the values that one span of their type holds whole. A span that stops short of the last character does not count; one that is too long does. Over all the samples, but for the two rows that say otherwise.

| Type | Values | `airlock_pii` | `airlock_pii_ner` | `naive_regex` | `presidio_default` | `presidio_ru` |
|---|---:|---:|---:|---:|---:|---:|
| PERSON | 1,304 | 0.873 | 0.928 | 0.000 | 0.482 | 0.521 |
| PERSON as typed: names the library has on no list | 181 | 0.508 | 0.829 | 0.000 | 0.591 | 0.652 |
| PERSON as typed: names from the library's own lists (circular) | 849 | 0.994 | 0.994 | 0.000 | 0.514 | 0.562 |
| IIN | 528 | 0.930 | 0.930 | 0.930 | 0.000 | 0.000 |
| BIN | 145 | 0.938 | 0.938 | 0.000 | 0.000 | 0.000 |
| KZ_IBAN | 348 | 0.963 | 0.963 | 0.557 | 0.902 | 0.902 |
| CARD | 248 | 0.903 | 0.903 | 0.895 | 0.794 | 0.794 |
| PHONE | 424 | 0.965 | 0.965 | 0.965 | 0.465 | 0.465 |
| EMAIL | 290 | 0.990 | 0.990 | 0.990 | 0.634 | 0.634 |
| SECRET | 164 | 0.963 | 0.963 | 0.000 | 0.000 | 0.000 |
| **All types** | 3,451 | 0.921 | 0.942 | 0.465 | 0.441 | 0.456 |
| All types, under a span of any type | 3,451 | 0.921 | 0.942 | 0.504 | 0.680 | 0.482 |

### Read this with the tables

1. **The results for IIN, BIN, KZ_IBAN, CARD, PHONE, EMAIL and SECRET are circular.** The generator of the data (`airlock_pii.synthetic`) and the validators that `airlock_pii` detects with come from the same library. A recall of 1.000 shows that the two agree, not how accurate the library is on real documents.
2. **PERSON is circular too, except on the held-out names.** Four names in five are drawn from lists that the library's own word lists hold in full. The row "names the library has on no list" is the one figure for names that does not grade the library on its own lists: given names it has never seen, with surnames that the generator of the library does not draw. The templates and the rules were still written by the same hand.
3. **The data is synthetic.** Every text is a template written for this benchmark, with the help of a language model, and filled with generated values. It reads like notes, forms and chat messages of a bank or an insurer; it is not real business text, and no real document was used. The Kazakh text has not been checked by a native speaker.
4. **What Presidio cannot detect counts as missed.** It has no type for an IIN, a BIN or a secret, so those rows are 0 by construction and not by failure. What it reports under a type that is none of ours (a date, a place, a number it takes for an American bank account) is neither a hit nor a false positive; the row "under a span of any type" counts those spans as well.
5. **Presidio was not tuned for Kazakhstan.** `presidio_default` is `AnalyzerEngine()` with no arguments: the English model, the language `en` for every text. `presidio_ru` is the same with spaCy's `ru_core_news_lg` and the language `ru`, and with the credit-card recognizer registered for Russian by hand, because Presidio's default list does not load it for that language. Its phone recognizer keeps its default regions, which do not include Kazakhstan. A Presidio with recognizers written for these values would do better.
6. **E-mail addresses under `.test`.** Two of the generator's five domains are under `.test`, which is reserved for testing. Presidio's e-mail recognizer accepts an address only under a public suffix and reports none of those. The table of slices shows EMAIL for the two kinds of domain apart.
7. **OCR-like noise is left out of the thresholds, and shown.** One sample in ten has digits and letters that look alike swapped, and a broken space. A value that the noise touched is still labeled, and most detectors of digits miss it.
8. **Latency** is the time of one call for one sample on the machine that made the table, in one process, after a warm-up call. Presidio's includes its language model.
<!-- benchmark:end -->

The third table (capitals, OCR-like noise, Kazakh in Latin letters, kinds of names) and what
the baselines ran with are in
[`bench/results/`](https://github.com/airlockteam/airlock-pii/tree/main/bench/results); how
the data is made and how a span is matched, in
[`bench/METHODOLOGY.md`](https://github.com/airlockteam/airlock-pii/blob/main/bench/METHODOLOGY.md).
`make bench-compare` makes the tables again from a clean checkout.

## Not in this version

- A password that is an ordinary word (`пароль: qwerty`), a passphrase of several words, a
  password that follows its word after other words and no separator (`пароль от почты
  Qwerty123`), and the values after the first in a list (`passwords: Qwerty123, Asdfgh456`).
- A key with no prefix of its own and no name in front of it: random characters alone are not
  reported, whatever their entropy.
- Names in lower case, and what the section on names lists as not found by the rules.
- A Kazakh NER. The model of the `ner-ru` extra knows Russian.
- IBANs of other countries. They are not reported, and their digits are not taken for a card.
- An IIN or a BIN split by spaces or dashes. They are written as twelve digits in a row.
- A card that begins with 0, 1, 7, 8 or 9 when no card word stands before it. Some national
  schemes use such numbers.

## Development

```
git clone https://github.com/airlockteam/airlock-pii && cd airlock-pii
make setup models     # the environment from uv.lock, and the model files of the NER
make check            # ruff, mypy, the tests, the benchmark's smoke run
```

The repository is an export of the one the library is developed in; a pull request is
welcome and is applied there. How that works, and how to report a vulnerability, is in
[CONTRIBUTING.md](https://github.com/airlockteam/airlock-pii/blob/main/CONTRIBUTING.md) and
[SECURITY.md](https://github.com/airlockteam/airlock-pii/blob/main/SECURITY.md).

## Кратко по-русски

`airlock-pii` находит в тексте персональные данные Казахстана и секреты:

- ИИН и БИН (с проверкой контрольного разряда), IBAN Казахстана, номера банковских карт,
  телефоны на +7, адреса электронной почты;
- имена людей в русском, казахском и смешанном тексте, в том числе казахские фамилии без
  окончаний и имена, набранные без казахских букв;
- ключи API, токены, приватные ключи, строки подключения с паролем, пароли.

```
pip install airlock-pii
```

Дальше `airlock_pii.detect(text)` возвращает найденные фрагменты: тип, позиции в тексте и
уверенность. Сам текст в результат не попадает, поэтому его можно писать в журнал. Примеры
выше на этой странице запускаются как тесты.

Ядро библиотеки написано на чистом Python и не имеет зависимостей. Дополнительно
подключаются русская модель NER для имён (`airlock-pii[ner-ru]`) и адаптер для Presidio
(`airlock-pii[presidio]`).

Что важно знать:

- это альфа-версия, первый публичный выпуск;
- бенчмарк синтетический: шаблоны и значения сгенерированы, реальных документов в нём нет, а
  результаты по ИИН, БИН, IBAN, картам, телефонам, почте и секретам «круговые» (генератор и
  проверки взяты из одной библиотеки);
- списки казахских имён и казахские тексты бенчмарка не проверены носителем языка;
- все значения в документации и тестах вымышлены.

## License

Apache-2.0. The patterns for keys with a prefix are adapted from gitleaks (MIT), and the
Russian NER runs slovnet's model with navec's embeddings (natasha project, MIT): see the
`NOTICE` file.
