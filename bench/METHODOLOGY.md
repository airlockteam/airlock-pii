# How the benchmark is made, and what its numbers mean

This benchmark measures `airlock_pii` and three baselines on text that looks like the notes,
forms and chat messages of a bank or an insurer in Kazakhstan. It was built to catch
regressions and to show where a detector is weak. **It does not measure accuracy on real
documents:** it has none. Section 6 says which of its figures can be read as a result and
which only show that two parts of one library agree.

## 1. Running it

| Command | What it does |
|---|---|
| `make bench` | generates the data and evaluates the library, without and with its Russian NER |
| `make bench-smoke` | the first 200 samples, against the thresholds; this runs in CI |
| `make bench-compare` | the same data, five detectors, one set of tables |

`make bench-compare` is the one command behind the published table. From a clean checkout it
needs [uv](https://docs.astral.sh/uv/), GNU Make and a network connection, and does the rest:

1. fetches the two model files of the library's Russian NER (29 MB) and checks their SHA-256;
2. generates `bench/data/v1.jsonl`: 2,000 samples, seed 42;
3. builds the environment of the Presidio baselines from `bench/presidio/uv.lock`, about
   940 MB of downloads the first time, of which two spaCy models are 400 and 513 MB;
4. runs every detector and writes `bench/results/<date>_<detector>.json` for each, and
   `<date>_compare.md` and `<date>_compare.json` for the comparison.

The same seed gives the same data on every machine and on Python 3.11 to 3.14: every draw
goes through `random.Random.random()`, which Python promises to keep stable (`draw.py`). The
result files name the data by its SHA-256, so two runs can be seen to be of the same data.

**Two environments.** Presidio and spaCy bring some thirty packages that the product does not
need. They live in a project of their own, `bench/presidio/`, with its own `pyproject.toml`
and `uv.lock`, and are never installed with the library. A Presidio baseline
runs there as a separate process (`predict.py`) and hands what it found back as JSON;
`evaluate.py` does all the counting, in one place, for every detector alike.

## 2. The data

A sample is a **template** with its **slots** filled. A template is a sentence or a short
document written for this benchmark, for example a loan decision, a transfer, an HR form, a
contract, a call-centre note or an IT incident. Its slots are the values to be found
(`{IIN}`, `{PERSON}`) and the values that only look like one (`{order_no}`, `{amount}`).

- **Templates.** About 400 with at least one value to find, and about 100 negatives, in
  Russian, Kazakh, both mixed in one text, English, and Kazakh typed in Latin letters. Seven
  domains. Every template is used as often as any other.
- **Values.** Every value comes from `airlock_pii.synthetic`: IINs and BINs with a valid
  check digit, IBANs that pass mod-97, card numbers that pass Luhn, mobile numbers in eight
  formats, e-mail addresses under `example.com`, `.org`, `.net` and `.test`, secrets of
  thirteen kinds. A value that a text repeats keeps its form two times in three.
- **Negatives.** Every fifth sample holds nothing to find and much that looks as if it did:
  twelve digits with a wrong check digit, and with a right one but no word that calls them an
  IIN; sixteen digits that fail Luhn; eleven-digit order numbers; UUIDs, commit hashes and
  timestamps; streets named after people, cities, invented companies; ordinary words that end
  like a surname.
- **Names.** Kazakh and Russian, in shares that depend on the language of the text; of the
  names that the library's generator draws, one in three with a patronymic; in Latin letters
  in English text and now and then elsewhere. Of
  the Kazakh names one in four has a surname in the Kazakh style, an ancestor's name with no
  ending, and some of those are written surname first. Some templates hold a surname alone,
  in a list or after a label, as minutes and rosters do. The data marks each of these.
- **Held-out names.** One name in five is a given name and a surname that `airlock_pii` has
  on no list (`heldout.py`): rarer Kazakh names, names of the other peoples of Kazakhstan and
  its neighbours, and surnames of three kinds (with the ending `-ов`, `-ев` or `-ин`; in the
  Kazakh style; Korean, German, Ukrainian and others). A test fails if one of the given names
  turns up on a list of the library.
- **Names in a Russian case.** In a Russian template a name stands in the case the sentence
  asks for (`от Ивановой Марии`), by rules of school grammar written for this benchmark
  (`decline.py`) and not taken from the library. In a Kazakh or a mixed text the name stays in
  its base form, and the suffix that Kazakh would add is left out: that is not measured.
- **Names that open a sentence.** Marked where a name stands first in the text, a line or a
  sentence, where a capital letter says nothing.

Every share above is a choice. None is a count of how people are named or how often a kind
of text occurs.

**Two variants** of a sample are drawn on top, each from a stream of its own, so the text and
its values are what they would be without them (`variants.py`):

- **In capitals** (one sample in twelve): the whole text, names included. An e-mail address
  and a secret stay as typed.
- **OCR-like noise** (one in ten): the digit `0` and the capital letter `O`, and the digit `1`
  and the small `l`, swapped each way in one place in eight, and one broken space: a space
  put inside a word or a number, or lost between two words. **The label stays on a value that
  the noise touched**, moved and stretched with it: a reader still knows an IIN with one
  digit misread for an IIN, and it must not leave in clear.

## 3. Matching

A span is a type and two positions. A found span is compared with a labeled one **of the same
type**: an IIN reported as a BIN is a miss for IIN and a false positive for BIN. `scoring.py`
defines four ways of matching, and a span takes part in one match at most.

| Matching | A found span matches a labeled one if | Used for |
|---|---|---|
| Exact | type, start and end are the same | precision, recall and F1 in the result files |
| Overlap | the type is the same and they share half or more of what they cover together | precision, recall and F1 in the comparison |
| Covered | one found span of the value's type holds all of the value | "would it be masked"; the thresholds |
| Covered by any type | one found span holds all of the value, whatever it is called | detectors whose types are not ours |

- **Precision** = TP / (TP + FP), **recall** = TP / (TP + FN), **F1** = 2 TP / (2 TP + FP + FN).
  A ratio with nothing under the line is shown as `–`, not as 0 or 1.
- **Why overlap in the comparison.** Detectors draw the edges of a value differently. Presidio
  reports a phone number without its `+7`; exact matching would grade that convention as a
  miss and a false positive at once.
- **Why covered for masking.** A span that stops one character short leaves that character in
  clear, so it does not count. A span that is too long still hides the value, so it does. The
  result of a detector also says, for PERSON, how many names are found exactly and how many
  lie inside a longer span, which holds a word that is no part of the name.
- **False-positive rate on negatives**: the share of the negatives in which the detector
  reports anything of a type of the data.
- **Latency**: the time of one call for one sample, in one process, after a warm-up call; the
  median and the 95th percentile over the samples. One machine, no batching.

**Thresholds** (`make bench-smoke`): for IIN, BIN, KZ_IBAN, CARD, PHONE, EMAIL and SECRET at
least 0.99 of the values covered, and one of those types reported in at most 1% of the
negatives. PERSON is reported and not checked. **Samples with OCR-like noise are left out of
the thresholds**, values and negatives alike, and reported on lines of their own: a digit
misread as a letter turns twelve digits into something no detector of digits finds, and the
noise would decide the verdict. Samples in capitals are typed text and stay in.

**Slices.** A result also holds the covered share on parts of the data: the samples as typed,
in capitals and with noise; within the noisy ones, the values the noise touched and those it
left alone; and, counted over the samples as typed so that each says one thing, Kazakh in
Latin letters and each kind of name the data marks.

## 4. The detectors

| Detector | What it is |
|---|---|
| `airlock_pii` | the library as `pip install airlock-pii` gives it: every recognizer but the NER |
| `airlock_pii_ner` | the same with its Russian NER (slovnet) switched on |
| `naive_regex` | one regular expression per type, nothing validated: any twelve digits are an IIN |
| `presidio_default` | Presidio's `AnalyzerEngine()` with no arguments, asked for English |
| `presidio_ru` | Presidio with spaCy's `ru_core_news_lg`, asked for Russian |

**Presidio** is [presidio-analyzer](https://github.com/data-privacy-stack/presidio) 2.2.364
with spaCy 3.8.16; each result file records the versions, the model, the list of recognizers
and the mapping it ran with.

- `presidio_default` is what its documentation's first example gives: the English model
  `en_core_web_lg` 3.8.0, the default recognizers, the language `en` for every text.
- `presidio_ru` uses Presidio's own NLP configuration with the model replaced by
  `ru_core_news_lg` 3.8.0, and the predefined recognizers that Presidio loads for a language
  it has no special list for. **One thing is added by hand: the credit-card recognizer is
  registered for Russian.** Presidio's default list names it for English, Spanish, Italian and
  Polish only, so without that a card number would be missed by an accident of configuration.
  A fair baseline has it.
- **Types.** Presidio's `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD` and
  `IBAN_CODE` map to ours. It has no type for an IIN, a BIN or a secret, and what a detector
  cannot report counts as missed. What it reports under any other type (a date, a place, a
  number it takes for an American bank account) is neither a hit nor a false positive; the
  matching "covered by any type" counts those spans too.
- **Not tuned.** The phone recognizer keeps its default regions, which do not include
  Kazakhstan. No recognizer was written for an IIN or a BIN. A Presidio set up for these
  values would do better; this benchmark measures what its documentation gives.
- **E-mail under `.test`.** Presidio accepts an address only under a public suffix. Two of
  the generator's five domains are under `.test`, which is reserved for testing, so Presidio
  reports none of those. The comparison shows EMAIL for the two kinds of domain apart.

## 5. Reading the tables

The comparison has three tables: recall, precision and F1 by type (overlap), with the
false-positive rate and the latency; the covered share by type, with PERSON split into
held-out names and the library's own; and the covered share by slice. The notes under them
are part of the result and are written every time.

## 6. Which figures are circular

A figure is circular when the thing that made the data and the thing that is measured are
the same code or the same lists. It then shows that the two agree and nothing more.

| Figure, for `airlock_pii` | Circular? | Why |
|---|---|---|
| IIN, BIN, KZ_IBAN, CARD: recall | **yes** | the generator and the validators are one library: a value is made to pass the check that finds it |
| PHONE, EMAIL: recall | **yes** | the formats are the generator's own; a format it does not write is not in the data |
| SECRET: recall | **yes** | the thirteen kinds are the generator's, and the recognizer was written against them |
| PERSON, names from the library's lists | **yes** | every given name the generator draws is on a list the recognizer reads |
| PERSON, held-out names | **no** for the names | the library has never seen them. The templates around them were still written by the hand that wrote the rules |
| False positives on negatives | partly | the look-alikes are built to fail the checks; the texts were written by the hand that wrote the rules |
| Values that OCR-like noise touched | **no** | the noise knows nothing of the detector |
| Any figure of a baseline | not circular, **not neutral** | the data was not made with the baseline, and it was made with the library's formats and for the library's types |
| Latency | **no** | measured, on one machine |

So the recall of 1.000 that the library reaches on structured values as typed is a check that
the generator and the validators agree. It is the reason the thresholds exist, and it is no
claim about documents.

## 7. Limitations

- **Synthetic data only.** No real document, customer record or real person's data was used,
  and none may be: every value is generated.
- **The templates were written with a language model**, by the same hand as the detection
  rules, and reviewed by one person. They are plausible business text and no sample of it.
  A rule and a template can share a blind spot.
- **No native speaker has checked the Kazakh text**, in Cyrillic or in Latin letters, or the
  word lists of the library.
- **News-style models, business-style text, and neither is the real thing.** The NER of the
  library (slovnet) and spaCy's Russian model were trained on news. The benchmark's text
  imitates notes and forms. Real business text, with its abbreviations, tables, typos and
  code-switching, differs from both.
- **Names are drawn from short lists.** About 125 given names and 105 surnames of the
  library's generator, and about 60 given names and 105 surnames held out. Real names are far
  more varied.
- **The held-out slice is small**: a few hundred names, of one author's choosing.
- **Kazakh case suffixes are not written.** A name in a Kazakh sentence stands in its base
  form, without the suffix the sentence would add.
- **The noise is a model of OCR, not OCR.** Two confusions and one broken space.
- **The baselines are not tuned**, as section 4 says. The comparison is with what a team
  gets from the documentation, not with the best each tool can do.
- **Latency is not throughput.** One text at a time, one process, one machine.
- **Types the benchmark does not have**: addresses, identity documents, dates of birth. A
  detector is neither rewarded nor punished for finding them.
