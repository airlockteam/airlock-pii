# Data of airlock_pii

Word lists that the name recognizers read (`airlock_pii.recognizers._names`), and the lock of
the model files.

**Every word list here is unverified.** They were written from general knowledge of Kazakh
and Russian names and of business writing. None is taken from a register, a census, a
customer's data or any other source of real personal data, and no one who speaks Kazakh as a
first language has checked them yet. Treat a missing or a misspelled name as a bug.

| File | What it holds |
|---|---|
| `given_names_kk.txt` | Kazakh given names, male and female, in Kazakh spelling, with the Russian spellings that are more than a change of letters (`Әсел`, `Асель`). Among them are names of an older kind (`Оспан`, `Ысқақ`, `Қазыбек`), which today are surnames in the Kazakh style as often as given names, and thirty names in `-ғали` and `-қали`, whose plain spellings end like a Russian verb |
| `given_names_ru.txt` | Russian given names and their short forms (`Саша`) |
| `ordinary_words.txt` | The given names that are also ordinary words (`Арман`, `Мақсат`, `Вера`); alone at the start of a sentence they are not taken for a person. Its section `[calendar]` holds the weekdays and months among them (`Жұма`, `Наурыз`), which alone are a person only next to a greeting or a form of address |
| `name_endings.txt` | The endings of surnames and patronymics, in Cyrillic and Latin letters. Its section `[given-kk]` holds what Kazakh given names often end with (`-бай`, `-бек`, `-хан`): a word that ends so is no form of a word for a street or a person (`Көшербай` is not `көше`), and before a given name at the start of a sentence it is the surname (`Жұмабай Ерлан`). `[given-kk-latin]` has the same endings as Latin letters spell them (`-bay`, `-bek`, `-khan`): there a word that ends so is also the surname of a given name that is an ordinary word (`Arman Zhumabay`) |
| `latin.txt` | How each Cyrillic letter is written in Latin letters: passport practice and the official Kazakh Latin alphabet without diacritics |
| `context_words.txt` | Words for streets and places, forms of organization, words for a person, verbs that open a request (`напиши`, `ответь`, `write`), forms of address, greetings, the words that follow a weekday or a month when the day is meant (`күні`, `айында`, `мейрамымен`, `мерекесімен`, `күнгі`), the words that join the names of a list (`[and]`: `и`, `или`, `пен`), the words after which the subject of a sentence stands (`[before-a-subject]`: `что`, `вчера`, `кеше`), which is where a surname in its base form is a person, the words that head a name only with a colon after them (`[label]`: `Ответственный:`, `Присутствовали:`), the fields of a form that hold a name (`[name-label]`: `Фамилия:`, `ФИО:`, `Тегі:`, `Surname:`), after which the capitalized words are a name whatever they are, and the abbreviations that such a field holds where it points to the name instead (`[abbreviation]`: `См.`, `Стр.`), which with their full stop are no name |
| `places.txt` | Cities, regions and countries. Its section `[like-a-surname]` holds towns that end like a surname (`Ростов`, `Пушкин`): alone in the middle of a sentence such a word is the town, next to a given name it is a surname |
| `not_names.txt` | Ordinary words that end like a surname (`документов`, `деловой`, `машина`, `телятина`), the units of an organization (`департамент`), and words that open a sentence (`Сегодня`, `Кстати`, `Бүгін`), which are not the surname of the name after them. A word of these is no part of a name anywhere. Its section `[surname-too]` holds ordinary words that people are called as well (`калина`, `рябина`, `родина`): alone such a word is the word, next to a given name it is a surname |
| `models.lock` | Names, sizes and SHA-256 of the NER model files; a copy of `models.lock` at the root of the repository |

## Format

UTF-8 text, one entry per line. `#` starts a comment, `[name]` opens a section. In the lists
of given names a comma separates the spellings of one name, and a stem in brackets is the one
its oblique cases are built on (`Павел (Павл)`).

## How a name is looked up

Words are compared by `airlock_pii.validators.name_key`: in lower case, with `ё` as `е` and
the nine Kazakh letters as the Russian letters people type in their place. `Айгерім`,
`Айгерим` and `Айгерiм` (with a Latin `i`) are one entry. Russian case forms (`Нурлана`,
`Марии`) and Latin spellings (`Aigerim`, `Nurlan`) are worked out from the entries when the
lists are read, so neither is written down here.

## Adding a name

Add it in its usual spelling to the list of its language, in the section of its sex or in
both. If it is also an ordinary word, add it to `ordinary_words.txt` with its meaning. The
tests in `tests/test_data_files.py` check the format, that no name stands twice and that the
lists do not contradict each other.
