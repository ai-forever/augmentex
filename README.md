<p align="center">
    <a href="https://github.com/ai-forever/augmentex/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/ai-forever/augmentex/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2308.09435">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2308.09435-red">
    </a>
<!--     <a href="https://github.com/ai-forever/augmentex/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/ai-forever/augmentex-green">
    </a> -->
</p>

# Augmentex — a library for augmenting texts with errors
Augmentex introduces rule-based and common statistic (empowered by [KartaSlov](https://kartaslov.ru) project) 
approach to insert errors in text. It is fully described again in the [Paper](https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf)
and in this 🗣️[Talk](https://youtu.be/yFfkV0Qjuu0?si=XmKfocCSLnKihxS_).

## Contents
- [Contents](#contents)
- [Installation](#installation)
- [Implemented functionality](#implemented-functionality)
- [Usage](#usage)
    - [Word level](#word-level)
    - [Character level](#character-level)
- [Contributing](#contributing)
- [Usage](#usage)
- [Contributing](#contributing)
    - [Issue](#issue)
    - [Pull request](#pull-request)
- [References](#references)
- [Authors](#authors)

## Installation
```commandline
pip install augmentex
```

## Implemented functionality
We collected statistics from different languages and from different input sources. This table shows what functionality the library currently supports.

|             | Russian     | English     |
| -----------:|:-----------:|:-----------:|
| PC keyboard |      ✅     |      ✅     |
| Mobile kb   |      ✅     |      ❌     |

In the future, it is planned to scale the functionality to new languages and various input sources.

## Usage
🖇️ Augmentex allows you to operate on two levels of granularity when it comes to text corruption and offers you sets of 
specific methods suited for particular level:
- **Word level**:
  - _replace_ - replace a random word with its incorrect counterpart;
  - _delete_ - delete random word;
  - _swap_ - swap two random words;
  - _stopword_ - add random words from stop-list;
  - _split_ - add spaces between letters to the word;
  - _reverse_ - change a case of the first letter of a random word;
  - _text2emoji_ - change the word to the corresponding emoji.
- **Character level**:
  - _shift_ - randomly swaps upper / lower case in a string;
  - _orfo_ - substitute correct characters with their common incorrect counterparts;
  - _typo_ - substitute correct characters as if they are mistyped on a keyboard;
  - _delete_ - delete random character;
  - _insert_ - insert random character;
  - _multiply_ - multiply random character;
  - _swap_ - swap two adjacent characters.

### **Word level**
```python
from augmentex.word import WordAug

word_aug = WordAug(
    unit_prob=0.4, # Percentage of the phrase to which augmentations will be applied
    min_aug=1, # Minimum number of augmentations
    max_aug=5, # Maximum number of augmentations
    lang="rus", # supports: "rus", "eng"
    platform="pc", # supports: "pc", "mobile"
    random_seed=42,
    )
```

1. Replace a random word with its incorrect counterpart;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='replace')
# Съешь ещё этих мягких французских булок, дло выпей чаю.
```

2. Delete random word;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='delete')
# Съешь ещё французских булок, да выпей
```

3. Swap two random words;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='swap')
# Съешь ещё этих мягких французских булок, да чаю. выпей
```

4. Add random words from stop-list;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='stopword')
# Съешь да ещё этих во мягких это французских булок, да выпей чаю.
```

5. Adds spaces between letters to the word;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='split')
# С ъ е ш ь ещё этих мягких французских булок, д а в ы п е й чаю.
```

6. Change a case of the first letter of a random word;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='reverse')
# Съешь ещё этих мягких Французских булок, Да выпей Чаю.
```

7. Changes the word to the corresponding emoji.
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
word_aug.augment(text=text, action='text2emoji')
# Съешь ещё этих мягких французских булок, да выпей чаю.
```

### **Character level**
```python
from augmentex.char import CharAug

char_aug = CharAug(
    unit_prob=0.3, # Percentage of the phrase to which augmentations will be applied
    min_aug=1, # Minimum number of augmentations
    max_aug=5, # Maximum number of augmentations
    mult_num=3, # Maximum number of repetitions of characters (only for the multiply method)
    lang="rus", # supports: "rus", "eng"
    platform="pc", # supports: "pc", "mobile"
    random_seed=42,
    )
```

1. Randomly swaps upper / lower case in a string;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='shift')
# СъЕшь ещё этих мягКих фраНцузских булок, да выпей Чаю.
```

2. Substitute correct characters with their common incorrect counterparts;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='orfo')
# Съешь ещё этиз мягкех французских булок, ла тыпей саю.
```

3. Substitute correct characters as if they are mistyped on a keyboard;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='typo')
# Съель езё этих мягких французскпх булок, да аыпей чпю.
```

4. Delete random character;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='delete')
# Съеь щё эих мягких французскх булок, да выей чаю.
```

5. Insert random character;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='insert')
# Съешь ещё этих мягкцих фчранцэузскиьх булок, да выпей шчаю.
```

6. Multiply random character;
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='multiply')
# Съешь ещё этих мяггких французских булок, даа выпей чаю.
```

7. Swap two adjacent characters.
```python
text = "Съешь ещё этих мягких французских булок, да выпей чаю."
char_aug.augment(text=text, action='swap')
# Съешь ещёэ тихм якгих французских буолк, ад выпей чаю.
```

## Contributing
### Issue
- If you see an open issue and are willing to do it, add yourself to the performers and write about how much time it will take to fix it. See the pull request module below.
- If you want to add something new or if you find a bug, you should start by creating a new issue and describing the problem/feature. Don't forget to include the appropriate labels.

### Pull request
How to make a pull request.
1. Clone the repository;
2. Create a new branch, for example `git checkout -b issue-id-short-name`;
3. Make changes to the code (make sure you are definitely working in the new branch);
4. `git push`;
5. Create a pull request to the `develop` branch;
6. Add a brief description of the work done;
7. Expect comments from the authors.

## References
- [SAGE](https://github.com/ai-forever/sage) — superlib, developed jointly with our friends by the AGI NLP team, which provides advanced spelling corruptions and spell checking techniques, including using Augmentex.

## Authors
- [Aleksandr Abramov](https://github.com/Ab1992ao) — Source code and algorithm author;
- [Mark Baushenko](https://github.com/e0xextazy) — Source code lead developer.
