from augmentex import WordAug


word_aug_rus_pc = WordAug(
    unit_prob=0.4,
    min_aug=1,
    max_aug=5,
    random_seed=42,
    lang="rus",
    platform="pc",
)
word_aug_rus_mobile = WordAug(
    unit_prob=0.4,
    min_aug=1,
    max_aug=5,
    random_seed=42,
    lang="rus",
    platform="mobile",
)
word_aug_eng_pc = WordAug(
    unit_prob=0.4,
    min_aug=1,
    max_aug=5,
    random_seed=42,
    lang="eng",
    platform="pc",
)
word_aug_eng_mobile = WordAug(
    unit_prob=0.4,
    min_aug=1,
    max_aug=5,
    random_seed=42,
    lang="eng",
    platform="mobile",
)

text_rus = "Привет, как дела?"
text_eng = "I am going home."


def test_methods() -> None:
    assert word_aug_rus_pc.actions_list == [
        "replace", "delete", "swap", "stopword", "reverse", "text2emoji", "split", "ngram"]


def test_replace() -> None:
    assert "Привет, как делло?" == word_aug_rus_pc.augment(
        text=text_rus, action="replace")
    assert "Привет, ккак дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="replace")
    assert "I am gone home." == word_aug_eng_pc.augment(
        text=text_eng, action="replace")
    assert "I m going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="replace")


def test_delete() -> None:
    assert "как дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="delete")
    assert "Привет, как" == word_aug_rus_mobile.augment(
        text=text_rus, action="delete")
    assert "I am going" == word_aug_eng_pc.augment(
        text=text_eng, action="delete")
    assert "am going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="delete")


def test_swap() -> None:
    assert "как Привет, дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="swap")
    assert "как Привет, дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="swap")
    assert "I am going home." == word_aug_eng_pc.augment(
        text=text_eng, action="swap")
    assert "I am going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="swap")


def test_stopword() -> None:
    assert "Привет, как в общем-то дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="stopword")
    assert "Привет, хотя как дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="stopword")
    assert "totally I am going home." == word_aug_eng_pc.augment(
        text=text_eng, action="stopword")
    assert "I am going okay home." == word_aug_eng_mobile.augment(
        text=text_eng, action="stopword")


def test_reverse() -> None:
    assert "Привет, Как дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="reverse")
    assert "Привет, как Дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="reverse")
    assert "I am going Home." == word_aug_eng_pc.augment(
        text=text_eng, action="reverse")
    assert "I am Going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="reverse")


def test_text2emoji() -> None:
    assert "Привет, как дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="text2emoji")
    assert "✋, как дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="text2emoji")
    assert "ℹ am going home." == word_aug_eng_pc.augment(
        text=text_eng, action="text2emoji")
    assert "I am going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="text2emoji")


def test_split() -> None:
    assert "Привет, как д е л а ?" == word_aug_rus_pc.augment(
        text=text_rus, action="split")
    assert "Привет, к а к дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="split")
    assert "I am going home." == word_aug_eng_pc.augment(
        text=text_eng, action="split")
    assert "I am g o i n g home." == word_aug_eng_mobile.augment(
        text=text_eng, action="split")


def test_ngram() -> None:
    assert "Привет, как дела?" == word_aug_rus_pc.augment(
        text=text_rus, action="ngram")
    assert "Пбриет, как дела?" == word_aug_rus_mobile.augment(
        text=text_rus, action="ngram")
    assert "I am going hoom." == word_aug_eng_pc.augment(
        text=text_eng, action="ngram")
    assert "I am going home." == word_aug_eng_mobile.augment(
        text=text_eng, action="ngram")
