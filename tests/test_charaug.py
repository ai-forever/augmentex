from augmentex import CharAug


char_aug_rus_pc = CharAug(
    unit_prob=0.3,
    min_aug=1,
    max_aug=5,
    mult_num=3,
    random_seed=42,
    lang="rus",
    platform="pc",
)
char_aug_rus_mobile = CharAug(
    unit_prob=0.3,
    min_aug=1,
    max_aug=5,
    mult_num=3,
    random_seed=42,
    lang="rus",
    platform="mobile",
)
char_aug_eng_pc = CharAug(
    unit_prob=0.3,
    min_aug=1,
    max_aug=5,
    mult_num=3,
    random_seed=42,
    lang="eng",
    platform="pc",
)
char_aug_eng_mobile = CharAug(
    unit_prob=0.3,
    min_aug=1,
    max_aug=5,
    mult_num=3,
    random_seed=42,
    lang="eng",
    platform="mobile",
)

text_rus = "Привет, как дела?"
text_eng = "I am going home."


def test_methods() -> None:
    assert char_aug_rus_pc.actions_list == [
        "shift", "orfo", "typo", "delete", "multiply", "swap", "insert"]


def test_shift() -> None:
    assert "приВЕт, как дела?" == char_aug_rus_pc.augment(
        text=text_rus, action="shift")
    assert "ПРивЕт, каК дела?" == char_aug_rus_mobile.augment(
        text=text_rus, action="shift")
    assert "i Am gOinG home." == char_aug_eng_pc.augment(
        text=text_eng, action="shift")
    assert "i aM going hoMe." == char_aug_eng_mobile.augment(
        text=text_eng, action="shift")


def test_orfo() -> None:
    assert "Приыет, еак дела?" == char_aug_rus_pc.augment(
        text=text_rus, action="orfo")
    assert "Привит, кек доло?" == char_aug_rus_mobile.augment(
        text=text_rus, action="orfo")
    assert "I om going hamt." == char_aug_eng_pc.augment(
        text=text_eng, action="orfo")
    assert "I wm aoing hope." == char_aug_eng_mobile.augment(
        text=text_eng, action="orfo")


def test_typo() -> None:
    assert "Ппивет, каа деоа?" == char_aug_rus_pc.augment(
        text=text_rus, action="typo")
    assert "Приаео, квк днла?" == char_aug_rus_mobile.augment(
        text=text_rus, action="typo")
    assert "I am goijg bime." == char_aug_eng_pc.augment(
        text=text_eng, action="typo")
    assert "I am gpijg homw." == char_aug_eng_mobile.augment(
        text=text_eng, action="typo")


def test_delete() -> None:
    assert "Првет, к дл?" == char_aug_rus_pc.augment(
        text=text_rus, action="delete")
    assert "ивет какдела" == char_aug_rus_mobile.augment(
        text=text_rus, action="delete")
    assert "Iamgong hme." == char_aug_eng_pc.augment(
        text=text_eng, action="delete")
    assert "Iam gng hoe." == char_aug_eng_mobile.augment(
        text=text_eng, action="delete")


def test_insert() -> None:
    assert "Приветв, кщако денлъа?" == char_aug_rus_pc.augment(
        text=text_rus, action="insert")
    assert "Привретё, какг удела?з" == char_aug_rus_mobile.augment(
        text=text_rus, action="insert")
    assert "I am goinyg rhnomze." == char_aug_eng_pc.augment(
        text=text_eng, action="insert")
    assert "I aim goingz uhome.b" == char_aug_eng_mobile.augment(
        text=text_eng, action="insert")


def test_multiply() -> None:
    assert "Привеетт, как дела?" == char_aug_rus_pc.augment(
        text=text_rus, action="multiply")
    assert "Привет, как деелла?" == char_aug_rus_mobile.augment(
        text=text_rus, action="multiply")
    assert "I am going homme." == char_aug_eng_pc.augment(
        text=text_eng, action="multiply")
    assert "I am going home." == char_aug_eng_mobile.augment(
        text=text_eng, action="multiply")


def test_swap() -> None:
    assert "Првие, ткка деал?" == char_aug_rus_pc.augment(
        text=text_rus, action="swap")
    assert "Пиревт, какд лае?" == char_aug_rus_mobile.augment(
        text=text_rus, action="swap")
    assert "I am ginogh moe." == char_aug_eng_pc.augment(
        text=text_eng, action="swap")
    assert "I am ogngih ome." == char_aug_eng_mobile.augment(
        text=text_eng, action="swap")
