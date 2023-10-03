import os
import json
from typing import List

import numpy as np

from augmentex.base import BaseAug


RUSSIAN_VOCAB = [
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ё",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]


class CharAug(BaseAug):
    """Augmentation at the character level."""

    def __init__(
        self,
        unit_prob: float = 0.3,
        min_aug: int = 1,
        max_aug: int = 5,
        mult_num: int = 5,
        random_seed: int = None,
    ) -> None:
        """
        Args:
            unit_prob (float, optional): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
            min_aug (int, optional): The minimum amount of augmentation. Defaults to 1.
            max_aug (int, optional): The maximum amount of augmentation. Defaults to 5.
            mult_num (int, optional): Maximum repetitions of characters. Defaults to 5.
            random_seed (int, optional): Random seed. Default to None.
        """
        super().__init__(min_aug=min_aug, max_aug=max_aug, random_seed=random_seed)
        dir_path = os.path.dirname(os.path.abspath(__file__))

        with open(
            os.path.join(dir_path, "static_data", "typos_ru_en_digits_chars.json")
        ) as f:
            self.typo_dict = json.load(f)
        with open(os.path.join(dir_path, "static_data", "orfo_ru_chars.json")) as f:
            self.orfo_dict = json.load(f)
        with open(
            os.path.join(dir_path, "static_data", "shift_ru_en_digits.json")
        ) as f:
            self.shift_dict = json.load(f)

        self.mult_num = mult_num
        self.unit_prob = unit_prob
        self.__actions = [
            "shift",
            "orfo",
            "typo",
            "delete",
            "multiply",
            "swap",
            "insert",
        ]

    @property
    def actions_list(self) -> List[str]:
        """
        Returns:
            List[str]: A list of possible methods.
        """
        
        return self.__actions

    def _typo(self, char: str) -> str:
        """A method that simulates a typo by an adjacent key.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: A new symbol.
        """
        typo_char = np.random.choice(self.typo_dict.get(char, [char]))
        
        return typo_char

    def _shift(self, char: str) -> str:
        """Changes the case of the symbol.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: The same character but with a different case.
        """
        shift_char = self.shift_dict.get(char, char)
        
        return shift_char

    def _orfo(self, char: str) -> str:
        """Changes the symbol depending on the error statistics.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: A new symbol.
        """
        orfo_char = np.random.choice(
            RUSSIAN_VOCAB, p=self.orfo_dict.get(char, [1 / 33] * 33)
        )
        
        return orfo_char

    def _delete(self) -> str:
        """Deletes a random character.

        Returns:
            str: Empty string.
        """
        
        return ""

    def _insert(self, char: str) -> str:
        """Inserts a random character.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: A symbol + new symbol.
        """
        
        return char + np.random.choice(RUSSIAN_VOCAB)

    def _multiply(self, char: str) -> str:
        """Repeats a randomly selected character.

        Args:
            char (str): A symbol from the word.

        Returns:
            str: A symbol from the word matmul n times.
        """
        if char in [" ", ",", ".", "?", "!", "-"]:
            return char
        else:
            n = np.random.randint(1, self.mult_num)
            return char * n

    # def _clean_punc(self, text: str) -> str:
    #     """Clears the text from punctuation.

    #     Args:
    #         text (str): Original text.

    #     Returns:
    #         str: Text without punctuation.
    #     """
    #     return text.translate(str.maketrans("", "", string.punctuation))

    def augment(self, text, action=None):
        if action is None:
            action = np.random.choice(self.__actions)

        typo_text_arr = list(text)
        aug_idxs = self._aug_indexing(typo_text_arr, self.unit_prob, clip=True)
        for idx in aug_idxs:
            if action == "typo":
                typo_text_arr[idx] = self._typo(typo_text_arr[idx])
            elif action == "shift":
                typo_text_arr[idx] = self._shift(typo_text_arr[idx])
            elif action == "delete":
                typo_text_arr[idx] = self._delete()
            elif action == "insert":
                typo_text_arr[idx] = self._insert(typo_text_arr[idx])
            elif action == "orfo":
                typo_text_arr[idx] = self._orfo(typo_text_arr[idx])
            elif action == "multiply":
                typo_text_arr[idx] = self._multiply(typo_text_arr[idx])
            elif action == "swap":
                sw = max(0, idx - 1)
                typo_text_arr[sw], typo_text_arr[idx] = (
                    typo_text_arr[idx],
                    typo_text_arr[sw],
                )
            else:
                raise NameError(
                    """These type of augmentation is not available, please try TypoAug.actions_list() to see
                available augmentations"""
                )

        return "".join(typo_text_arr)
