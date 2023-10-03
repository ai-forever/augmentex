import os
import re
import json
from typing import List

import numpy as np

from augmentex.base import BaseAug


class WordAug(BaseAug):
    """Augmentation at the level of words."""

    def __init__(
        self,
        min_aug: int = 1,
        max_aug: int = 5,
        unit_prob: float = 0.3,
        random_seed: int = None,
        lang: str = "rus",
        platform: str = "pc",
    ) -> None:
        """
        Args:
            min_aug (int, optional): The minimum amount of augmentation. Defaults to 1.
            max_aug (int, optional): The maximum amount of augmentation. Defaults to 5.
            unit_prob (float, optional): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
            random_seed (int, optional): Random seed. Default to None.
            lang (str, optional): Language of texts. Default to 'rus'.
            platform (str, optional): Type of platform where statistic was collected. Defaults to 'pc'.
        """
        super().__init__(min_aug=min_aug, max_aug=max_aug,
                         random_seed=random_seed, lang=lang, platform=platform)
        dir_path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(dir_path, "static_data", self.lang, "stopwords.json")) as f:
            self.stopwords = json.load(f)
        with open(os.path.join(dir_path, "static_data", self.lang, self.platform, "orfo_words.json")) as f:
            self.orfo_words = json.load(f)
        with open(os.path.join(dir_path, "static_data", self.lang, "text2emoji.json")) as f:
            self.text2emoji_map = json.load(f)

        self.unit_prob = unit_prob
        self.__actions = [
            "replace",
            "delete",
            "swap",
            "stopword",
            "reverse",
            "text2emoji",
            "split",
        ]

    @property
    def actions_list(self) -> List[str]:
        """
        Returns:
            List[str]: A list of possible methods.
        """

        return self.__actions

    def _reverse_case(self, word: str) -> str:
        """Changes the case of the first letter to the reverse.

        Args:
            word (str): The initial word.

        Returns:
            str: A word with a different case of the first letter.
        """
        if len(word):
            if word[0].isupper():
                word = word.lower()
            else:
                word = word.capitalize()

        return word

    def _text2emoji(self, word: str) -> str:
        """Replace word to emoji.

        Args:
            word (str): A word with the correct spelling.

        Returns:
            str: Emoji that matches this word.
        """
        word = re.findall("[а-яА-ЯёЁa-zA-Z0-9']+|[.,!?;]+", word)
        words = self.text2emoji_map.get(word[0].lower(), [word[0]])
        word[0] = np.random.choice(words)

        return "".join(word)

    def _split(self, word: str) -> str:
        """Divides a word character-by-character.

        Args:
            word (str): A word with the correct spelling.

        Returns:
            str: Word with spaces.
        """
        word = " ".join(list(word))

        return word

    def _replace(self, word: str) -> str:
        """Replaces a word with the correct spelling with a word with spelling errors.

        Args:
            word (str): A word with the correct spelling.

        Returns:
            str: A misspelled word.
        """
        word = re.findall("[а-яА-ЯёЁa-zA-Z0-9']+|[.,!?;]+", word)
        word_probas = self.orfo_words.get(word[0].lower(), [[word[0]], [1.0]])
        word[0] = np.random.choice(word_probas[0], p=word_probas[1])

        return "".join(word)

    def _delete(self) -> str:
        """Deletes a random word.

        Returns:
            str: Empty string.
        """

        return ""

    def _stopword(self, word: str) -> str:
        """Adds a stop word before the word.

        Args:
            word (str): Just word.

        Returns:
            str: Stopword + word.
        """
        stopword = np.random.choice(self.stopwords)

        return " ".join([stopword, word])

    def augment(self, text: str, action: str = None) -> str:
        if action is None:
            action = np.random.choice(self.__actions)

        aug_sent_arr = text.split()
        aug_idxs = self._aug_indexing(aug_sent_arr, self.unit_prob, clip=True)
        for idx in aug_idxs:
            if action == "delete":
                aug_sent_arr[idx] = self._delete()
            elif action == "reverse":
                aug_sent_arr[idx] = self._reverse_case(aug_sent_arr[idx])
            elif action == "swap":
                swap_idx = np.random.randint(0, len(aug_sent_arr) - 1)
                aug_sent_arr[swap_idx], aug_sent_arr[idx] = (
                    aug_sent_arr[idx],
                    aug_sent_arr[swap_idx],
                )
            elif action == "stopword":
                aug_sent_arr[idx] = self._stopword(aug_sent_arr[idx])
            elif action == "replace":
                aug_sent_arr[idx] = self._replace(aug_sent_arr[idx])
            elif action == "text2emoji":
                aug_sent_arr[idx] = self._text2emoji(aug_sent_arr[idx])
            elif action == "split":
                aug_sent_arr[idx] = self._split(aug_sent_arr[idx])
            else:
                raise NameError(
                    """These type of augmentation is not available, please check EDAAug.actions_list() to see
                available augmentations"""
                )

        return re.sub(" +", " ", " ".join(aug_sent_arr).strip())
