import random
import json
import re
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Set

import pymorphy2
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

from augmentex.variables import SUPPORT_LANGUAGES, SUPPORT_PLATFORMS


class BaseAug(ABC):
    def __init__(self, min_aug: int = 1, max_aug: int = 5, random_seed: int = None, lang: str = "rus", platform: str = "pc") -> None:
        """
        Args:
            min_aug (int, optional): The minimum amount of augmentation. Defaults to 1.
            max_aug (int, optional): The maximum amount of augmentation. Defaults to 5.
            random_seed (int, optional): Random seed. Default to None.
            lang (str, optional): Language of texts. Default to 'rus'.
            platform (str, optional): Type of platform where statistic was collected. Defaults to 'pc'.
        """
        self.min_aug = min_aug
        self.max_aug = max_aug
        self.random_seed = random_seed
        self.lang = lang
        self.platform = platform

        if self.random_seed:
            self.__fix_random_seed(self.random_seed)

        if self.lang not in SUPPORT_LANGUAGES:
            raise ValueError(
                f"""Augmentex support only {', '.join(SUPPORT_LANGUAGES)} languages.
                You put {self.lang}.""")
        if self.platform not in SUPPORT_PLATFORMS:
            raise ValueError(
                f"""Augmentex support only {', '.join(SUPPORT_PLATFORMS)} platforms.
                You put {self.platform}.""")

    def _read_json(self, path: str) -> Dict:
        """Read JSON to Dict.

        Args:
            path (str): Path to file.

        Returns:
            Dict: dict with data.
        """
        with open(path) as f:
            data = json.load(f)

        return data

    def __fix_random_seed(self, random_seed: int) -> None:
        """Fixing random seed.

        Args:
            random_seed (int): Integer digit.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

    def __augs_count(self, size: int, rate: float) -> int:
        """Counts the number of augmentations and performs circumcision by the maximum or minimum number.

        Args:
            size (int): The number of units (chars or words) in the text.
            rate (float): The percentage of units to which augmentation will be applied.

        Returns:
            int: The amount of augmentation.
        """
        cnt = 0
        if size > 1:
            cnt = int(rate * size)

        return cnt

    def __get_random_idx(self, inputs: List[str], aug_count: int) -> List[int]:
        """Randomly select indexes for augmentation

        Args:
            inputs (List[str]): List of units.
            aug_count (int): The amount of augmentation.

        Returns:
            List[int]: List of indices.
        """
        token_idxes = [i for i in range(len(inputs))]
        aug_idxs = random.sample(token_idxes, aug_count)

        return aug_idxs

    def _aug_indexing(
        self, inputs: List[str], rate: float, clip: bool = False
    ) -> List[int]:
        """
        Args:
            inputs (List[str]): List of units.
            rate (float): The percentage of units to which augmentation will be applied.
            clip (bool): Takes into account the maximum and minimum values. Defaults to False.

        Returns:
            List[int]: List of indices.
        """
        aug_count = self.__augs_count(len(inputs), rate)
        if clip:
            aug_count = max(aug_count, self.min_aug)
            aug_count = min(aug_count, self.max_aug)

        aug_idxs = self.__get_random_idx(inputs, aug_count)

        return aug_idxs

    def aug_batch(
        self,
        batch: List[str],
        batch_prob: float = 1.0,
        action: Union[None, str] = None,
    ) -> List[str]:
        """The use of augmentation to several lines

        Args:
            batch (List[str]): List of lines for augmentation.
            batch_prob (float, optional): The percentage of units to which augmentation will be applied. Defaults to 1.0.
            action (Union[None, str], optional): Indicates what action will be applied. Defaults to None. If None, then a random action is chosen.

        Returns:
            List[str]: List of augmented lines.
        """
        aug_batch = batch.copy()
        aug_idxs = self._aug_indexing(aug_batch, batch_prob)
        for idx in aug_idxs:
            aug_batch[idx] = self.augment(aug_batch[idx], action)

        return aug_batch

    @abstractmethod
    def augment(self, text, action):
        pass


class AttackBase(ABC):
    """A base class that combines methods independent of the PyTorch or TensorFlow training framework.
    """

    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float) -> None:
        """
        Args:
            observer_model (_type_): _description_
            syn_dict (Dict[str, List[str]]): A dictionary with synonyms for each word. It should be in the form of:
                {'word_1': ['syn_11', 'syn_12', ..., 'syn_1N'],
                'word_2': ['syn_21', 'syn_22', ..., 'syn_2N'],
                ...}
            p_phrase (float): Percentage of the phrase to which the algorithm will be applied.
        """
        if not hasattr(observer_model, "get_embedding"):
            raise NameError(
                """For the algorithm to work correctly, your model class must contain a method called 'get_embedding', 
                which takes Union[str, List[str]] as input and returns a vector (tensor) of the dimension of the hidden state of your model.""")
        if not isinstance(syn_dict, dict):
            raise TypeError("syn_dict should be dict type.")
        if not isinstance(p_phrase, float):
            raise TypeError("Works only with Float type from 0 to 1")
        if p_phrase <= 0.0:
            raise ValueError("Work with Positive Numbers Only")
        if p_phrase > 1.0:
            p_phrase = 1.0
            print("Your p_phrase value was greater than 1.0, so it is now 1.0.")

        self.observer_model = observer_model
        self.p_phrase = p_phrase
        self.russian_stopwords = set(stopwords.words("russian"))
        self.syn_dict = syn_dict
        self.morph = pymorphy2.MorphAnalyzer()

    def clear_string(self, text: str) -> str:
        """Leaves only symbols, numbers and punctuation marks in the text.

        Args:
            text (str): Source text.

        Returns:
            str: Cleared text.
        """
        text = re.sub(r"[^\w\s]", "", text)

        return text

    def get_top_damage_word(self, cnds: List[List[str]], ids: List[npt.NDArray[np.int_]]) -> List[List[str]]:
        """A function that selects the best candidates based on the obtained indexes and clears them of stop words.

        Args:
            cnds (List[List[str]]): A list of batch size length that stores lists with candidates for replacement.
            ids (List[npt.NDArray[np.int_]]): A list of batch size length that stores lists with indexes of words making the greatest contribution to the sentence. 
                Sorted by descending contribution.

        Returns:
            List[List[str]]: A list of batch size length that stores lists with the best candidates for replacement in the order of 'ids'.
        """
        new_cnds = []
        for i, cnd in enumerate(cnds):
            tmp_lst = np.array(cnd)[ids[i]]
            tmp_lst = [c for c in tmp_lst if c not in self.russian_stopwords]
            new_cnds.append(tmp_lst)

        return new_cnds

    def choose_damage_words(self, target_model, text: List[str], label) -> List[List[str]]:
        """A function that searches for the best candidates in a replacement offer.

        Args:
            text (List[str]): A list of batch size length that stores source text.
            target_model (_type_): _description_

        Returns:
            List[List[str]]: A list of batch size length that stores lists with the best candidates for replacement in the order of 'ids'.
        """
        text_clear = [self.clear_string(txt) for txt in text]
        cnds = [txt.split() for txt in text_clear]
        ids = self.iterate_words_score(target_model, cnds, text_clear, label)

        return self.get_top_damage_word(cnds, ids)

    def get_inflect(self, word: str, inflect_rules: Set[str]) -> Union[str, None]:
        """A function that puts a word in the desired form based on the rules.

        Args:
            word (str): The word in the initial form.
            inflect_rules (Set[str]): Rules for forming words derived from Pymorphy2.

        Returns:
            Union[str, None]: A word in the right form or None if it could not be converted.
        """
        try:
            p = self.morph.parse(word)[0]
            return p.inflect(inflect_rules)[0]
        except:
            return None

    def get_synonims(self, cnds: List[str]) -> Dict[str, List[str]]:
        """A function that searches for synonyms for each word in the list.

        Args:
            cnds (List[str]): A list containing the words candidates for replacement.

        Returns:
            Dict[str, List[str]]: A dictionary where the keys are candidate words to replace, and the values are a list of synonyms for each candidate.
        """
        dct_cnd_syn = {c: [] for c in cnds}
        syns_keys = set(self.syn_dict.keys())

        for cand in cnds:
            p = self.morph.parse(cand)[0]
            cand_norm = p.normal_form
            try:
                if cand_norm in syns_keys:
                    norm_pos = self.morph.parse(cand_norm)[0].tag.POS
                    cand_poses = [p.tag.POS, p.tag.animacy, p.tag.aspect, p.tag.case, p.tag.gender, p.tag.involvement,
                                  p.tag.mood, p.tag.number, p.tag.person, p.tag.tense, p.tag.transitivity, p.tag.voice]
                    inflect_rules = {l for l in cand_poses if l != None}

                    syns = self.syn_dict[cand_norm]
                    syn_cnt = len(syns)
                    syn_cand_pos = [str(self.morph.parse(s)[0].tag.POS)
                                    for s in syns]
                    syns_pos = [syns[i] for i in range(
                        syn_cnt) if syn_cand_pos[i] == norm_pos]

                    ids_cs = self.get_top_nearest_words(cand, syns_pos)
                    dct_cnd_syn[cand] = [s for s in np.array(syns)[ids_cs]]
                    dct_cnd_syn[cand] = [self.get_inflect(
                        s, inflect_rules) for s in dct_cnd_syn[cand]]
                    dct_cnd_syn[cand] = [c for c in dct_cnd_syn[cand] if c != None and len(
                        c.split('-')) == 1 and len(c.split()) == 1]
                else:
                    continue
            except:
                continue

        return dct_cnd_syn

    def get_change_hard_word2sentence(self, text: str, dct_cnd_syn: Dict[str, List[str]]) -> List[str]:
        """A function that generates sentences with possible substitutions.

        Args:
            text (str): Original sentence.
            dct_cnd_syn (Dict[str, List[str]]): A dictionary where the keys are candidate words to replace, and the values are a list of synonyms for each candidate.

        Returns:
            List[str]: List of sentences with possible replacements.
        """
        hard_strs = []
        for key in dct_cnd_syn.keys():
            for s in dct_cnd_syn[key]:
                if s == '':
                    continue
                else:
                    st_ch = text.replace(key, s)
                    hard_strs.append(st_ch)

        return hard_strs

    def paraphrase(self, target_model, text: List[str], label) -> List[str]:
        """
        Args:
            text (List[str]): Original sentence.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        cnds_s = self.choose_damage_words(target_model, text, label)
        dct_s = [self.get_synonims(cnds) for cnds in cnds_s]
        dct_s = [{k: v for k, v in d.items() if len(v) != 0} for d in dct_s]

        result = []
        for i, d in enumerate(dct_s):
            if len(d) == 0:
                result.append(text[i])
                continue
            else:
                p_h = self.get_change_hard_word2sentence(text[i], d)
                adv_cands = self.original2advers_sim(text[i], p_h)

            if len(adv_cands) != 0:
                result.append(np.random.choice(adv_cands))
            else:
                result.append(text[i])
                continue

        return result

    def get_top_nearest_words(self, word: str, syns: List[str]) -> npt.NDArray[np.int_]:
        """
        Args:
            word (str): The original word.
            syns (List[str]): A list of words in which we will look for similar ones.

        Returns:
            npt.NDArray[np.int_]: Indexes of the nearest words.
        """
        scores = -1 * cosine_similarity(self.get_embedding_cpu(
            self.observer_model, word), self.get_embedding_cpu(self.observer_model, syns))[0]
        len_s = len(scores)
        top = int(len_s * 0.5) if int(len_s * 0.5) > 1 else int(len_s * 0.75)

        return np.argsort(scores)[:top]

    def original2advers_sim(self, text: str, p_h: List[str]) -> List[str]:
        """A function that selects 25% of the best sentences for replacement.

        Args:
            text (str): Original sentence.
            p_h (List[str]): List of sentences with possible replacements.

        Returns:
            List[str]: List of sentences with best possible replacements.
        """
        scores = -1 * cosine_similarity(self.get_embedding_cpu(
            self.observer_model, text), self.get_embedding_cpu(self.observer_model, p_h))[0]
        top = int(len(p_h) * 0.25)
        ids_cs = np.argsort(scores)[:top]

        return [s for s in np.array(p_h)[ids_cs]]


class EncoderAttackBase(AttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5) -> None:
        """
        Args:
            observer_model (_type_): _description_
            syn_dict (Dict[str, List[str]]): A dictionary with synonyms for each word. It should be in the form of:
                {'word_1': ['syn_1', 'syn_2', ..., 'syn_N'],
                'word_2': ['syn_1', 'syn_2', ..., 'syn_N'],
                ...}
            p_phrase (float, optional): Percentage of the phrase to which the algorithm will be applied. Defaults to 0.5.
        """
        super().__init__(observer_model, syn_dict, p_phrase)

    def get_embedding_cpu(self, model, batch_text: Union[str, List[str]]) -> npt.NDArray[np.float32]:
        """A function that calculate a embedding of text and moves it into RAM.

        Args:
            model (_type_): _description_
            batch_text (Union[str, List[str]]): The text or list of texts to calculate embedding for.

        Returns:
            npt.NDArray[np.float32]: Embedding texts.
        """
        embs = model.get_embedding(batch_text).cpu().numpy()

        return embs


class DecoderAttackBase(AttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5) -> None:
        """
        Args:
            observer_model (_type_): _description_
            syn_dict (Dict[str, List[str]]): A dictionary with synonyms for each word. It should be in the form of:
                {'word_1': ['syn_1', 'syn_2', ..., 'syn_N'],
                'word_2': ['syn_1', 'syn_2', ..., 'syn_N'],
                ...}
            p_phrase (float, optional): Percentage of the phrase to which the algorithm will be applied. Defaults to 0.5.
        """
        super().__init__(observer_model, syn_dict, p_phrase)

    def get_embedding_cpu(self, model, batch_text: Union[str, List[str]]) -> npt.NDArray[np.float32]:
        """A function that calculate a embedding of text and moves it into RAM.

        Args:
            model (_type_): _description_
            batch_text (Union[str, List[str]]): The text or list of texts to calculate embedding for.

        Returns:
            npt.NDArray[np.float32]: Embedding texts.
        """
        embs = model.get_embedding(batch_text).cpu().numpy()

        return embs
