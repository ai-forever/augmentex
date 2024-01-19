from typing import Dict, List, Union

import torch
import evaluate
import numpy.typing as npt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from augmentex.base import EncoderAttackBase, DecoderAttackBase


class UnsupervisedDecoderAttack(DecoderAttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5) -> None:
        self.perplexity = evaluate.load("perplexity", module_type="metric")

        super().__init__(observer_model, syn_dict, p_phrase)

    def iterate_words_score(self, target_model, cnds: List[List[str]], text_clear: List[str], label) -> List[npt.NDArray[np.int_]]:
        """A function that counts the contribution of each candidate in a sentence and returns the top len(sentences) * p_phrase of candidate indexes.

        Args:
            cnds (List[List[str]]): A batch size list that stores lists with all candidates for each offer.
            text_clear (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[npt.NDArray[np.int_]]: Indexes of the best candidates.
        """
        texts_wo_cnd = []
        for i, txt in enumerate(text_clear):
            tmp_list = []
            for cnd in cnds[i]:
                tmp_list.append(txt.replace(
                    cnd, "").replace("  ", " ").strip(" "))
            texts_wo_cnd.append(tmp_list)
        texts_wo_cnd_ppl = [self.get_perplexity(
            target_model, text_wo_cnd) for text_wo_cnd in texts_wo_cnd]

        return [np.flip(np.argsort(text_wo_cnd_ppl)[:min(int(len(text_wo_cnd_ppl) * self.p_phrase) + 1, len(text_wo_cnd_ppl))]) for text_wo_cnd_ppl in texts_wo_cnd_ppl]

    def get_perplexity(self, model, batch_text: Union[str, List[str]]) -> npt.NDArray[np.float32]:
        """A function that calculate a embedding of text and moves it into RAM.

        Args:
            model (_type_): _description_
            batch_text (Union[str, List[str]]): The text or list of texts to calculate embedding for.

        Returns:
            npt.NDArray[np.float32]: Embedding texts.
        """
        ppl = self.perplexity.compute(model_id=model.path,
                                      add_start_token=False,
                                      predictions=batch_text)['perplexities']

        return ppl

    def attack(self, target_model, sample: List[str], label=None) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(target_model, sample, label)
        target_model.train()

        return sample


class UnsupervisedEncoderAttack(EncoderAttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5) -> None:
        super().__init__(observer_model, syn_dict, p_phrase)

    def iterate_words_score(self, target_model, cnds: List[List[str]], text_clear: List[str], label) -> List[npt.NDArray[np.int_]]:
        """A function that counts the contribution of each candidate in a sentence and returns the top len(sentences) * p_phrase of candidate indexes.

        Args:
            cnds (List[List[str]]): A batch size list that stores lists with all candidates for each offer.
            text_clear (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[npt.NDArray[np.int_]]: Indexes of the best candidates.
        """
        text_clr_embs = self.get_embedding_cpu(target_model, text_clear)

        texts_wo_cnd = []
        for i, txt in enumerate(text_clear):
            tmp_list = []
            for cnd in cnds[i]:
                tmp_list.append(txt.replace(
                    cnd, "").replace("  ", " ").strip(" "))
            texts_wo_cnd.append(tmp_list)
        texts_wo_cnd_embs = [self.get_embedding_cpu(
            target_model, text_wo_cnd) for text_wo_cnd in texts_wo_cnd]
        scores = [cosine_similarity([text_clr_embs[i]], texts_wo_cnd_embs[i])[
            0] for i in range(len(text_clear))]

        return [np.argsort(score)[:min(int(len(score) * self.p_phrase) + 1, len(score))] for score in scores]

    def attack(self, target_model, sample: List[str], label=None) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(target_model, sample, label)
        target_model.train()

        return sample


class ClassificationEncoderAttack(EncoderAttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5, norm: str = "mae") -> None:
        self.norm = norm

        super().__init__(observer_model, syn_dict, p_phrase)

    def iterate_words_score(self, target_model, cnds: List[List[str]], text_clear: List[str], label: List[int]) -> List[npt.NDArray[np.int_]]:
        texts_wo_cnd = []
        for i, txt in enumerate(text_clear):
            tmp_list = []
            for cnd in cnds[i]:
                tmp_list.append(txt.replace(
                    cnd, "").replace("  ", " ").strip(" "))
            texts_wo_cnd.append(tmp_list)
        texts_wo_cnd_scores = [target_model(
            text_wo_cnd) for text_wo_cnd in texts_wo_cnd]
        if self.norm == "mae":
            scores = [[abs(1.0 - texts_wo_cnd_scores[i][j][label[i]])
                       for j in range(len(texts_wo_cnd_scores[i]))] for i in range(len(text_clear))]
        elif self.norm == "mse":
            scores = [[abs(1.0 - texts_wo_cnd_scores[i][j][label[i]])**2 for j in range(
                len(texts_wo_cnd_scores[i]))] for i in range(len(text_clear))]

        return [np.argsort(score)[::-1][:min(int(len(score) * self.p_phrase) + 1, len(score))] for score in scores]

    def attack(self, target_model, sample: List[str], label: List[int]) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(target_model, sample, label)
        target_model.train()

        return sample


class RegressionEncoderAttack(EncoderAttackBase):
    def __init__(self, observer_model, syn_dict: Dict[str, List[str]], p_phrase: float = 0.5, norm: str = "mae") -> None:
        self.norm = norm

        super().__init__(observer_model, syn_dict, p_phrase)

    def iterate_words_score(self, target_model, cnds: List[List[str]], text_clear: List[str], label: List[int]) -> List[npt.NDArray[np.int_]]:
        texts_wo_cnd = []
        for i, txt in enumerate(text_clear):
            tmp_list = []
            for cnd in cnds[i]:
                tmp_list.append(txt.replace(
                    cnd, "").replace("  ", " ").strip(" "))
            texts_wo_cnd.append(tmp_list)
        texts_wo_cnd_scores = [target_model(
            text_wo_cnd)[0] for text_wo_cnd in texts_wo_cnd]
        if self.norm == "mae":
            scores = [[abs(label[i] - texts_wo_cnd_scores[i][j])
                       for j in range(len(texts_wo_cnd_scores[i]))] for i in range(len(text_clear))]
        elif self.norm == "mse":
            scores = [[abs(label[i] - texts_wo_cnd_scores[i][j])**2 for j in range(
                len(texts_wo_cnd_scores[i]))] for i in range(len(text_clear))]

        return [np.argsort(score)[::-1][:min(int(len(score) * self.p_phrase) + 1, len(score))] for score in scores]

    def attack(self, target_model, sample: List[str], label: List[int]) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(target_model, sample, label)
        target_model.train()

        return sample
