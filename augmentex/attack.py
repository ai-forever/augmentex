from typing import Dict, List, Union

import torch
import evaluate
import numpy.typing as npt
import numpy as np

from augmentex.base import EncoderAttackBase, DecoderAttackBase


class EncoderAttack(EncoderAttackBase):
    """A wrapper class that takes into account the features of the PyTorch framework.

    Args:
        AttackBase: A base class with basic functions.
    """

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

        self.device = next(observer_model.parameters()).device

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

    def attack(self, sample: List[str], target_model) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(sample, target_model)
        target_model.train()

        return sample


class DecoderAttack(DecoderAttackBase):
    """A wrapper class that takes into account the features of the PyTorch framework.

    Args:
        AttackBase: A base class with basic functions.
    """

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

        self.device = next(observer_model.parameters()).device
        self.perplexity = evaluate.load("perplexity", module_type="metric")

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

    def attack(self, sample: List[str], target_model) -> List[str]:
        """
        Args:
            sample (List[str]): List of initial sentences.
            target_model (_type_): _description_

        Returns:
            List[str]: The final sentences that will be instead of the original ones.
        """
        target_model.eval()
        with torch.no_grad():
            sample = self.paraphrase(sample, target_model)
        target_model.train()

        return sample
