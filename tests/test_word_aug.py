"""Regression tests for WordAug — covers IndexError on unmatchable tokens."""

import unittest

from augmentex import WordAug


class TestWordAugReplace(unittest.TestCase):
    """Tests for the 'replace' action."""

    def setUp(self):
        self.aug = WordAug(
            unit_prob=1.0,
            min_aug=1,
            max_aug=5,
            lang="rus",
            platform="pc",
            random_seed=42,
        )

    def test_replace_does_not_crash_on_parentheses(self):
        """Regression for #20: tokens like '(' don't match the regex."""
        text = "один из партнёров ( партнёршу)"
        # Should not raise IndexError
        result = self.aug.augment(text=text, action="replace")
        self.assertIsInstance(result, str)

    def test_replace_stress_with_original_issue_text(self):
        """Exact reproduction from issue #20 — 100 iterations."""
        text = (
            "это когда в отношениях, один из партнёров насилует и истязает "
            "своего партнёра ( партнёршу) бывает абъюзив и по отношению "
            "родителей к своим детям"
        )
        for _ in range(100):
            result = self.aug.augment(text=text, action="replace")
            self.assertIsInstance(result, str)

    def test_replace_pure_special_chars(self):
        """Tokens consisting entirely of unmatchable characters."""
        text = "hello — world"
        result = self.aug.augment(text=text, action="replace")
        self.assertIsInstance(result, str)


class TestWordAugText2Emoji(unittest.TestCase):
    """Tests for the 'text2emoji' action — same bug pattern as replace."""

    def setUp(self):
        self.aug = WordAug(
            unit_prob=1.0,
            min_aug=1,
            max_aug=5,
            lang="rus",
            platform="pc",
            random_seed=42,
        )

    def test_text2emoji_does_not_crash_on_special_chars(self):
        text = "привет — мир"
        result = self.aug.augment(text=text, action="text2emoji")
        self.assertIsInstance(result, str)

    def test_text2emoji_with_parentheses(self):
        text = "слово ( другое)"
        result = self.aug.augment(text=text, action="text2emoji")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
