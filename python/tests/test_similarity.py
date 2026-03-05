"""Tests for text similarity metrics (src/analysis/similarity.py)."""

import pytest

from src.analysis.similarity import TextSimilarity


@pytest.fixture
def sim():
    return TextSimilarity()


# ── cosine_similarity ────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_texts(self, sim):
        text = "the quick brown fox jumps over the lazy dog"
        assert sim.cosine_similarity(text, text) == pytest.approx(1.0)

    def test_disjoint_texts(self, sim):
        assert sim.cosine_similarity("aaa bbb ccc", "xxx yyy zzz") == pytest.approx(0.0)

    def test_both_empty(self, sim):
        assert sim.cosine_similarity("", "") == 0.0

    def test_one_empty(self, sim):
        assert sim.cosine_similarity("hello world", "") == 0.0
        assert sim.cosine_similarity("", "hello world") == 0.0

    def test_partial_overlap(self, sim):
        val = sim.cosine_similarity("the cat sat", "the dog sat")
        assert 0.0 < val < 1.0

    def test_case_insensitive(self, sim):
        a = "Hello World"
        b = "hello world"
        assert sim.cosine_similarity(a, b) == pytest.approx(1.0)

    def test_unicode(self, sim):
        a = "café résumé naïve"
        b = "café résumé naïve"
        assert sim.cosine_similarity(a, b) == pytest.approx(1.0)

    def test_repeated_tokens(self, sim):
        """Repeated tokens affect TF weighting."""
        a = "cat cat cat"
        b = "cat dog dog"
        val = sim.cosine_similarity(a, b)
        assert 0.0 < val < 1.0


# ── jaccard_token_similarity ─────────────────────────────────────────────


class TestJaccardTokenSimilarity:
    def test_identical(self, sim):
        text = "alpha beta gamma"
        assert sim.jaccard_token_similarity(text, text) == pytest.approx(1.0)

    def test_disjoint(self, sim):
        assert sim.jaccard_token_similarity("aaa bbb", "xxx yyy") == pytest.approx(0.0)

    def test_both_empty(self, sim):
        assert sim.jaccard_token_similarity("", "") == 1.0

    def test_one_empty(self, sim):
        assert sim.jaccard_token_similarity("hello", "") == 0.0
        assert sim.jaccard_token_similarity("", "hello") == 0.0

    def test_superset(self, sim):
        val = sim.jaccard_token_similarity("a b c d", "a b")
        # intersection=2, union=4 → 0.5
        assert val == pytest.approx(0.5)

    def test_partial(self, sim):
        val = sim.jaccard_token_similarity("a b c", "b c d")
        # intersection=2, union=4 → 0.5
        assert val == pytest.approx(0.5)


# ── edit_distance_normalized ─────────────────────────────────────────────


class TestEditDistanceNormalized:
    def test_identical(self, sim):
        assert sim.edit_distance_normalized("hello", "hello") == 0.0

    def test_completely_different(self, sim):
        val = sim.edit_distance_normalized("abc", "xyz")
        assert val == pytest.approx(1.0)

    def test_one_empty(self, sim):
        assert sim.edit_distance_normalized("abc", "") == 1.0
        assert sim.edit_distance_normalized("", "abc") == 1.0

    def test_both_empty(self, sim):
        assert sim.edit_distance_normalized("", "") == 0.0

    def test_single_char_diff(self, sim):
        val = sim.edit_distance_normalized("cat", "bat")
        assert val == pytest.approx(1 / 3)

    def test_different_lengths(self, sim):
        val = sim.edit_distance_normalized("ab", "abcd")
        # distance=2, max_len=4 → 0.5
        assert val == pytest.approx(0.5)

    def test_range_zero_to_one(self, sim):
        val = sim.edit_distance_normalized("testing", "resting")
        assert 0.0 <= val <= 1.0


# ── bleu_1gram ───────────────────────────────────────────────────────────


class TestBleu1gram:
    def test_identical(self, sim):
        text = "the quick brown fox"
        assert sim.bleu_1gram(text, text) == pytest.approx(1.0)

    def test_no_overlap(self, sim):
        assert sim.bleu_1gram("aaa bbb", "xxx yyy") == pytest.approx(0.0)

    def test_empty_candidate(self, sim):
        assert sim.bleu_1gram("hello world", "") == 0.0

    def test_empty_reference(self, sim):
        # candidate tokens exist but none match empty reference
        assert sim.bleu_1gram("", "hello world") == pytest.approx(0.0)

    def test_partial(self, sim):
        val = sim.bleu_1gram("a b c d", "a b x y")
        # candidate tokens: a, b, x, y (4 total). Matches: a, b → clipped=2 → 2/4=0.5
        assert val == pytest.approx(0.5)

    def test_repeated_token_clipping(self, sim):
        # reference has "a" once, candidate repeats "a" three times
        val = sim.bleu_1gram("a b", "a a a")
        # candidate counts: a=3. ref counts: a=1, b=1. clipped: min(3,1)=1 → 1/3
        assert val == pytest.approx(1 / 3)
