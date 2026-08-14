"""Tests for special tokens support in grammar matching and bitmask generation."""

import sys
from typing import Set

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
)

STOP_TOKEN_ID = 1  # "</s>" in our test vocab


class _Checker:
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self._token_bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def __call__(self, matcher: xgr.GrammarMatcher, *, accept: int, reject: Set[int]) -> None:
        matcher.fill_next_token_bitmask(self._token_bitmask)
        rejected = set(_get_masked_tokens_from_bitmask(self._token_bitmask, self._vocab_size))
        assert rejected == reject

        for token in reject:
            assert not matcher.accept_token(token), f"{token=}"
        assert matcher.accept_token(accept)


def _make_matcher_and_checker(vocab, grammar_str):
    """Create matcher and checker with a custom vocab and grammar."""
    tokenizer_info = xgr.TokenizerInfo(vocab, additional_special_token_ids=[2])
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar_str, tokenizer_info)
    return matcher, _Checker(tokenizer_info.vocab_size)


def test_reject_special_token_by_token():
    """Special tokens not in the Token() set should be rejected."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], "root ::= Token(4)\n"
    )

    check(matcher, accept=4, reject={0, 1, 2, 3})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_reject_special_token_by_exclude_token():
    """ExcludeToken() should reject special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], "root ::= ExcludeToken(3)\n"
    )

    check(matcher, accept=4, reject={0, 1, 2, 3})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_reject_special_token_by_char_class():
    """Byte path should reject special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc", "a"], "root ::= [abc]*\n"
    )

    check(matcher, accept=3, reject={0, 2})  # "bb"
    check(matcher, accept=4, reject={0, 2})  # "cc"
    check(matcher, accept=5, reject={0, 2})  # "a"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2})
    assert matcher.is_terminated()


def test_reject_special_token_by_negative_char_class():
    """Byte path should reject special tokens."""
    matcher, check = _make_matcher_and_checker(["", "</s>", "aa", "bb", "cc"], "root ::= [^b]*\n")

    check(matcher, accept=4, reject={0, 2, 3})  # "cc"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3})
    assert matcher.is_terminated()


def test_reject_special_token_by_recursive_string():
    """Byte path should reject special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "[]", "[", "]", "]]"],
        """
body ::= "" | "[" body "]"
root ::= body
""",
    )

    check(matcher, accept=3, reject={0, 2, 4, 5})  # "["
    check(matcher, accept=3, reject={0, 1, 2, 5})  # "["
    check(matcher, accept=3, reject={0, 1, 2})  # "["
    check(matcher, accept=5, reject={0, 1, 2})  # "]]"
    check(matcher, accept=4, reject={0, 1, 2, 3, 5})  # "]"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4, 5})
    assert matcher.is_terminated()


def test_accept_autodetected_special_token():
    """Token(0) should accept special token ID 0."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], "root ::= Token(0)\n"
    )

    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_additional_special_token():
    """Token(2) should accept special token ID 2."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], "root ::= Token(2)\n"
    )

    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_special_token_then_string():
    """Token followed by string literal: Token(2) "bb" ."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], 'root ::= Token(2) "bb"\n'
    )

    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=3, reject={0, 1, 2, 4})  # "bb"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_special_token_or_string():
    """Alternation: Token(2) | "bb" ."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], 'root ::= Token(2) | "bb"\n'
    )

    check(matcher, accept=2, reject={0, 1, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()

    matcher.reset()

    check(matcher, accept=3, reject={0, 1, 4})  # "bb"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_string_in_special_tokens():
    """String wrapped in special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], "root ::= Token(0) [^]* Token(2)\n"
    )

    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=3, reject={0, 1})  # "bb"
    check(matcher, accept=4, reject={0, 1})  # "cc"
    check(matcher, accept=2, reject={0, 1})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_string_with_suffix_then_special_token():
    """Dynamic string with fixed suffix followed by special token."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"], 'root ::= [^]* "b" Token(2)\n'
    )

    check(matcher, accept=4, reject={0, 1, 2})  # "cc"
    check(matcher, accept=3, reject={0, 1, 2})  # "bb"
    check(matcher, accept=2, reject={0, 1})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_special_token_in_recursive_string():
    """Recursive string with special token in the middle."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "[]", "[", "]", "]]"],
        """
body ::= Token(2) | "[" body "]"
root ::= body
""",
    )

    check(matcher, accept=3, reject={0, 1, 4, 5})  # "["
    check(matcher, accept=3, reject={0, 1, 4, 5})  # "["
    check(matcher, accept=3, reject={0, 1, 4, 5})  # "["
    check(matcher, accept=2, reject={0, 1, 4, 5})  # "["
    check(matcher, accept=5, reject={0, 1, 2, 3})  # "]]"
    check(matcher, accept=4, reject={0, 1, 2, 3, 5})  # "]"
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4, 5})
    assert matcher.is_terminated()


def test_accept_max_tokens_in_special_tokens():
    """String with max_tokens wrapped in special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"],
        """
body[max_tokens=2] ::= [^]*
root ::= Token(2) body Token(2)
""",
    )

    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=3, reject={0, 1})  # "bb"
    check(matcher, accept=4, reject={0, 1})  # "cc"
    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


@pytest.mark.parametrize("budget", [4, 5])
def test_accept_max_chars_in_special_tokens(budget: int):
    """String with max_chars wrapped in special tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"],
        f"""
body[max_chars={budget}] ::= [^]*
root ::= Token(2) body Token(2)
""",
    )

    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=3, reject={0, 1})  # "bb"
    check(matcher, accept=4, reject={0, 1})  # "cc"
    check(matcher, accept=2, reject={0, 1, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_special_tokens_max_tokens():
    """Special tokens under max_tokens."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"],
        """
body[max_tokens=2] ::= Token(2, 3){0, -1}
root ::= Token(0) body Token(0)
""",
    )

    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=2, reject={1, 4})
    check(matcher, accept=3, reject={1, 4})
    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


def test_accept_special_tokens_max_chars():
    """Special tokens under max_chars."""
    matcher, check = _make_matcher_and_checker(
        ["", "</s>", "aa", "bb", "cc"],
        """
body[max_chars=4] ::= Token(2, 3){0, -1}
root ::= Token(0) body Token(0)
""",
    )

    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=2, reject={1, 4})
    check(matcher, accept=3, reject={1, 4})
    check(matcher, accept=2, reject={1, 4})
    check(matcher, accept=3, reject={1, 4})
    check(matcher, accept=0, reject={1, 2, 3, 4})
    check(matcher, accept=STOP_TOKEN_ID, reject={0, 2, 3, 4})
    assert matcher.is_terminated()


if __name__ == "__main__":
    pytest.main(sys.argv)
