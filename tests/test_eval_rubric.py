"""Rubric-level tests for the v2 evaluation harness.

The judge itself must be tested: a buggy grader silently corrupts every
eval run (see the 12/36 incident - keyword checks that were stricter
than reality).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval_v2 import _check_keywords, _normalize_for_match


class TestNormalizeForMatch:
    def test_chinese_vs_arabic_numerals(self):
        # Corpus writes 十五天; LLMs answer "15 天". Same fact.
        assert _normalize_for_match("十五天") == _normalize_for_match("15 天")
        assert _normalize_for_match("二十天") == _normalize_for_match("20天")
        assert _normalize_for_match("十二位") == _normalize_for_match("12位")
        assert _normalize_for_match("三十天") in _normalize_for_match("提前 30 天书面")

    def test_bare_ten_and_compound(self):
        assert _normalize_for_match("十") == "10"
        assert _normalize_for_match("十天") == "10天"
        assert _normalize_for_match("一百二十") == "120"

    def test_whitespace_and_case(self):
        assert _normalize_for_match("Hospital Certificate") == _normalize_for_match(
            "hospital  certificate"
        )

    def test_english_untouched(self):
        assert _normalize_for_match("password") == "password"


class TestCheckKeywords:
    def test_numeral_paraphrase_counts_as_hit(self):
        missing: list[str] = []
        ok = _check_keywords(
            "每位员工每年享有 15 天带薪年假，满 10 年为 20 天。[1]",
            ["十五天", "二十天"],
            missing,
        )
        assert ok and not missing

    def test_wrong_number_still_fails(self):
        # 事实错误必须被抓：20 天 ≠ 15 天，即使数字形式相同。
        missing: list[str] = []
        ok = _check_keywords(
            "每位员工每年享有 20 天带薪年假。[1]", ["十五天"], missing
        )
        assert not ok and missing == ["十五天"]

    def test_missing_keyword_reported_verbatim(self):
        missing: list[str] = []
        ok = _check_keywords("无关回答", ["十五天", "密码"], missing)
        assert not ok and "十五天" in missing

    def test_no_keywords_always_passes(self):
        assert _check_keywords("anything", [], [])
