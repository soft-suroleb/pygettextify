"""Tests for gettextify.features — feature computation for the CatBoost model."""

import pytest

from gettextify.features import (
    FEATURE_NAMES,
    STOPLIST,
    _has_inner_format,
    _is_camel_case,
    _is_html,
    _is_json,
    _is_snake_case,
    _is_xml,
    _looks_like_path,
    compute_features,
)


# ---------------------------------------------------------------------------
#  Helper predicates
# ---------------------------------------------------------------------------

class TestIsSnakeCase:
    def test_lowercase_snake(self):
        assert _is_snake_case("hello_world") is True

    def test_uppercase_snake(self):
        assert _is_snake_case("HELLO_WORLD") is True

    def test_single_word_lower(self):
        assert _is_snake_case("hello") is True

    def test_single_word_upper(self):
        assert _is_snake_case("HELLO") is True

    def test_mixed_case_not_snake(self):
        assert _is_snake_case("Hello_World") is False

    def test_sentence_not_snake(self):
        assert _is_snake_case("hello world") is False

    def test_with_digits(self):
        assert _is_snake_case("value_2") is True

    def test_empty_string(self):
        assert _is_snake_case("") is False


class TestIsCamelCase:
    def test_lower_camel(self):
        assert _is_camel_case("helloWorld") is True

    def test_upper_camel(self):
        assert _is_camel_case("HelloWorld") is True

    def test_single_word(self):
        assert _is_camel_case("hello") is True

    def test_with_spaces(self):
        assert _is_camel_case("hello world") is False

    def test_with_underscore(self):
        assert _is_camel_case("hello_world") is False

    def test_all_upper(self):
        assert _is_camel_case("HTTP") is True


class TestIsJson:
    def test_valid_json_object(self):
        assert _is_json('{"key": "value"}') is True

    def test_valid_json_array(self):
        assert _is_json('[1, 2, 3]') is True

    def test_not_json(self):
        assert _is_json("hello world") is False

    def test_plain_string_is_json(self):
        assert _is_json('"hello"') is True

    def test_empty_string_not_json(self):
        assert _is_json("") is False


class TestIsXml:
    def test_valid_xml(self):
        assert _is_xml("<root><child/></root>") is True

    def test_not_xml(self):
        assert _is_xml("hello world") is False

    def test_empty_string(self):
        assert _is_xml("") is False


class TestIsHtml:
    def test_valid_html_tag(self):
        assert _is_html("<div>hello</div>") is True

    def test_self_closing(self):
        assert _is_html("<br/>") is True

    def test_not_html(self):
        assert _is_html("hello world") is False

    def test_empty_string(self):
        assert _is_html("") is False


class TestLooksLikePath:
    def test_unix_path(self):
        assert _looks_like_path("/usr/bin/python") is True

    def test_windows_path(self):
        assert _looks_like_path("C:\\Users\\admin") is True

    def test_relative_path(self):
        assert _looks_like_path("src/main.py") is True

    def test_file_with_extension(self):
        assert _looks_like_path("config.yaml") is True

    def test_plain_word(self):
        assert _looks_like_path("hello") is False

    def test_empty_string(self):
        assert _looks_like_path("") is False


class TestHasInnerFormat:
    def test_percent_named(self):
        assert _has_inner_format("%(name)s value") is True

    def test_percent_asctime(self):
        assert _has_inner_format("%(asctime)s [%(levelname)s]") is True

    def test_strftime_year(self):
        assert _has_inner_format("%Y-%m-%d") is True

    def test_strftime_hour(self):
        assert _has_inner_format("%H:%M:%S") is True

    def test_brace_named(self):
        assert _has_inner_format("Hello {name}!") is True

    def test_brace_positional(self):
        assert _has_inner_format("{0} and {1}") is True

    def test_brace_with_format_spec(self):
        assert _has_inner_format("{value:.2f}") is True

    def test_no_format(self):
        assert _has_inner_format("hello world") is False

    def test_empty_string(self):
        assert _has_inner_format("") is False

    def test_percent_s_alone_not_inner_format(self):
        assert _has_inner_format("hello %s") is False

    def test_logging_format(self):
        assert _has_inner_format("%(asctime)s [%(levelname)s] %(message)s") is True


# ---------------------------------------------------------------------------
#  compute_features
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    def test_returns_all_feature_names(self):
        result = compute_features("hello", with_format=False, count=1)
        for name in FEATURE_NAMES:
            assert name in result, f"Missing feature: {name}"

    def test_length(self):
        result = compute_features("hello", with_format=False, count=1)
        assert result["length"] == 5

    def test_empty_string(self):
        result = compute_features("", with_format=False, count=1)
        assert result["is_empty"] == 1
        assert result["length"] == 0

    def test_with_format_flag(self):
        result = compute_features("hello", with_format=True, count=1)
        assert result["with_format"] == 1

    def test_without_format_flag(self):
        result = compute_features("hello", with_format=False, count=1)
        assert result["with_format"] == 0

    def test_count_passed_through(self):
        result = compute_features("hello", with_format=False, count=5)
        assert result["count"] == 5

    def test_global_count(self):
        result = compute_features("hello", with_format=False, count=1, global_count=10)
        assert result["global_count"] == 10

    def test_spaces(self):
        result = compute_features("hello world foo", with_format=False, count=1)
        assert result["spaces"] == 2

    def test_underscore(self):
        result = compute_features("hello_world", with_format=False, count=1)
        assert result["underscore"] == 1

    def test_is_upper(self):
        result = compute_features("HTTP_ERROR", with_format=False, count=1)
        assert result["is_upper"] == 1

    def test_is_lower(self):
        result = compute_features("hello world", with_format=False, count=1)
        assert result["is_lower"] == 1

    def test_is_capital(self):
        result = compute_features("Hello world", with_format=False, count=1)
        assert result["is_capital"] == 1

    def test_not_capital_single_char(self):
        result = compute_features("H", with_format=False, count=1)
        assert result["is_capital"] == 0

    def test_last_punctuation_period(self):
        result = compute_features("Hello.", with_format=False, count=1)
        assert result["last_punctuation"] == 1

    def test_last_punctuation_exclamation(self):
        result = compute_features("Hello!", with_format=False, count=1)
        assert result["last_punctuation"] == 1

    def test_last_punctuation_question(self):
        result = compute_features("Really?", with_format=False, count=1)
        assert result["last_punctuation"] == 1

    def test_no_last_punctuation(self):
        result = compute_features("Hello", with_format=False, count=1)
        assert result["last_punctuation"] == 0

    def test_snake_case_feature(self):
        result = compute_features("hello_world", with_format=False, count=1)
        assert result["snake_case"] == 1

    def test_camel_case_feature(self):
        result = compute_features("helloWorld", with_format=False, count=1)
        assert result["camel_case"] == 1

    def test_json_feature(self):
        result = compute_features('{"key": "val"}', with_format=False, count=1)
        assert result["json"] == 1

    def test_html_feature(self):
        result = compute_features("<div>text</div>", with_format=False, count=1)
        assert result["html"] == 1

    def test_path_feature(self):
        result = compute_features("/usr/bin/python", with_format=False, count=1)
        assert result["path"] == 1

    def test_in_stoplist(self):
        result = compute_features("DEBUG", with_format=False, count=1)
        assert result["in_stoplist"] == 1

    def test_not_in_stoplist(self):
        result = compute_features("Hello world!", with_format=False, count=1)
        assert result["in_stoplist"] == 0

    def test_special_letters_count(self):
        result = compute_features("a!b@c#", with_format=False, count=1)
        assert result["special_letters"] == 3

    def test_with_inner_format_logging(self):
        result = compute_features(
            "%(asctime)s [%(levelname)s] %(message)s",
            with_format=False,
            count=1,
        )
        assert result["with_inner_format"] == 1

    def test_with_inner_format_strftime(self):
        result = compute_features("%Y-%m-%d %H:%M:%S", with_format=False, count=1)
        assert result["with_inner_format"] == 1

    def test_no_inner_format_plain(self):
        result = compute_features("Hello world!", with_format=False, count=1)
        assert result["with_inner_format"] == 0


class TestStoplist:
    def test_stoplist_is_set(self):
        assert isinstance(STOPLIST, set)

    def test_common_entries(self):
        for word in ("DEBUG", "VERSION", "ERROR", "CONFIG", "API_KEY"):
            assert word in STOPLIST

    def test_case_sensitivity(self):
        assert "debug" not in STOPLIST
        assert "DEBUG" in STOPLIST
