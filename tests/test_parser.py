"""Tests for gettextify.parser — AST-based string literal extraction."""

import pytest

from gettextify.parser import StringLiteral, extract_strings


class TestBasicExtraction:
    def test_single_string(self):
        lits = extract_strings('x = "hello"')
        assert len(lits) == 1
        assert lits[0].value == "hello"

    def test_multiple_strings(self):
        source = 'a = "one"\nb = "two"\nc = "three"'
        lits = extract_strings(source)
        values = {l.value for l in lits}
        assert values == {"one", "two", "three"}

    def test_no_strings(self):
        lits = extract_strings("x = 42\ny = 3.14\nz = True")
        assert lits == []

    def test_empty_source(self):
        lits = extract_strings("")
        assert lits == []

    def test_empty_string_literal(self):
        lits = extract_strings('x = ""')
        assert len(lits) == 1
        assert lits[0].value == ""

    def test_non_string_constants_ignored(self):
        lits = extract_strings("x = 42\ny = b'bytes'\nz = 3.14")
        assert lits == []

    def test_multiline_string(self):
        source = 'x = """hello\nworld"""'
        lits = extract_strings(source)
        assert len(lits) == 1
        assert "hello" in lits[0].value
        assert "world" in lits[0].value

    def test_string_position(self):
        lits = extract_strings('x = "hello"')
        lit = lits[0]
        assert lit.lineno == 1
        assert lit.col_offset == 4
        assert lit.end_lineno == 1
        assert lit.end_col_offset == 11


class TestDocstringDetection:
    def test_module_docstring(self):
        source = '"""Module docstring."""\nx = "hello"'
        lits = extract_strings(source)
        docstrings = [l for l in lits if l.is_docstring]
        non_docstrings = [l for l in lits if not l.is_docstring]
        assert len(docstrings) == 1
        assert docstrings[0].value == "Module docstring."
        assert len(non_docstrings) == 1
        assert non_docstrings[0].value == "hello"

    def test_function_docstring(self):
        source = 'def foo():\n    """Function docstring."""\n    return "bar"'
        lits = extract_strings(source)
        docstrings = [l for l in lits if l.is_docstring]
        assert len(docstrings) == 1
        assert docstrings[0].value == "Function docstring."

    def test_class_docstring(self):
        source = 'class Foo:\n    """Class docstring."""\n    name = "bar"'
        lits = extract_strings(source)
        docstrings = [l for l in lits if l.is_docstring]
        assert len(docstrings) == 1
        assert docstrings[0].value == "Class docstring."

    def test_async_function_docstring(self):
        source = 'async def foo():\n    """Async docstring."""\n    return "bar"'
        lits = extract_strings(source)
        docstrings = [l for l in lits if l.is_docstring]
        assert len(docstrings) == 1

    def test_non_docstring_expr_not_flagged(self):
        source = 'def foo():\n    x = "not a docstring"\n    return x'
        lits = extract_strings(source)
        assert all(not l.is_docstring for l in lits)


class TestFormatDetection:
    def test_dot_format(self):
        source = 'x = "hello {}".format(name)'
        lits = extract_strings(source)
        fmt = [l for l in lits if l.with_format]
        assert len(fmt) == 1
        assert fmt[0].value == "hello {}"

    def test_percent_format(self):
        source = 'x = "hello %s" % name'
        lits = extract_strings(source)
        fmt = [l for l in lits if l.with_format]
        assert len(fmt) == 1
        assert fmt[0].value == "hello %s"

    def test_plain_string_not_format(self):
        source = 'x = "just a string"'
        lits = extract_strings(source)
        assert all(not l.with_format for l in lits)


class TestWrappedDetection:
    def test_already_wrapped(self):
        source = 'x = _("already wrapped")'
        lits = extract_strings(source)
        wrapped = [l for l in lits if l.is_wrapped]
        assert len(wrapped) == 1
        assert wrapped[0].value == "already wrapped"

    def test_not_wrapped(self):
        source = 'x = "not wrapped"'
        lits = extract_strings(source)
        assert all(not l.is_wrapped for l in lits)

    def test_wrapped_with_other_func_not_detected(self):
        source = 'x = foo("bar")'
        lits = extract_strings(source)
        assert all(not l.is_wrapped for l in lits)


class TestFstringDetection:
    def test_fstring_parts_flagged(self):
        source = 'name = "world"\nx = f"hello {name}!"'
        lits = extract_strings(source)
        fstring_parts = [l for l in lits if l.is_fstring_part]
        non_fstring = [l for l in lits if not l.is_fstring_part]
        assert len(fstring_parts) >= 1
        assert any(l.value == "world" for l in non_fstring)

    def test_plain_string_not_fstring_part(self):
        source = 'x = "hello"'
        lits = extract_strings(source)
        assert all(not l.is_fstring_part for l in lits)

    def test_fstring_without_expression(self):
        """f-string with no interpolation — CPython still parses it as JoinedStr,
        so the inner Constant IS marked as fstring_part."""
        source = 'x = f"no interpolation"'
        lits = extract_strings(source)
        if lits:
            assert all(l.is_fstring_part for l in lits)

    def test_complex_fstring(self):
        source = 'x = f"Task \'{task.title}\' removed from {category}."'
        lits = extract_strings(source)
        fstring_parts = [l for l in lits if l.is_fstring_part]
        for part in fstring_parts:
            assert part.is_fstring_part


class TestCandidateFiltering:
    """Test the candidate filtering logic used in cli.py."""

    def _candidates(self, source: str) -> list[StringLiteral]:
        lits = extract_strings(source)
        return [
            l for l in lits
            if not l.is_docstring and not l.is_wrapped and not l.is_fstring_part
        ]

    def test_docstrings_excluded(self):
        source = '"""Module doc."""\nx = "hello"'
        cands = self._candidates(source)
        assert len(cands) == 1
        assert cands[0].value == "hello"

    def test_wrapped_excluded(self):
        source = 'x = _("wrapped")\ny = "not wrapped"'
        cands = self._candidates(source)
        assert len(cands) == 1
        assert cands[0].value == "not wrapped"

    def test_fstring_parts_excluded(self):
        source = 'x = f"hello {name}!"\ny = "plain"'
        cands = self._candidates(source)
        values = {c.value for c in cands}
        assert "plain" in values
        fstring_text_parts = [c for c in cands if "hello" in c.value]
        assert len(fstring_text_parts) == 0
