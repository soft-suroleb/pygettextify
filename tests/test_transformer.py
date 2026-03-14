"""Tests for gettextify.transformer — source code wrapping and import injection."""

import pytest

from gettextify.parser import StringLiteral, extract_strings
from gettextify.transformer import add_gettext_import, wrap_strings


# ---------------------------------------------------------------------------
#  wrap_strings
# ---------------------------------------------------------------------------

class TestWrapStrings:
    def test_single_string(self):
        source = 'x = "hello"'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert result == 'x = _("hello")'

    def test_no_literals_returns_unchanged(self):
        source = "x = 42"
        result = wrap_strings(source, [])
        assert result == source

    def test_multiple_strings_same_line(self):
        source = 'x, y = "one", "two"'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert '_("one")' in result
        assert '_("two")' in result

    def test_multiple_strings_different_lines(self):
        source = 'a = "hello"\nb = "world"'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert '_("hello")' in result
        assert '_("world")' in result

    def test_multiline_string(self):
        source = 'x = """hello\nworld"""'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert result.startswith('x = _("""hello')
        assert result.endswith('world""")')

    def test_preserves_other_code(self):
        source = 'x = 42\ny = "hello"\nz = True'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert "x = 42" in result
        assert "z = True" in result
        assert '_("hello")' in result

    def test_preserves_indentation(self):
        source = 'def foo():\n    x = "hello"'
        lits = extract_strings(source)
        non_doc = [l for l in lits if not l.is_docstring]
        result = wrap_strings(source, non_doc)
        assert '    x = _("hello")' in result

    def test_wrapping_subset_of_literals(self):
        source = 'a = "wrap me"\nb = "leave me"'
        lits = extract_strings(source)
        result = wrap_strings(source, [lits[0]])
        assert '_("wrap me")' in result
        assert '"leave me"' in result
        assert '_("leave me")' not in result

    def test_single_quoted_string(self):
        source = "x = 'hello'"
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert result == "x = _('hello')"

    def test_triple_single_quoted(self):
        source = "x = '''multi\nline'''"
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert "_('''multi" in result
        assert "line''')" in result


class TestWrapStringsBottomUp:
    """Ensure bottom-up processing doesn't corrupt offsets."""

    def test_two_strings_on_adjacent_lines(self):
        source = 'a = "first"\nb = "second"'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        lines = result.splitlines()
        assert lines[0] == 'a = _("first")'
        assert lines[1] == 'b = _("second")'

    def test_three_strings_same_line(self):
        source = 'x = ("a", "b", "c")'
        lits = extract_strings(source)
        result = wrap_strings(source, lits)
        assert '_("a")' in result
        assert '_("b")' in result
        assert '_("c")' in result


# ---------------------------------------------------------------------------
#  add_gettext_import
# ---------------------------------------------------------------------------

class TestAddGettextImport:
    def test_adds_import_when_missing(self):
        source = 'x = "hello"'
        result = add_gettext_import(source)
        assert "from gettext import gettext as _" in result

    def test_import_placed_after_existing_imports(self):
        source = "import os\nimport sys\n\nx = 42"
        result = add_gettext_import(source)
        lines = result.splitlines()
        gettext_idx = next(
            i for i, l in enumerate(lines)
            if "from gettext import gettext as _" in l
        )
        os_idx = next(i for i, l in enumerate(lines) if "import os" in l)
        sys_idx = next(i for i, l in enumerate(lines) if "import sys" in l)
        assert gettext_idx > os_idx
        assert gettext_idx > sys_idx

    def test_does_not_duplicate_existing_gettext(self):
        source = "from gettext import gettext as _\nx = 42"
        result = add_gettext_import(source)
        assert result.count("from gettext import gettext as _") == 1

    def test_detects_django_gettext(self):
        source = "from django.utils.translation import gettext as _\nx = 42"
        result = add_gettext_import(source)
        assert result.count("from gettext import gettext as _") == 0

    def test_detects_ugettext(self):
        source = "from django.utils.translation import ugettext as _\nx = 42"
        result = add_gettext_import(source)
        assert result.count("from gettext import gettext as _") == 0

    def test_detects_underscore_assignment(self):
        source = "import gettext\n_ = gettext.gettext\nx = 42"
        result = add_gettext_import(source)
        assert "from gettext import gettext as _" not in result

    def test_adds_at_top_if_no_imports(self):
        source = "x = 42\ny = 10"
        result = add_gettext_import(source)
        lines = result.splitlines()
        assert "from gettext import gettext as _" in lines[0]

    def test_empty_source(self):
        result = add_gettext_import("")
        assert "from gettext import gettext as _" in result

    def test_detects_import_gettext_module(self):
        source = "import gettext\nx = 42"
        result = add_gettext_import(source)
        assert result.count("from gettext import gettext as _") == 0


class TestAddGettextImportIdempotency:
    def test_double_call_idempotent(self):
        source = 'x = "hello"'
        result1 = add_gettext_import(source)
        result2 = add_gettext_import(result1)
        assert result1 == result2
