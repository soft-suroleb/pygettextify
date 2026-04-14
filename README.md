# gettextify

A command-line tool that automates the preparation of Python code for internationalization (i18n). Instead of manually searching for translatable strings, gettextify analyzes source code via AST, extracts string literals, computes 20 features for each one (length, case, format usage, snake_case/camelCase, JSON/XML/HTML content, etc.), and uses a trained CatBoost model to predict which strings are user-facing and should be wrapped in `_()`. The tool then automatically transforms the source code by adding `_()` calls and the `gettext` import.

Strings the model is uncertain about are marked with the comment `# i18n: review` for manual review.

**Pipeline:** AST parsing → literal extraction → feature engineering → CatBoost prediction → code transformation.

## Installation

```bash
pip install .
```

## Usage

### Marking strings

Process a file or directory — wraps translatable strings in `_()` and annotates uncertain ones with `# i18n: review`.

```bash
# Process a single file (output to stdout)
gettextify app.py

# Process a single file in-place
gettextify app.py --inplace

# Process an entire directory recursively (always in-place)
gettextify src/

# Adjust the prediction threshold (default: 0.73)
gettextify app.py --threshold 0.6

# Control IN / OUT thresholds independently
gettextify app.py --threshold-in 0.8 --threshold-out 0.6

# Verbose output
gettextify app.py --inplace --verbose
```

### Scanning for gray-zone comments

Find all lines previously marked with `# i18n: review` in a file or directory:

```bash
# Scan a single file
gettextify scan app.py

# Scan a directory recursively
gettextify scan src/
```

Output format: `<file>:<line>: <line content>`, followed by a total count.

## Features

- Single-file processing and recursive directory traversal
- Configurable probability thresholds (`--threshold`, `--threshold-in`, `--threshold-out`)
- In-place editing or stdout output
- Gray-zone detection: uncertain strings are annotated with `# i18n: review` for manual review
- `scan` command to find all gray-zone annotations across a codebase
- Skips docstrings, f-string parts, and already wrapped strings
- Automatically adds `gettext` import when needed
- Supports `.format()` and `%`-style string detection

## Requirements

- Python >= 3.10
- CatBoost
