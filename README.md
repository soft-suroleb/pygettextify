# gettextify

A command-line tool that automates the preparation of Python code for internationalization (i18n). Instead of manually searching for translatable strings, gettextify analyzes source code via AST, extracts string literals, computes 20 features for each one (length, case, format usage, snake_case/camelCase, JSON/XML/HTML content, etc.), and uses a trained CatBoost model to predict which strings are user-facing and should be wrapped in `_()`. The tool then automatically transforms the source code by adding `_()` calls and the `gettext` import.

**Pipeline:** AST parsing → literal extraction → feature engineering → CatBoost prediction → code transformation.

## Installation

```bash
pip install .
```

## Usage

```bash
# Process a single file (output to stdout)
gettextify app.py

# Process a single file in-place
gettextify app.py --inplace

# Process an entire directory recursively (always in-place)
gettextify src/

# Adjust the prediction threshold (default: 0.78)
gettextify app.py --threshold 0.6

# Verbose output
gettextify app.py --inplace --verbose
```

## Features

- Single-file processing and recursive directory traversal
- Configurable probability threshold
- In-place editing or stdout output
- Skips docstrings and already wrapped strings
- Automatically adds `gettext` import when needed
- Supports `.format()` and `%`-style string detection

## Requirements

- Python >= 3.10
- CatBoost
