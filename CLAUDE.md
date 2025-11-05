# FracTime Development Guide

## Build & Test Commands
```bash
# Install in development mode
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"

# Run tests
pytest                         # Run all tests
pytest tests/test_core.py      # Run tests in specific file
pytest tests/test_core.py::test_sample_data  # Run specific test

# Linting & Formatting
black fractime/ tests/         # Format code
ruff check fractime/ tests/    # Lint code
mypy fractime/                 # Type checking

# Run examples
python examples/forecasting_comparison.py --ticker ^GSPC --window-size 60
```

## Code Style
- Python 3.11+ compatibility
- Line length: 88 characters (Black default)
- Type hints required for all functions
- Comprehensive docstrings for classes and functions
- Error handling with fallbacks for numerical operations
- Use Numba acceleration for performance-critical functions
- Prefer pandas/polars for data manipulation
- Import order: stdlib → third party → local
- Use meaningful variable names following snake_case convention