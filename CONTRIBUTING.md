# Contributing to Labs 02 — Streaming Learned Bloom Filters

Thanks for your interest in contributing. This is an open-source research project by CypherLabs.

## How to contribute

### Reporting issues
- Use GitHub Issues to report bugs or suggest enhancements.
- Include: what you expected, what happened, and steps to reproduce.

### Adding a new filter variant
1. Add your implementation in `src/` (e.g., `src/my_lbf.py`).
2. Add tests in `tests/test_my_lbf.py`.
3. Add a benchmark section in `README.md` if it produces novel results.
4. Submit a PR.

### Adding a new benchmark
1. Add a `test_my_benchmark.py` in the root.
2. Ensure it runs with `python test_my_benchmark.py`.
3. Document the expected output in the PR.

### Code style
- Follow existing patterns in `src/`.
- Public methods should have docstrings.
- Keep imports minimal.

## Development setup

```bash
git clone https://github.com/AshayK003/learned-bloom-filter
cd learned-bloom-filter
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code of conduct

Be respectful. This is a research project, not a battleground.

## License

By contributing, you agree your contributions will be licensed under the MIT License.
