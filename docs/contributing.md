# Contributing to CoLA

Thanks for contributing!

## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/wilson-labs/cola.git
cd cola
pip install -e .
pip install -r docs/requirements.txt
```

In order to properly develop for CoLA, you will need to have both [JaX](https://github.com/google/jax#installation)
and [PyTorch](https://pytorch.org/get-started/locally/) installed.


## Our Development Process

### Docstrings
We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
If you are not familiar with this style, look around the code base for examples.


### Unit Tests

We use `pytest` to run unit tests:
```bash
pytest -m "not tricky and not big" tests/
```
The not big excludes tests of very large matrices, and not tricky excludes some known edge cases and situations that are not easy to solve right away. To isolate these tricky cases, just run `pytest -m "tricky" tests/`.
There are also marks for `torch` and `jax` to isolate tests only for that framework.

Tests can be filtered with -k, such as
```bash
pytest -m "not tricky and not big" -k "trace" tests/
```
to get tests involving the trace.


You can also
- run tests within a specific directory, run (e.g.) `pytest tests/algorithms/`.
- run a specific unit test, run (e.g.) `pytest tests/algorithms/test_cg.py::test_cg_vjp`.


### Documentation

CoLA uses sphinx to generate documentation, and ReadTheDocs to host documentation.
To build the documentation locally, ensure that sphinx and its plugins are properly installed (see the [development installation section](#development-installation) for instructions).
Then run:

```bash
cd docs
make html
cd build/html
python -m http.server 8000
```

The documentation will be available at http://localhost:8000.
You will have to rerun the `make html` command every time you wish to update the docs.


## Pull Requests
We greatly appreciate PRs! To minimze back-and-forward communication, please ensure that your PR includes the following:

1. **Code changes.** (the bug fix/new feature/updated documentation/etc.)
1. **Unit tests.** If you are updating any code, you should add an appropraite unit test.
   - If you are fixing a bug, make sure that there's a new unit test that catches the bug.
     (I.e., there should be a new unit test that fails before your bug fix, but passes after your bug fix.
     This ensures that we don't have any accidental regressions in the code base.)
   - If you are adding a new feature, you should add unit tests for this new feature.
1. **Documentation.** Any new objects/methods should have [appropriate docstrings](#docstrings).

Before submitting a PR, ensure the following:
1. **Unit tests pass.** See [the unit tests section](#unit-tests) for more info.
1. **Documentation renders correctly without warnings.** [Build the documentation locally](#documentation) to ensure that your new class/docstrings are rendered correctly. Ensure that sphinx can build the documentation without warnings.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

We accept the following types of issues:
- Bug reports
- Requests for documentation/examples
- Feature requests
- Opportuntities to refactor code
- Performance issues (speed, memory, etc.)


## License

By contributing to CoLA, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
