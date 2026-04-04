# Agent Instructions for Mad World Codebase

Welcome to the `mad-world` repository. Please adhere to the following guidelines
when working in this codebase.

## 1. Tooling and Workflow
* **Package Manager:** `uv` is used for all dependency management and
  environment isolation. Do not use `pip` or `venv` directly. Add dependencies
  with `uv add` or `uv add --dev`.
* **Task Runner:** A `Makefile` is provided for common operations.
  * Run `make check` and `make test` frequently to run Ruff (linting), Mypy
    (type checking), and Pytest (testing) against the current uncommitted
    codebase.
  * Run `make format` to automatically format the code and fix safe linting
    errors. If the linter makes a change (e.g., moving something to a
    typechecking block because it isn't used at runtime) DO NOT REVERT IT unless
    you are making NEW changes that would require it.
* **Pre-commit:** The project uses `pre-commit`. Hooks (like Mypy) are
  configured to run locally via `uv` to share the same environment and
  dependencies as the project.

## 2. Code Style and Typing
* **Strict Typing:** The project enforces strict Mypy checks. Ensure all new
  functions, methods, and variables are properly typed.
* **Python 3.12+ Type Parameters:** This project uses the new type parameter
  syntax introduced in Python 3.12 (PEP 695).
  * Prefer the square-bracket syntax for generics (e.g., `def get_attr[T](...)`)
    instead of manual `TypeVar` declarations.
  * Use the built-in `type[]` instead of the deprecated `typing.Type[]`.
  * **Abstract Classes:** Note that Mypy does not allow passing abstract base
    classes (like `GamePlayer`) to parameters typed as `type[T]`. If a utility
    function must accept an ABC as a filter or type check, use `Any` for that
    specific parameter to avoid `type-abstract` diagnostic errors.
* **Pydantic Models:** The codebase heavily utilizes Pydantic `BaseModel`s for
  state management.
  * Always use **keyword arguments** when instantiating Pydantic models (e.g.,
    `OperationDefinition(name="domestic-investment", ...)`), as positional
    arguments will fail strict Mypy checks.
* **Line Length (E501):** Ruff is configured with a line length of 80
  characters. The formatter intentionally does *not* auto-wrap long string
  literals (like game text or descriptions). If you encounter an `E501 Line too
  long` error on a string, you must **manually break it up** using implicit
  string concatenation (wrapped in parentheses).
* **Diagnostic Suppressions:** DO NOT use diagnostic suppressions (like `#
  noqa:...` or `# type: ...` without first using the `ask_user` tool to ask the
  user whether this is acceptable or if you should investigate a more
  appropriate fix.
* **Exception handling:** DO NOT use `except Exception`. You should catch
  specific exception subtypes that you expect the code to emit and propagate any
  others up the stack.

## 3. Game Architecture
* **State Mutations:** All changes to the core game state variables (such as
  `doomsday_clock`, `gdp`, and `influence`) **must** be routed through the event
  logging mechanic. Do not modify these variables directly in game logic
  functions like `process_bid` or `resolve_operation`. Instead, construct a
  `GameEvent` object and apply it using `game.apply_event(event)`.
* **Hidden Information:** For multiplayer contexts involving fog-of-war, prefer
  **Dynamic Exclusion** when dumping Pydantic models to JSON (e.g., passing
  nested dictionary exclusions to the `exclude` argument in `model_dump_json()`)
  rather than statically excluding fields at the model level with
  `Field(exclude=True)`.

## 4. Testing
* **Test Framework:** `pytest` is used for all testing.
* **Trivial Players:** Simple, predictable bot implementations for testing and
  baseline gameplay should be placed in `src/mad_world/trivial_players.py`.
  Their corresponding unit tests belong in `tests/test_trivial_players.py`.
* **Data-Driven Tests:** Integration and game loop tests in `tests/test_core.py`
  utilize a data-driven approach using `pytest.mark.parametrize`. When adding
  new players or mechanics, add new match-up scenarios to the `TEST_CASES` list.
* **Reasoning:** When fixing or updating tests, do not assume the existing
  outcome is correct. Always reason through the game logic from first principles
  to determine the expected result (e.g., tracking influence budgets, clock
  impact, and GDP over rounds).
