"""Unit tests for core pure functions in arc_prep_phase.py."""
import pytest
from arc_prep_phase import (
    parse_grid,
    extract_code,
    compile_and_validate,
    evaluate_program,
    vote_outputs,
    apply_safe,
    grids_match,
)


class TestParseGrid:
    def test_simple_json(self):
        assert parse_grid("[[1,2],[3,4]]") == [[1, 2], [3, 4]]

    def test_json_with_prefix(self):
        assert parse_grid("Here is the answer: [[1,2],[3,4]]") == [[1, 2], [3, 4]]

    def test_json_with_markdown(self):
        assert parse_grid("```\n[[5,6],[7,8]]\n```") == [[5, 6], [7, 8]]

    def test_malformed_returns_none(self):
        assert parse_grid("not a grid") is None
        assert parse_grid("") is None
        assert parse_grid("[[1,2") is None

    def test_oversize_grid_returns_none(self):
        # 31x1 — exceeds 30-row cap
        big = "[" + ",".join(["[1]"] * 31) + "]"
        assert parse_grid(big) is None

    def test_invalid_values_returns_none(self):
        # cell value > 9
        assert parse_grid("[[1,2],[3,10]]") is None
        # negative
        assert parse_grid("[[-1,2],[3,4]]") is None
        # string
        assert parse_grid('[["a",2],[3,4]]') is None


class TestExtractCode:
    def test_python_fence(self):
        text = "explanation\n```python\ndef transform(g):\n    return g\n```\nafter"
        code = extract_code(text)
        assert "def transform" in code
        assert "explanation" not in code

    def test_bare_fence(self):
        text = "```\ndef transform(g):\n    return g\n```"
        code = extract_code(text)
        assert "def transform" in code

    def test_no_fence_with_def(self):
        text = "Here:\ndef transform(g):\n    return g"
        code = extract_code(text)
        assert "def transform" in code

    def test_empty_returns_empty(self):
        assert extract_code("") == ""
        assert extract_code("no code here") == ""

    def test_picks_first_python_block(self):
        text = "```python\nfirst = 1\n```\nmore\n```python\nsecond = 2\n```"
        code = extract_code(text)
        assert "first = 1" in code
        assert "second" not in code


class TestCompileAndValidate:
    def test_identity_passes(self, sample_train_identity, identity_code):
        fn = compile_and_validate(extract_code(identity_code), sample_train_identity)
        assert fn is not None
        assert fn([[1, 2]]) == [[1, 2]]

    def test_h_flip_passes(self, sample_train_h_flip):
        code = "def transform(grid):\n    return [list(reversed(r)) for r in grid]"
        fn = compile_and_validate(code, sample_train_h_flip)
        assert fn is not None

    def test_wrong_logic_fails(self, sample_train_h_flip):
        code = "def transform(grid):\n    return grid"
        fn = compile_and_validate(code, sample_train_h_flip)
        assert fn is None

    def test_syntax_error_returns_none(self, sample_train_identity, bad_code):
        fn = compile_and_validate(extract_code(bad_code), sample_train_identity)
        assert fn is None

    def test_empty_code_returns_none(self, sample_train_identity):
        assert compile_and_validate("", sample_train_identity) is None
        assert compile_and_validate(None, sample_train_identity) is None

    def test_no_transform_function(self, sample_train_identity):
        code = "def other_func(grid):\n    return grid"
        assert compile_and_validate(code, sample_train_identity) is None

    def test_infinite_loop_returns_none(self, sample_train_identity):
        code = "def transform(grid):\n    while True: pass\n    return grid"
        # _run_with_timeout caps at 5s — must return None, not hang test
        fn = compile_and_validate(code, sample_train_identity)
        assert fn is None


class TestEvaluateProgram:
    def test_all_pass(self, sample_train_h_flip):
        code = "def transform(grid):\n    return [list(reversed(r)) for r in grid]"
        info = evaluate_program(code, sample_train_h_flip)
        assert info["pass_count"] == 3
        assert info["failing"] == []
        assert info["fn"] is not None

    def test_partial_pass(self):
        train = [
            {"input": [[1, 2]], "output": [[2, 1]]},  # flip — h_flip wins
            {"input": [[1, 2]], "output": [[1, 2]]},  # identity — h_flip fails
        ]
        code = "def transform(grid):\n    return [list(reversed(r)) for r in grid]"
        info = evaluate_program(code, train)
        assert info["pass_count"] == 1
        assert len(info["failing"]) == 1

    def test_all_fail(self, sample_train_h_flip):
        code = "def transform(grid):\n    return grid"
        info = evaluate_program(code, sample_train_h_flip)
        assert info["pass_count"] == 0
        assert len(info["failing"]) == 3

    def test_no_fn_in_code(self, sample_train_h_flip):
        info = evaluate_program("x = 1", sample_train_h_flip)
        assert info["pass_count"] == 0
        assert info["fn"] is None

    def test_empty_code(self, sample_train_h_flip):
        info = evaluate_program("", sample_train_h_flip)
        assert info == {"fn": None, "pass_count": 0, "failing": []}


class TestVoteOutputs:
    def test_majority_wins(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        # a appears 3 times, b once
        result = vote_outputs([a, a, a, b])
        assert result == a

    def test_single_output(self):
        a = [[1, 2]]
        assert vote_outputs([a]) == a

    def test_all_different_returns_one(self):
        # ties broken by max() — at minimum returns one of them
        a = [[1]]
        b = [[2]]
        c = [[3]]
        result = vote_outputs([a, b, c])
        assert result in [a, b, c]


class TestApplySafe:
    def test_normal(self):
        fn = lambda g: g
        assert apply_safe(fn, [[1, 2]]) == [[1, 2]]

    def test_returns_none_on_exception(self):
        fn = lambda g: 1 / 0
        assert apply_safe(fn, [[1]]) is None

    def test_oversize_returns_none(self):
        fn = lambda g: [[1]] * 31
        assert apply_safe(fn, [[1]]) is None

    def test_invalid_values_returns_none(self):
        fn = lambda g: [[15]]  # > 9
        assert apply_safe(fn, [[1]]) is None


class TestGridsMatch:
    def test_equal(self):
        assert grids_match([[1, 2], [3, 4]], [[1, 2], [3, 4]]) is True

    def test_different_values(self):
        assert grids_match([[1, 2]], [[1, 3]]) is False

    def test_different_shapes(self):
        assert grids_match([[1]], [[1, 2]]) is False
        assert grids_match([[1]], [[1], [2]]) is False
