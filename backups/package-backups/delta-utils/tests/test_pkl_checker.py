import pickle
from collections import defaultdict
from pathlib import Path

import pytest
from src.delta_utils.check_saved_file import pkl_checker_value_dict

HERE = Path(__file__).parent.resolve()


def test_pkl_checker_works() -> None:
    test_game_dir = HERE / "test_game_dir"
    valid_dict = {"test": 1.0, "Test2": 2.0}
    valid_pkl_location = test_game_dir / "test_dict.pkl"
    with open(valid_pkl_location, "wb") as f:
        pickle.dump(valid_dict, f)

    def load_pkl(team_name: str) -> dict:
        assert team_name == "test"
        with open(valid_pkl_location, "rb") as f:
            return pickle.load(f)

    with open(valid_pkl_location, "rb") as f:
        pkl_checker_value_dict(load_pkl, "test", dict)

    valid_pkl_location.unlink()


def def_value() -> float:
    return 0.0


def test_pkl_checker_should_fail() -> None:
    """Case directly from user code."""
    test_game_dir = HERE / "test_game_dir"

    invalid_dict = defaultdict(def_value)
    invalid_dict["test"] = 1.0
    invalid_pkl_location = test_game_dir / "invalid_dict_test.pkl"

    with open(invalid_pkl_location, "wb") as f:
        pickle.dump(invalid_dict, f)

    def load_pkl(team_name: str) -> dict:
        assert team_name == "test"
        with open(invalid_pkl_location, "rb") as f:
            return pickle.load(f)

    with pytest.raises(AssertionError):
        pkl_checker_value_dict(load_pkl, "test", dict)

    invalid_pkl_location.unlink()
