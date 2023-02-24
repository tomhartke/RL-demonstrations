from pathlib import Path

from src.delta_utils.utils import find


def test_find() -> None:
    assert find("check_submission") == Path("").resolve() / Path(
        "src/delta_utils/check_submission.py"
    )
