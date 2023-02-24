from collections import defaultdict
from typing import Callable, Dict, Type


def pkl_checker_value_dict(
    pkl_loader: Callable[[str], Dict], team_name: str, expected_type: Type
) -> None:
    """Checks a dictionary acting as a value lookup table."""
    pkl = pkl_loader(team_name)

    try:
        assert isinstance(
            pkl, expected_type
        ), f"The .pkl file you saved is the wrong type! It should be a {expected_type}"
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Value dictionary file called 'dict_{team_name}.pkl' cannot be found! "
            f"Check the file exists & that the name matches."
        ) from e
    if pkl is not None:
        print(f"It'll be using value function file called 'dict_{team_name}.pkl'")

    if isinstance(pkl, defaultdict):
        assert not callable(
            pkl.default_factory
        ), "Please don't use functions within default dictionaries in your pickle file!"

    assert len(pkl) > 0, "Your dictionary is empty!"

    for k, v in pkl.items():
        assert isinstance(v, (float, int)), (
            f"Your value function dictionary values should all be numbers, "
            f"but for key {k}, the value {v} is of type {type(v)}!"
        )
