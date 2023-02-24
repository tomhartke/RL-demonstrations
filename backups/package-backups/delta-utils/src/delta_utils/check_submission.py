import datetime
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from delta_utils.hash_game_mechanics import hash_game_mechanics, load_game_mechanics_hash
from delta_utils.utils import find


"""
Example usage (connect4)

    check_submission(
        example_choose_move_input = {"state": get_empty_board(), "pkl_file": user_pkl_file},
        check_choose_move_output = get_discrete_choose_move_out_checker(possible_outputs = [0, 1, 2, 3, 4, 5, 6, 7]),
        current_folder = Path(__file__).parent.resolve(),
    )
"""


def check_submission(
    example_choose_move_input: Dict[str, Any],
    check_choose_move_output: Callable[[Any], None],
    current_folder: Path,
    # DEPRECATED:
    example_state: Any = None,
    expected_choose_move_return_type: Type = None,
    pkl_file: Optional[Any] = None,
    expected_pkl_type: Union[None, Type, Tuple[Type, ...]] = None,
    pkl_checker_function: Optional[Callable] = None,
    choose_move_extra_argument: Optional[Dict[str, Any]] = None,
) -> None:
    """Checks a user submission is valid.

    Args:
        example_choose_move_input: dictionary of {argument_name: argument_value} for choose_move
        check_choose_move_output: function to check the output of choose_move is valid
        current_folder: The folder path of the user's game code (main.py etc)

        DEPRECATED ARGS:
        example_state: Example of the argument to the user's choose_move function
        expected_choose_move_return_type: of the users choose_move_function
        pkl_file: The user's loaded pkl file (None if not using a stored pkl file)
        expected_pkl_type: Expected type of the above (None if not using a stored pkl file)
        pkl_checker_function: The function to check that pkl_file is valid
                                         (None if not using a stored pkl file)
    """
    for arg in [
        example_state,
        expected_choose_move_return_type,
        pkl_file,
        expected_pkl_type,
        pkl_checker_function,
        choose_move_extra_argument,
    ]:
        if arg is not None:
            warnings.warn(
                f"{arg} is deprecated, please remove this argument from check_submission()",
                DeprecationWarning,
            )

    if (current_folder / "game_mechanics_hash.txt").exists():
        game_mechanics_path = current_folder
    elif (current_folder / "game_mechanics" / "game_mechanics_hash.txt").exists():
        game_mechanics_path = current_folder / "game_mechanics"
    else:
        raise FileNotFoundError(
            f"game_mechanics_hash.txt not found in {current_folder} or {current_folder / 'game_mechanics'}"
        )

    # TODO: Add link to GitHub in this error message?
    assert hash_game_mechanics(game_mechanics_path) == load_game_mechanics_hash(
        game_mechanics_path
    ), (
        "You've changed game_mechanics.py, please don't do this! :'( "
        "(if you can't escape this error message, reach out to us on slack)"
    )

    main = find("main")
    assert main.exists(), "You need a main.py file!"
    assert main.is_file(), "main.py isn't a Python file!"

    file_name = main.stem

    pre_import_time = datetime.datetime.now()
    mod = __import__(f"{file_name}", fromlist=["None"])
    time_to_import = (datetime.datetime.now() - pre_import_time).total_seconds()

    # Check importing takes a reasonable amount of time
    assert time_to_import < 2, (
        f"Your main.py file took {time_to_import} seconds to import.\n"
        f"This is much longer than expected.\n"
        f"Please make sure it's not running anything (training, testing etc) outside the "
        f"if __name__ == '__main__': at the bottom of the file"
    )

    # Check the choose_move() function exists
    try:
        choose_move = getattr(mod, "choose_move")
    except AttributeError as e:
        raise AttributeError(f"No function 'choose_move()' found in file {file_name}.py") from e

    # Desiderata:
    # 1. Check the signature matches
    #     a) Check all args in the signature are in the example input
    #     b) Check all args in the example input are in the signature
    #     c) Check the types of all args are correct
    # 2. Check the function runs without error

    # Check the choose_move function signature matches the example inputs given
    fn_signature = inspect.signature(choose_move)

    # Check the function signature has the correct arguments
    for param in fn_signature.parameters.values():
        assert param.name in example_choose_move_input, (
            f"Your choose_move() function has unexpected argument '{param.name}'.\n\n"
            f"The expected arguments are: {list(example_choose_move_input.keys())}\n\n"
        )
        # if not isinstance(param.annotation, type(example_choose_move_input[param.name])):
        #     warnings.warn(
        #         f"Your choose_move() function has argument '{param.name}' with type annotation: '{param.annotation}'.\n\n"
        #         f"The expected type of this argument is: {type(example_choose_move_input[param.name])}\n\n"
        #     )

    # Check all arguments in the example input are in the signature
    for param_name in example_choose_move_input:
        assert param_name in fn_signature.parameters, (
            f"Your choose_move() function is missing argument '{param_name}'.\n\n"
            f"The expected arguments are: {list(example_choose_move_input.keys())}\n\n"
        )

    action = choose_move(**example_choose_move_input)
    check_choose_move_output(action)

    # Least important checks at the bottom!
    # Check there is a TEAM_NAME attribute
    try:
        team_name = getattr(mod, "TEAM_NAME")
    except AttributeError as e:
        raise Exception(f"No TEAM_NAME found in file {file_name}.py") from e

    # Check TEAM_NAME isn't empty
    if len(team_name) == 0:
        raise ValueError(f"TEAM_NAME is empty in file {file_name}.py")

    # Check TEAM_NAME isn't still 'Team Name'
    if team_name == "Team Name":
        raise ValueError(
            f"TEAM_NAME='Team Name' which is what it starts as - "
            f"please change this in file {file_name}.py to your team name!"
        )

    print("Congratulations! Your Repl is ready to submit :)")
