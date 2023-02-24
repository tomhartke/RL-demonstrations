from typing import Any, Callable, List


def get_discrete_choose_move_out_checker(possible_outputs: List[Any]) -> Callable[[Any], None]:
    """Returns a function to check the output of choose_move is valid.

    Args:
        possible_outputs: List of possible outputs from choose_move

    Returns:
        A function to check the output of choose_move is valid
    """

    def check_choose_move_output(output: Any) -> None:
        """Checks the output of choose_move is valid.

        Args:
            output: Output of choose_move

        Raises:
            AssertionError: If the output is not a valid move
        """
        assert output in possible_outputs, (
            f"Your choose_move function returned {output}, "
            f"but it should return one of {possible_outputs}!"
        )

    return check_choose_move_output
