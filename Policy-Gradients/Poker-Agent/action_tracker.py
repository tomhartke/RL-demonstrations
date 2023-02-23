
class ActionCountTracker:
    """
    This class is just used for debugging which actions were taken
     in what proportion during training.
    """
    def __init__(self):
        self.action_counts = {i: 0 for i in range(5)}

    def track(self, action: int):
        self.action_counts[action] += 1

    def __add__(self, other: "ActionCountTracker") -> "ActionCountTracker":
        new_tracker = ActionCountTracker()
        for action in range(5):
            new_tracker.action_counts[action] = self.action_counts[action] + other.action_counts[action]
        return new_tracker

    def __repr__(self) -> str:
        out = "Actions chosen:"
        total_actions = sum(self.action_counts.values())
        for action in range(5):
            out += f" {action}: {round(self.action_counts[action] / total_actions, 3)}"
        return out