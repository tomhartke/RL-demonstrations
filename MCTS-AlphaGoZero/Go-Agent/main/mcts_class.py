from typing import Any, Optional, Tuple, Dict, List, NamedTuple
import torch
from torch import nn
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
# at end run "tensorboard --logdir runs" in terminal to visualize

# from check_submission import check_submission
from game_mechanics import (
    State,
    is_terminal,
    transition_function,
    reward_function,
    all_legal_moves
)
from agzNetwork import alphaGoZeroNet, preprocess_state, get_all_legal_moves_tensor, get_all_legal_moves_smart

ACTIONSPACE_LENGTH = 82
NodeID = Tuple[Tuple[int], int]

class PlayerMove(NamedTuple):
    """A hashable class representing a move made by a player.
    Can be used as a dictionary key.
    I.e the following is valid:
        d: Dict[PlayerMove, int] = {PlayerMove(color=1, move=2): 100}

    Args:
        color: BLACK or WHITE
        move: integer representing the move made
    """

    color: int
    move: int

class Node:
    def __init__(self, state: State, last_action: Optional[int]):
        self.state = state
        self.last_action = last_action
        self.is_terminal = (is_terminal(state)
                            if last_action is not None else False)
        self.legal_actions = get_all_legal_moves_smart(self.state)

        self.child_node_ids = self._get_possible_children(self.legal_actions)  # a dict of action/state ids

        self.key: NodeID = self.state.recent_moves, self.last_action

    def _get_possible_children(self, legal_actions) -> Dict[int, State]:
        """Gets the possible children of this node."""
        if self.is_terminal:
            return {}
        children = {}
        for action in legal_actions:
            child_recent_moves = tuple( list(self.state.recent_moves) + [PlayerMove(self.state.to_play, action)])
            children[action] = (child_recent_moves, action)
        return children

    # def _get_possible_children(self, legal_actions) -> Dict[int, State]:
    #     """Gets the possible children of this node."""
    #     if self.is_terminal:
    #         return {}
    #     children = {}
    #     for action in legal_actions:
    #         children[action] = transition_function(self.state, action)
    #     return children

class MCTS:

    def __init__(
            self
    ):
        self.root_node = None # Node(initial_state, None)
        self.total_return: Dict[NodeID:float] = {}  # {self.root_node.key: 0.0}
        self.N: Dict[NodeID:int] = {}  # {self.root_node.key: 0}
        self.tree: Dict[NodeID:Node] = {}  # {self.root_node.key: self.root_node}
        self.tree_probs: Dict[NodeID:Node] = {}  # will contain action probs after initialization
        self.tree_pred_values: Dict[NodeID:Node] = {}  # will contain network predicted values after initialization
        self.explore_coeff = 5  # because our boards are smaller, our sample space of actions is smaller, so larger probs, maybe need to reduce coeff.
        self.verbose = 0 # 0 1 2 or 3

    def do_rollout(self, network: nn.Module) -> None:
        """
        Master control for rollout of MCTS policy.
        Assumes network as the structure of alphaGoZeroNetwork and can be called to give action probs.
        """

        if self.verbose > 1:
            print("\nNew rollout started from state", self.root_node.state.recent_moves)

        path_taken, total_return = self._select(network)
        self._backup(path_taken, total_return)

    def _select(self, network: nn.Module) -> Tuple[List[Node], float]:
        """
        Selects a node to simulate from, given the current state
         and tree.

        Returns a list of nodes of the path taken from the root
        to the selected node.

        Otherwise follow PUCT path down tree
        """
        node = self.root_node
        path_taken = [node]

        flag_node_added_to_tree = False
        while not node.is_terminal and not flag_node_added_to_tree:
            node, flag_node_added_to_tree = self._puct_select_node(node, network)
            # this adds chosen node to tree if it is not already contained
            path_taken.append(node)
        # Now path_taken ends with a terminal node, or with a node we just added to the tree, or both

        # Calculate return
        total_return = self.tree_pred_values[node.key]
            # default value is predicted by network. This is from the root node's perspective by construction
        if node.is_terminal:  # Then use the true value of the state, since nn is less accurate
            player_is_black = True if self.root_node.state.to_play == 1 else False
            # HUGE caution. Reward function reports reward from the perspective of black player, not white.
            # so we have to sometime invert this if we want to have perspective of root node
            if player_is_black:  # This gets the total return from the players perspective.
                total_return = reward_function(node.state)
            else:
                total_return = - reward_function(node.state)

        if self.verbose > 1:
            print("     Selected final state: ", path_taken[-1].state.recent_moves)

        return path_taken, total_return

    def Q(self, node_id: NodeID) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    def _puct_select_node(self, node: Node, network: nn.Module) -> Tuple[Node, bool]:
        # we always want to select the next node from the perspective of the current player.
        # if it is opponents turn, then we select next node to maximize minus q
        # if it is players turn, then we directly maximize q.

        # However, the q value saved is the expected return FOR THE PERSON DOING THE ROLLOUt
        # So when it's the opponents turn to choose the action, they will choose the minimum q value rather than max
        opponents_turn = True if self.root_node.state.to_play != node.state.to_play else False
        q_sign = -1.0 if opponents_turn else 1.0
        c_exp = self.explore_coeff

        # First get everything as if its the root players turn, and correct via opponent sign later.
        N_node = self.N[node.key]
        q_default = self.Q(node.key)  # default if we don't have any Q value is to assume it's the parent value
        policy_prob_children = self.tree_probs[node.key]  # numpy array of policy prob of visiting child
        q_children = np.ones(ACTIONSPACE_LENGTH) * (-np.inf) * q_sign
            # will be updated if child in tree
            # Default will be -inf in the optimization step. Multiplying by q_sign now ensures that.
        n_children = np.zeros(ACTIONSPACE_LENGTH)   # will be updated if we have this child in the tree

        # Get the n_child for each child state
        # Iterate through the child states, and if it's in tree, add this to n_children count
        for a, child_node_id in node.child_node_ids.items():  # these are the legal moves
            q_children[a] = q_default  # set it to be a possible action, with default q.
            if child_node_id in self.tree.keys():
                n_children[a] += self.N[child_node_id]
                q_children[a] = self.Q(child_node_id)  # update to best estimate of q if we have an estimate.
        # Now n and q_children hold number of visits, and best estimate q value, and policy_prob_children is policy

        # For puct step, if opponent is the one choosing, use opposite value of q. Choose low q values.
        puct_values = q_sign * q_children + c_exp * policy_prob_children * np.sqrt(N_node)/(1.0 + n_children)

        max_puct = np.amax(puct_values)
        possible_max_actions = np.where(puct_values == max_puct)[0]
        chosen_action = np.random.choice(possible_max_actions)
        resulting_state = transition_function(node.state, chosen_action)
        chosen_node_id = (resulting_state.recent_moves, chosen_action)

        # Put node in tree if necessary
        flag_node_added_to_tree = False
        if not (chosen_node_id in self.tree):
            flag_node_added_to_tree = True
            parent_node_id = node.key
            self.add_node_to_tree_with_probs(resulting_state, chosen_action, parent_node_id, network)

        if self.verbose > 2:
            print("     PUCT chose state: ", chosen_node.state.recent_moves)

        return self.tree[chosen_node_id], flag_node_added_to_tree

    def _backup(self, path_taken: List[Node], total_ep_return: float) -> None:
        """
        Update the action-value estimates of all
         parent nodes in the tree with the return from the
         simulated trajectory.
        """
        if self.verbose > 1:
            print("     Backing up with new total return: ", total_ep_return)
        for node in path_taken:
            if self.verbose > 2:
                print(
                    "        Before backup vals:",
                    node.state.recent_moves,
                    " with N visits ",
                    self.N[node.key],
                    " and total return ",
                    self.total_return[node.key],
                )
            self.total_return[node.key] += total_ep_return
            self.N[node.key] += 1
            if self.verbose > 2:
                print(
                    "        Backed up new node:",
                    node.state.recent_moves,
                    " with new N visits ",
                    self.N[node.key],
                    " and new total return ",
                    self.total_return[node.key],
                )

    def choose_action(self, temperature: float = 0.0) -> int:
        """
        Once we've simulated all the trajectories, we want to
         select the action at the current timestep which
         maximises the action-value estimate.

         Incorporates finite temperature possibly into the choice
        """

        temperature_epsilon = 1e-6
        temp_exponent = min(1.0/(temperature_epsilon+temperature), 20.0)  # cap it to prevent overflow?

        n_children = np.zeros(ACTIONSPACE_LENGTH)  # will be updated if we have this child in the tree
        possible_actions = np.arange(ACTIONSPACE_LENGTH)

        if self.verbose > 0:
            print("     Choosing action in state:", self.root_node.state.recent_moves, " given MCTS search params:")

        # Get the n_child for each child state
        # Iterate through the child states, and if it's in tree, add this to n_children count
        for a, child_node_id in self.root_node.child_node_ids.items():  # these are the legal moves
            if child_node_id in self.tree.keys():
                n_children[a] += self.N[child_node_id]

                if self.verbose > 0:
                    print("        Action:", a, " N_visits:", self.N[child_node_id], "\tQ estim.:",
                            round(self.Q(child_node_id), 4))

        probs_from_n = (n_children ** temp_exponent)/(np.sum(n_children ** temp_exponent))

        chosen_action = np.random.choice(possible_actions,p=probs_from_n)

        if self.verbose > 0:
            print("     Choose action:", chosen_action)

        if self.verbose < 0:  # just print chosen action q value
            chosen_node_id = self.root_node.child_node_ids[chosen_action]
            print("     Q predicted:", np.round(self.Q(chosen_node_id),4), ' playing in state ', self.root_node.state.to_play)

        return chosen_action

    def choose_action_probs(self) -> int:
        """
        Once we've simulated all the trajectories, we want to
         select the action at the current timestep which
         maximises the action-value estimate.

         Incorporates finite temperature possibly into the choice
        """

        n_children = np.zeros(ACTIONSPACE_LENGTH)  # will be updated if we have this child in the tree
        possible_actions = np.arange(ACTIONSPACE_LENGTH)

        if self.verbose > 0:
            print("     Choosing action in state:", self.root_node.state.recent_moves, " given MCTS search params:")

        # Get the n_child for each child state
        # Iterate through the child states, and if it's in tree, add this to n_children count
        for a, child_node_id in self.root_node.child_node_ids.items():  # these are the legal moves
            if child_node_id in self.tree.keys():
                n_children[a] += self.N[child_node_id]

                if self.verbose > 0:
                    print("        Action:", a, " N_visits:", self.N[child_node_id], "\tQ estim.:",
                            round(self.Q(child_node_id), 4))

        probs_from_n = (n_children)/(np.sum(n_children))

        return probs_from_n

    def prune_tree(self, successor_state: State, network: nn.Module) -> None:
        """
        Between steps in the real environment, clear out the old tree.
        But keep the set of states we might visit in future.

        successor_state is the new top of the tree.

        This directly modifies self in place
        """
        # If it's the terminal state we don't care about pruning the tree. We're almost done.
        if is_terminal(successor_state):
            if self.verbose > 0:
                print("     No tree pruning, we are in terminal state")
            return

        # If for some reason we never saw this state, then we need to just reset MCTS
        if not(successor_state.recent_moves in [key[0] for key in self.tree.keys()]):
            self.set_initial_state(successor_state, network, self.Q(self.root_node.key))
            # totally reset tree, but use previous root node Q as default Q value for new root node.

        # Else, we want to find the original node in tree and go from there
        tree_size_before = len(self.tree)
        # Figure out action taken and thus the node
        new_root_index = [key[0] for key in self.tree.keys()].index(successor_state.recent_moves)
        action_taken = list(self.tree.keys())[new_root_index][1]

        # reset the root node, then find all children and copy them into a new tree
        self.root_node = self.tree.get((successor_state.recent_moves, action_taken),
                                       Node(successor_state, action_taken))

        self.N[self.root_node.key] = 1.0  # these won't matter anymore for decisions, but this must be 1 for PUCT
        self.total_return[self.root_node.key] = 0.0  # these won't matter anymore for decisions

        # Build a new tree dictionary, where we will save the new children before replacing old with this
        new_tree = {self.root_node.key: self.root_node}

        prev_added_nodes = {self.root_node.key: self.root_node}
        while prev_added_nodes:
            newly_added_nodes = {}

            for node in prev_added_nodes.values():
                child_nodes = {child_node_id:
                                   self.tree[child_node_id]
                               for action, child_node_id in node.child_node_ids.items()
                               if child_node_id in self.tree}
                new_tree.update(child_nodes)  # adds these to new dictionary
                newly_added_nodes.update(child_nodes)

            prev_added_nodes = newly_added_nodes
            # this runs until you have a cycle where all child nodes are not in the original tree

        # Lastly copy the new tree, returns, and N into the self dictionaries
        self.tree = new_tree
        self.total_return = {key: self.total_return[key] for key in self.tree}
        self.N = {key: self.N[key] for key in self.tree}
        self.tree_probs = {key: self.tree_probs[key] for key in self.tree}
        self.tree_pred_values = {key: self.tree_pred_values[key] for key in self.tree}

        tree_size_after = len(self.tree)
        if self.verbose > 0:
            print("     Tree size: ", tree_size_before, "->", tree_size_after, " after pruning")

    def set_initial_state(self, initial_state: State, network: nn.Module, q_default: float = 0.0):
        self.root_node = Node(initial_state, None)
        self.total_return: Dict[NodeID:float] = {self.root_node.key: q_default}
        self.N: Dict[NodeID:int] = {self.root_node.key: 1.0}  # set to 1 or initial PUCT alg fails
        self.tree: Dict[NodeID:Node] = {self.root_node.key: self.root_node}
        policy_probs, pred_value = network.get_policy_probs_and_value_numpy_one_sample(initial_state)
        self.tree_probs: Dict[NodeID:Node] = {self.root_node.key: policy_probs}
        self.tree_pred_values: Dict[NodeID:Node] = {self.root_node.key: pred_value}

        pass

    def add_node_to_tree_with_probs(self, state: State, last_action: int,
                                    parent_node_id: NodeID, network: nn.Module):
        """
        This sub function calls the neural network forward pass

        CAREFUL. If the new state corresponds to the opponent making a move, then we need to flip the predicted value
        Since the neural network predicts the value for the actor, regardless of who they are.

        """

        node = Node(state, last_action)
        opponents_turn = True if self.root_node.state.to_play != node.state.to_play else False

        self.total_return[node.key] = -self.Q(parent_node_id)  # reverse sign because of alternating game
        self.N[node.key] = 1.0  # this essentially treats the initial guess as a sample
        self.tree[node.key] = node

        policy_probs, pred_value = network.get_policy_probs_and_value_numpy_one_sample(state)
        self.tree_probs[node.key] = policy_probs

        # Get whether we are adding a node that is the opponents turn to play, or our turn to play (root node)
        # If it is opponets turn, we have to flip the value predicted by the network
        # Since we always want the return in the MCTS to be positive for the root node state.
        opponents_turn = True if self.root_node.state.to_play != node.state.to_play else False
        if opponents_turn:
            pred_value = -1.0 * pred_value
        self.tree_pred_values[node.key] = pred_value

        pass
