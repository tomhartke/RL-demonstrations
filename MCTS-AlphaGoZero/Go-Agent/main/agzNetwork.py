from torch import nn
import torch
import numpy as np
from collections import OrderedDict
from game_mechanics import State, all_legal_moves, transition_function
from typing import Tuple
import scipy
import time

# Build 2 headed policy network and value network test to mimic AGZ
# Core code taken from: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/

class ResidualBlock(nn.Module):
    def __init__(self, n_kernels = 20, size_kernels = 3, stride_kernels = 1, pad_kernels = 1):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_kernels, n_kernels, size_kernels, stride_kernels, pad_kernels)
        self.bn_1 = nn.BatchNorm2d(n_kernels)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(n_kernels, n_kernels, size_kernels, stride_kernels, pad_kernels)
        self.bn_2 = nn.BatchNorm2d(n_kernels)  # Skip connection inserted after this block
        self.relu_2 = nn.ReLU()
    def forward(self, x):
        x = x + self.bn_2(self.conv_2(self.relu_1(self.bn_1(self.conv_1(x)))))  # residual connection
        x = self.relu_2(x)
        return x

class alphaGoZeroNet(nn.Module):
    """
    This torch network class is a 2 headed policy/value network using conv. nets for input/processing

    Follows alpha go zero architecture according to delta academy setup
    """

    def __init__(self, n_inputs = 17, n_outputs = 82,
                 n_hidden_layers=6, n_kernels = 80, img_width = 9,  # 9x9 go board here
                 learning_rate=0.01,
                 l2_regularizer=0.0,
                 device='cpu'):
        super(alphaGoZeroNet, self).__init__()

        self.device = device
        self.n_inputs = n_inputs  # the number of images stacked on the input. Will be 8 for go.
        self.n_outputs = n_outputs  # number of action choices
        self.n_hidden_layers = n_hidden_layers  # number of residual blocks
        self.n_kernels = n_kernels  # number in each convolution
        self.img_width = img_width  # size of image or board, assumes square
        self.learning_rate = learning_rate
        self.l2_regularizer = l2_regularizer
        self.action_space = np.arange(self.n_outputs)
        self.num_fully_connected_value_neurons = 300  # fully connected part of value head

        # Generate body layers
        self.layers = OrderedDict()
        # Generate input layer
        input_kernel_size = 3
        input_kernel_stride = 1
        input_padding = 1  # input padding 1 is good because walls are like our own type
        self.layers[str(0)] = nn.Sequential(
            nn.Conv2d(self.n_inputs, self.n_kernels, input_kernel_size, input_kernel_stride, input_padding),
            nn.ReLU()
        )
        for i in range(self.n_hidden_layers):
            self.layers[str(i+1)] = ResidualBlock(n_kernels=self.n_kernels)
        self.body = nn.Sequential(self.layers)

        # Define policy head
        num_policy_kernels = 2
        self.policy_logits = nn.Sequential(
            nn.Conv2d(self.n_kernels, num_policy_kernels, 1, 1, 0),  # just 2 simple kernels
            nn.BatchNorm2d(num_policy_kernels),
            nn.ReLU(),
            nn.Flatten(-3,-1),  # flatten last three dimensions. The 2 kernels, width, and height
            nn.Linear(num_policy_kernels*self.img_width**2, self.n_outputs)
        )

        # Define value head
        num_value_kernels = 1
        num_fully_connected = self.num_fully_connected_value_neurons
        self.value = nn.Sequential(
            nn.Conv2d(self.n_kernels, num_value_kernels, 1, 1, 0),  # just 1 simple kernel
            nn.BatchNorm2d(num_value_kernels),
            nn.ReLU(),
            nn.Flatten(-3,-1),  # flatten last three dimensions. The 2 kernels, width, and height
            nn.Linear(num_value_kernels*self.img_width**2, num_fully_connected),
            nn.ReLU(),
            nn.Linear(num_fully_connected, 1),
            nn.Tanh()
        )

        if self.device == 'cuda':
            self.body.cuda()  # can just directly call .cuda() on networks, but for tensors have to reassign the data name
            self.policy_logits.cuda()
            self.value.cuda()

        # I think this gets all parameters, inheriting from super class
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.l2_regularizer,
                                          )

    def get_predictions(self, state: torch.Tensor,  # preprocessed input tensor
                        legal_moves_tensor  # tensor of 1s and zeros for masking
                        ):
        state_t = state.to(device=self.device)
        body_output = self.body(state_t)  # just forward the body once
        logits = self.policy_logits(body_output)
        legal_moves_tensor = legal_moves_tensor.to(device=self.device)
        # mask legal moves on logits (set disallowed actions to be very negative). Good for cross entropy loss
        logits = torch.mul(logits, legal_moves_tensor) + torch.mul((logits * 0.0 - 10000000) , (1.0 - legal_moves_tensor))
        probs_approx = nn.Softmax(dim=-1)(logits)
        probs = (probs_approx * legal_moves_tensor)/torch.sum(probs_approx * legal_moves_tensor)  # final masking
        val = self.value(body_output)
        return probs, val, logits

    def get_action(self, state, legal_moves_tensor):
        probs = self.get_predictions(state, legal_moves_tensor)[0].detach().cpu().numpy()
        action = np.random.choice(self.action_space, p=probs)
        return action

    def get_policy_probs_and_value_numpy_one_sample(self, state: State) -> Tuple[np.ndarray]:
        # Caution sets network to eval mode for 1 sample
        self.eval()

        with torch.no_grad():
            state_processed = preprocess_state(state)
            state_processed.to(device=self.device)

            legal_moves_tensor = get_all_legal_moves_tensor(state)
            legal_moves_tensor = legal_moves_tensor.to(device=self.device)

            state_processed_batch = torch.unsqueeze(state_processed, dim=0)  # unsqueeze for batch of 1
            legal_moves_tensor_batch = torch.unsqueeze(legal_moves_tensor, dim=0)  # unsqueeze for batch of 1

            policy_probs, pred_values, policy_logits = self.get_predictions(state_processed_batch,
                                                                               legal_moves_tensor_batch)
            # The above assumes batch statistics, so has 1 extra dimension before. need to remove it
            policy_probs_numpy = torch.squeeze(policy_probs, dim=0).cpu().numpy()  # to return to length 82 tensor
            pred_value_numpy = torch.squeeze(pred_values, dim=0).cpu().numpy()[0]

        return policy_probs_numpy, pred_value_numpy

def preprocess_state(state: State) -> torch.Tensor:
    """
    Process input state to feed to nn based on environment

    State object is the board and some additional info, which we will use to construct the input tensor
    """
    current_board = state.board
    current_player = state.to_play
    current_player1or0 = 1 if current_player == 1 else 0  # tells me black or white to play next.
    board_deltas = state.board_deltas  # last index is most recent

    def get_past_8_boards(current_board, board_deltas):
        past_8_boards = []
        past_8_boards.append(current_board)
        temp_board = current_board.copy()
        for _board_ind in range(len(board_deltas)):
            temp_board = np.copy(temp_board - board_deltas[-_board_ind][0])  # start from end, the most recent move
            past_8_boards.append(temp_board)
        while len(past_8_boards) < 8:
            past_8_boards.append(temp_board)  # append the last one until it's full
        return np.array(past_8_boards)

    past_8_boards = get_past_8_boards(current_board, board_deltas)
    past_8_my_boards = np.where(past_8_boards == current_player, 1, 0)
    past_8_op_boards = np.where(past_8_boards == -1 * current_player, 1, 0)
    current_player_board = np.array([np.ones_like(current_board)*current_player1or0])

    board_array = np.concatenate((past_8_my_boards, past_8_op_boards, current_player_board), axis=0)
    network_input_tensor = torch.from_numpy(board_array).type(torch.float32)

    return network_input_tensor

def get_all_legal_moves_tensor(state: State) -> torch.Tensor:
    """
    Used to return a representation of the legal moves as 1s and 0s in a length 82 tensor

    :param state:
    :return:
    """
    legal_moves_numpy = get_all_legal_moves_smart(state)  # a list of integers corresponding to legal moves
    legal_moves_mask = np.array([1 if move in legal_moves_numpy else 0 for move in np.arange(82)])
    legal_moves_tensor = torch.from_numpy(legal_moves_mask).type(torch.float32)
    return legal_moves_tensor

def get_all_legal_moves_smart(state: State) -> np.ndarray:
    """
    This gets all fully surrounded eyes in my pieces, and prevents playing in them.

    It also gets all fully surrounded opponent pieces, and if playing in them does nothing, it remove them too.
    But if playing in the opponent does change the board, then it's an allowed move.
    """

    legal_moves_numpy = all_legal_moves(state.board, state.ko)  # this is a list of integers as a np.ndarray

    board_my_pieces = np.where(state.board == state.to_play, 1, 0)
    board_op_pieces = np.where(state.board == -state.to_play, 1, 0)

    def get_eyes(board_my_pieces):
        board_my_pieces_inverted = np.where(board_my_pieces == 1, 0, 1)
        # Only count absolutely real eyes fully surrounded.
        # Other cases are more complicated to determine if they are real eyes or not.
        # But here we are just trying to have a concrete heuristic that is always true.
        # It's never ever good to play inside your own fully surrounded eyes
        # and it's generally not good to play inside of your opponents fully surrounded eyes,
        # unless doing so does somehow change the board (then I allow it still below).
        disps = [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        shifts = np.array([scipy.ndimage.shift(board_my_pieces, _d, mode='constant', cval=1.0) for _d in disps])
        eyes = (np.prod(shifts, axis=0) * board_my_pieces_inverted)  # 1 if all pieces nearby mine, but empty
        return eyes
    def get_4_cross(board_my_pieces):
        board_my_pieces_inverted = np.where(board_my_pieces == 1, 0, 1)
        # These correspond to pieces that will be immediately recaptured potentially if they are surrounded
        # We want to check if playing on these leaves the board alone. if so, don't allow that move.
        disps = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        shifts = np.array([scipy.ndimage.shift(board_my_pieces, _d, mode='constant', cval=1.0) for _d in disps])
        eyes = (np.prod(shifts, axis=0) * board_my_pieces_inverted)  # 1 if all pieces nearby mine, but empty
        return eyes
    my_eyes = get_eyes(board_my_pieces)
    op_4_cross = get_4_cross(board_op_pieces)
    moves_where_my_eyes = np.where(my_eyes.flatten() == 1)[0]
    moves_where_op_4_cross = np.intersect1d(np.where(op_4_cross.flatten() == 1)[0], legal_moves_numpy)  # only take legal ones

    # For moves filling opponent eyes, check if they result in an unchanged board, and if so, remove them as options
    moves_where_op_4_cross_are_safe = []
    moves_where_op_4_cross_are_not_safe = []
    for move in moves_where_op_4_cross:
        new_board = transition_function(state, move).board
        cross_is_not_safe = True if np.sum(np.abs(state.board - new_board)) > 0 else False
        if not(cross_is_not_safe):
            moves_where_op_4_cross_are_safe.append(move)
        else:  # eye is not safe
            moves_where_op_4_cross_are_not_safe.append(move)
    moves_where_op_4_cross_are_safe = np.array(moves_where_op_4_cross_are_safe)
    moves_where_op_4_cross_are_not_safe = np.array(moves_where_op_4_cross_are_not_safe)

    legal_moves_not_filling_eyes_and_no_pass = np.array([move for move in np.arange(81)  # DOES NOT INCLUDE PASS
                                             if (move in legal_moves_numpy and
                                                 not (move in moves_where_my_eyes) and
                                                 not (move in moves_where_op_4_cross_are_safe)
                                                 )])

    # need to include pass as an option, because sometimes we have legal moves that are not filling full eyes
    # but are still bad things to do, because they kill off our protection in some other way, for example.
    if len(legal_moves_not_filling_eyes_and_no_pass) > 0 :
        legal_moves_not_filling_eyes_and_pass = np.concatenate((legal_moves_not_filling_eyes_and_no_pass, np.array([81])))
    else:
        legal_moves_not_filling_eyes_and_pass = np.array([81])
    return legal_moves_not_filling_eyes_and_pass
