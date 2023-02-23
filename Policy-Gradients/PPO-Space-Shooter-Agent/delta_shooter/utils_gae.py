import torch


def calculate_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        successor_values: torch.Tensor,
        is_terminals: torch.Tensor,
        gamma: float,
        lamda: float,
):
    """
    Calculate the Generalized Advantage Estimator (GAE) for a batch of transitions.

    GAE = \sum_{t=0}^{T-1} (gamma * lamda)^t * (r_{t+1} + gamma * V_{t+1} - V_t)
    """
    N = len(rewards)

    # Gets the delta terms: the TD-errors
    delta_terms = rewards + gamma * successor_values - values

    gamlam = gamma * lamda

    gamlam_geo_series = torch.tensor([gamlam ** n for n in range(N)])

    # Shift the coefficients to the right for each successive row
    full_gamlam_matrix = torch.stack([torch.roll(gamlam_geo_series, shifts=n) for n in range(N)])

    # Sets everything except upper-triangular to 0
    gamlam_matrix = torch.triu(full_gamlam_matrix)

    # Zero out terms that are after an episode termination
    for terminal_index in torch.squeeze(is_terminals.nonzero(), dim=1):
        full_gamlam_matrix[: terminal_index + 1, terminal_index + 1:] = 0

    return torch.matmul(gamlam_matrix, delta_terms)
