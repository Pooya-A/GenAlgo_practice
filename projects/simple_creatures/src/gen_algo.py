import torch
import numpy as np


def crossover(parent_1, parent_2):
    """
    Perform crossover between two neural network parents, generating a new set of
    weights and biases for a child network.

    Parameters
    ----------
    parent_1 : nn.Module
        The first parent neural network.
    parent_2 : nn.Module
        The second parent neural network.

    Returns
    -------
    child_params : dict
        A dictionary where the keys are the names of the layers in the child
        network and the values are the new weights and biases for the layers.
    """
    child_params = {}
    layers = zip(parent_1.named_children(), parent_2.named_children())
    for layer_1, layer_2 in layers:
        layer_name = layer_1[0]
        in_features = layer_1[1].in_features
        out_features = layer_1[1].out_features

        crossover_point = np.random.choice(range(out_features))

        combined_weights = torch.concat((layer_1[1].weight[:crossover_point],layer_2[1].weight[crossover_point:]))
        mutation_mask = (torch.rand((out_features, in_features)) < 0.05).int()
        mutation = (torch.rand((out_features, in_features)) * 2 - 1) * mutation_mask
        final_weights = combined_weights + mutation
        child_params[f'{layer_name}.weight'] = final_weights.clone().detach().requires_grad_(False)

        bias = (layer_1[1].bias + layer_2[1].bias) * 0.5
        child_params[f'{layer_name}.bias'] = bias.clone().detach().requires_grad_(False)

    return child_params
