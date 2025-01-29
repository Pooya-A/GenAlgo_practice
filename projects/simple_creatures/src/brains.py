import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layer_parameters=None):
        """
        Initialize a Net object.

        Parameters
        ----------
        layer_parameters : dict, optional
            A dictionary with the weights and biases of the layers of the
            network.
            If None, the network is initialized with random weights and biases.
            Default is None.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 2)

        if layer_parameters is not None:
            with torch.no_grad():
                self.fc1.weight.copy_(layer_parameters['fc1.weight'])
                self.fc2.weight.copy_(layer_parameters['fc2.weight'])
                self.fc3.weight.copy_(layer_parameters['fc3.weight'])
                self.fc4.weight.copy_(layer_parameters['fc4.weight'])
                self.fc1.bias.copy_(layer_parameters['fc1.bias'])
                self.fc2.bias.copy_(layer_parameters['fc2.bias'])
                self.fc3.bias.copy_(layer_parameters['fc3.bias'])
                self.fc4.bias.copy_(layer_parameters['fc4.bias'])

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the network.

        Returns
        -------
        torch.Tensor
            The output of the neural network.
        """

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = torch.sigmoid(x)
        return output
    def get_weights(self):
        """
        Return the weights and biases of the neural network as a dictionary.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the layers and the
            values are the weights and biases of the layers.
        """
        return {
            'fc1.weight': self.fc1.weight,
            'fc2.weight': self.fc2.weight,
            'fc3.weight': self.fc3.weight,
            'fc4.weight': self.fc4.weight,
            'fc1.bias': self.fc1.bias,
            'fc2.bias': self.fc2.bias,
            'fc3.bias': self.fc3.bias,
            'fc4.bias': self.fc4.bias
        }