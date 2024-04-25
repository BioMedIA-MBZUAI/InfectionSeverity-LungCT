"""
Author: Ibrahim Almakky
Date: 17/05/2021

"""

from typing import Optional
from google.protobuf.descriptor import Error
import torch.nn as nn

ACTIVATIONS = {
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "ReLU": nn.ReLU,
    "None": None,
}


class MLP(nn.Module):
    def __init__(
        self,
        layers: list,
        dropout: Optional["list[float]"],
        batch_norm=True,
        activation="ReLU",
        final_activation=nn.Sigmoid,
    ) -> None:
        super().__init__()
        if isinstance(layers, tuple):
            layers = list(layers)

        if len(layers) < 2:
            raise ValueError(
                """The layers for the MLP must have at least
                   two layers for the input and output."""
            )

        try:
            activation = ACTIVATIONS[activation]
        except KeyError:
            msg = "Unsupported hidden activation function."
            raise ValueError(msg)

        if isinstance(final_activation, str):
            try:
                final_activation = ACTIVATIONS[final_activation]
            except KeyError:
                msg = "Unsupported output activation function."
                raise ValueError(msg)

        # Model skeleton
        self.fc_layers = nn.Sequential()

        for i, layer in enumerate(layers[:-1], 0):

            # Validate the layers
            if (not isinstance(layer, int)) and (not isinstance(layers[i + 1], int)):
                raise TypeError("""Each element of the layers list must be an int.""")

            # Layer skeleton
            fc_layer = nn.Sequential()
            # Every layer will have a linear
            fc_layer.add_module("linear", nn.Linear(layers[i], layers[i + 1]))

            # Activation
            if i == len(layers) - 2:
                if final_activation:
                    fc_layer.add_module("activation", final_activation())
            else:
                if activation:
                    fc_layer.add_module("activation", activation())

            # Batch Norm (except for the final layer)
            if batch_norm and i == len(layers) - 2:
                fc_layer.add_module("BatchNorm", nn.BatchNorm1d(layers[i + 1]))

            # Dropout
            try:
                l_dropout = dropout[i]
                fc_layer.add_module("dropout", nn.Dropout(p=l_dropout))
            except (IndexError, TypeError):
                pass

            self.fc_layers.add_module("layer_" + str(i), fc_layer)

    def forward(self, x):
        return self.fc_layers(x)
