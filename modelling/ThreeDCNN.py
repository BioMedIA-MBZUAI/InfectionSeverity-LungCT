"""
Author: Ibrahim Almakky
Date: 07/04/2021

"""
import torch
from torch import nn


class ThreeDCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        inp_dims: list,
        conv_channels=[1, 64, 128, 256],
        kernels=[(7, 11, 11), (5, 7, 7), (3, 3, 3)],
        stride=(1, 2, 2),
        pooling=[2, 2, 2],
        fcn=[128],
        dropout=0,
    ) -> None:
        super().__init__()

        if isinstance(inp_dims, tuple):
            inp_dims = list(inp_dims)

        self.conv_layers = nn.Sequential()
        for layer_num in range(0, len(conv_channels) - 1):
            conv_layer = self.__conv_block__(
                (conv_channels[layer_num], conv_channels[layer_num + 1]),
                kernel=kernels[layer_num],
                stride=stride,
                pooling=tuple(pooling),
            )
            self.conv_layers.add_module("conv_block_" + str(layer_num), conv_layer)

        inp_sample = torch.ones(
            [
                1,
                1,
            ]
            + inp_dims
        )

        # Use the input sampler to caclulate the
        # the output size from the cnn layers
        sample_out = self.conv_layers(inp_sample)
        conv_out_shape = sample_out.shape

        out_size = 1
        for i in range(2, len(conv_out_shape)):
            out_size = out_size * conv_out_shape[i]

        # Handle the FC layers
        self.fully_connected = nn.Sequential()
        for layer_num in range(0, len(fcn)):
            if layer_num == 0:
                fc_layer = nn.Sequential(
                    nn.Linear(out_size * conv_channels[-1], fcn[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(fcn[0]),
                )
                if dropout:
                    fc_layer.add_module("dropout", nn.Dropout(p=dropout))
            else:
                fc_layer = nn.Sequential(
                    nn.Linear(fcn[layer_num - 1], fcn[layer_num]),
                    nn.ReLU(),
                    nn.BatchNorm1d(fcn[layer_num]),
                )
                if dropout:
                    fc_layer.add_module("dropout", nn.Dropout(p=dropout))

            self.fully_connected.add_module("fc_" + str(layer_num), fc_layer)

        self.classification_layer = nn.Sequential(nn.Linear(fcn[-1], num_classes))

        # This variable will hold the latend features
        # that can be used for other purposes
        self.latent_features = None

    def __conv_block__(
        self, conv_cannels: tuple, kernel: tuple, stride: tuple, pooling: tuple
    ):
        if isinstance(kernel, list):
            kernel = tuple(kernel)

        conv_layer = nn.Sequential(
            nn.Conv3d(
                conv_cannels[0],
                conv_cannels[1],
                kernel_size=kernel,
                stride=stride,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(conv_cannels[1]),
            nn.MaxPool3d(pooling),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layers[0](x)
        # print(out.shape)
        for i in range(1, len(self.conv_layers)):
            out = self.conv_layers[i](out)
            # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fully_connected(out)
        self.latent_features = out.clone()
        # print(self.latent_features.shape)
        out = self.classification_layer(out)
        return out

    def get_latent_features(self):
        return self.latent_features


def cnn(params, **kwargs):
    try:
        return ThreeDCNN(
            num_classes=kwargs["num_classes"],
            inp_dims=[
                kwargs["sample_duration"],
                kwargs["sample_size"],
                kwargs["sample_size"],
            ],
            conv_channels=params["conv_channels"],
            kernels=params["kernels"],
            stride=params["stride"],
            pooling=params["pooling"],
            fcn=params["fcn"],
            dropout=params["dropout"],
        )
    except KeyError as k_error:
        raise ValueError("Missing parameters to init 3DCNN") from k_error
