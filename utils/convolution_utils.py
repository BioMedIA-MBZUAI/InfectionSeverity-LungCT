from torch import nn


def calc_conv_out(inp_w, inp_h, model):
    out_h, out_w = inp_h, inp_w
    for layer in model.children():
        out_h, out_w = calc_conv_layer_out(out_h, out_w, layer)
        print(layer.__class__.__name__)
        print(out_h, out_w)
    return out_h, out_w


def calc_conv_layer_out(inp_h, inp_w, layer):
    if isinstance(layer, nn.Conv2d):
        return conv_output_shape((inp_h, inp_w), layer)
    elif isinstance(layer, nn.ConvTranspose2d):
        return convtransp_output_shape((inp_h, inp_w), layer)
    elif isinstance(layer, nn.LeakyReLU):
        return inp_h, inp_w
    elif isinstance(layer, nn.ReLU):
        return inp_h, inp_w
    elif isinstance(layer, nn.BatchNorm2d):
        return inp_h, inp_w
    elif isinstance(layer, nn.Sigmoid):
        return inp_h, inp_w
    elif isinstance(layer, nn.Tanh):
        return inp_h, inp_w
    else:
        raise Exception("Sorry, the convolutional layer type is not supported.")


def conv_output_shape(h_w, layer):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    h = (
        h_w[0]
        + (2 * layer.padding[0])
        - (layer.dilation[0] * (layer.kernel_size[0] - 1))
        - 1
    ) // layer.stride[0] + 1
    w = (
        h_w[1]
        + (2 * layer.padding[1])
        - (layer.dilation[1] * (layer.kernel_size[1] - 1))
        - 1
    ) // layer.stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, layer):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    # H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
    #                \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
    h = (
        (h_w[0] - 1) * layer.stride[0]
        - 2 * layer.padding[0]
        + layer.dilation[0] * (layer.kernel_size[0] - 1)
        + layer.output_padding[0]
        + 1
    )

    # W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
    #                \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1
    w = (
        (h_w[1] - 1) * layer.stride[1]
        - 2 * layer.padding[1]
        + layer.dilation[1] * (layer.kernel_size[1] - 1)
        + layer.output_padding[1]
        + 1
    )

    return h, w