import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Parameter

# See TPGN paper for notation
class LstmCellS(nn.Module):
    def __init__(self, input_size, hidden_size_s, hidden_size_u, bias=True):
        super(LstmCellS, self).__init__()
        self.input_size    = input_size
        self.hidden_size_s = hidden_size_s
        self.hidden_size_u = hidden_size_u
        self.bias          = bias

        self.weight_d1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, input_size))
        self.weight_d1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, input_size))
        self.weight_d1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, input_size))
        self.weight_d1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, input_size))
        self.weight_u1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_s, hidden_size_s))
        self.weight_u1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_s, hidden_size_s))
        self.weight_u1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_s, hidden_size_s))
        self.weight_u1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_s, hidden_size_s))
        self.weight_w1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_u))
        self.weight_w1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_u))
        self.weight_w1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_u))
        self.weight_w1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s, hidden_size_u))

        # TODO: Combine bias_d1f, bias_u1f, and bias_w1f into bias_1f.
        # TODO: Combine bias_d1i, bias_u1i, and bias_w1i into bias_1i.
        # TODO: Combine bias_d1o, bias_u1o, and bias_w1o into bias_1o.
        # TODO: Combine bias_d1c, bias_u1c, and bias_w1c into bias_1c.

        self.bias_d1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_d1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_d1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_d1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_u1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_u1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_u1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_u1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_w1f = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_w1i = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_w1o = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))
        self.bias_w1c = Parameter(torch.Tensor(hidden_size_s, hidden_size_s))

        self.reset_parameters()

    def forward(self, input, hidden_s, hidden_u, hidden_c):
        # input (batch, embed_d), hidden_s (batch, hidden_d, hidden_d), hidden_u (batch, hidden_d), hidden_c (batch, hidden_d, hidden_d)

        if len(input.shape) == 1:
            input = input.unsqueeze(0)    # (batch, embed_d)

        if len(hidden_s.shape) == 2:
            hidden_s = hidden_s.unsqueeze(0)    # (batch, hidden_d, hidden_d)

        if len(hidden_u.shape) == 1:
            hidden_u = hidden_u.unsqueeze(0)    # (batch, hidden_d)

        if len(hidden_c.shape) == 2:
            hidden_c = hidden_c.unsqueeze(0)    # (batch, hidden_d, hidden_d)

        # TODO: Combine bias_d1f, bias_u1f, and bias_w1f into bias_1f.
        # TODO: Combine bias_d1i, bias_u1i, and bias_w1i into bias_1i.
        # TODO: Combine bias_d1o, bias_u1o, and bias_w1o into bias_1o.
        # TODO: Combine bias_d1c, bias_u1c, and bias_w1c into bias_1c.

        w1f_dot_hidden_u          = torch.matmul(self.weight_w1f, hidden_u.t()).permute(2, 0, 1)                  # (batch, hidden_d, hidden_d)
        d1f_dot_input             = torch.matmul(self.weight_d1f, input.t()).permute(2, 0, 1)                     # (batch, hidden_d, hidden_d)
        reshaped_u1f              = self.weight_u1f.view(self.hidden_size_s ** 2, -1)                             # (hidden_d **2, hidden_d **2)
        reshaped_hidden_s         = hidden_s.view(-1, self.hidden_size_s ** 2)                                    # (batch, hidden_d ** 2)
        reshaped_u1f_dot_hidden_s = torch.matmul(reshaped_u1f, reshaped_hidden_s.t()).permute(1, 0)               # (batch, hidden_d ** 2)
        u1f_dot_hidden_s          = reshaped_u1f_dot_hidden_s.view(-1, self.hidden_size_s, self.hidden_size_s)    # (batch, hidden_d, hidden_d)

        forgetGate = torch.sigmoid(w1f_dot_hidden_u - d1f_dot_input + u1f_dot_hidden_s
                                   + self.bias_w1f + self.bias_d1f + self.bias_u1f)    # (batch, hidden_d, hidden_d)

        w1i_dot_hidden_u          = torch.matmul(self.weight_w1i, hidden_u.t()).permute(2, 0, 1)                   # (batch, hidden_d, hidden_d)
        d1i_dot_input             = torch.matmul(self.weight_d1i, input.t()).permute(2, 0, 1)                      # (batch, hidden_d, hidden_d)
        reshaped_u1i              = self.weight_u1i.view(self.hidden_size_s ** 2, -1)                              # (hidden_d **2, hidden_d **2)
        reshaped_hidden_s         = hidden_s.view(-1, self.hidden_size_s ** 2)                                     # (batch, hidden_d ** 2)
        reshaped_u1i_dot_hidden_s = torch.matmul(reshaped_u1i, reshaped_hidden_s.t()).permute(1, 0)                # (batch, hidden_d ** 2)
        u1i_dot_hidden_s          = reshaped_u1i_dot_hidden_s.view(-1, self.hidden_size_s, self.hidden_size_s)     # (batch, hidden_d, hidden_d)

        inputGate = torch.sigmoid(w1i_dot_hidden_u - d1i_dot_input + u1i_dot_hidden_s
                                  + self.bias_w1i + self.bias_d1i + self.bias_u1i)    # (batch, hidden_d, hidden_d)

        w1o_dot_hidden_u          = torch.matmul(self.weight_w1o, hidden_u.t()).permute(2, 0, 1)                   # (batch, hidden_d, hidden_d)
        d1o_dot_input             = torch.matmul(self.weight_d1o, input.t()).permute(2, 0, 1)                      # (batch, hidden_d, hidden_d)
        reshaped_u1o              = self.weight_u1o.view(self.hidden_size_s ** 2, -1)                              # (hidden_d **2, hidden_d **2)
        reshaped_hidden_s         = hidden_s.view(-1, self.hidden_size_s ** 2)                                     # (batch, hidden_d ** 2)
        reshaped_u1o_dot_hidden_s = torch.matmul(reshaped_u1o, reshaped_hidden_s.t()).permute(1, 0)                # (batch, hidden_d ** 2)
        u1o_dot_hidden_s          = reshaped_u1o_dot_hidden_s.view(-1, self.hidden_size_s, self.hidden_size_s)     # (batch, hidden_d, hidden_d)

        outputGate = torch.sigmoid(w1o_dot_hidden_u - d1o_dot_input + u1o_dot_hidden_s
                                  + self.bias_w1o + self.bias_d1o + self.bias_u1o)    # (batch, hidden_d, hidden_d)

        w1c_dot_hidden_u          = torch.matmul(self.weight_w1c, hidden_u.t()).permute(2, 0, 1)                   # (batch, hidden_d, hidden_d)
        d1c_dot_input             = torch.matmul(self.weight_d1c, input.t()).permute(2, 0, 1)                      # (batch, hidden_d, hidden_d)
        reshaped_u1c              = self.weight_u1c.view(self.hidden_size_s ** 2, -1)                              # (hidden_d **2, hidden_d **2)
        reshaped_hidden_s         = hidden_s.view(-1, self.hidden_size_s ** 2)                                     # (batch, hidden_d ** 2)
        reshaped_u1c_dot_hidden_s = torch.matmul(reshaped_u1c, reshaped_hidden_s.t()).permute(1, 0)                # (batch, hidden_d ** 2)
        u1c_dot_hidden_s          = reshaped_u1c_dot_hidden_s.view(-1, self.hidden_size_s, self.hidden_size_s)     # (batch, hidden_d, hidden_d)

        gate = torch.tanh(w1c_dot_hidden_u - d1c_dot_input + u1c_dot_hidden_s
                          + self.bias_w1c + self.bias_d1c + self.bias_u1c)    # (batch, hidden_d, hidden_d)

        cellState = (forgetGate * hidden_c) + (inputGate * gate)    # (batch, hidden_d, hidden_d)

        out_s = outputGate * torch.tanh(cellState)    # (batch, hidden_d, hidden_d)

        return out_s, cellState

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size_s * self.hidden_size_s)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

# See TPGN paper for notation    
class LstmCellU(nn.Module):
    def __init__(self, input_size, hidden_size_u, hidden_size_s, bias=True):
        super(LstmCellU, self).__init__()
        self.input_size    = input_size
        self.hidden_size_u = hidden_size_u
        self.hidden_size_s = hidden_size_s
        self.bias          = bias

        self.weight_d2f = Parameter(torch.Tensor(hidden_size_u, input_size))
        self.weight_d2i = Parameter(torch.Tensor(hidden_size_u, input_size))
        self.weight_d2o = Parameter(torch.Tensor(hidden_size_u, input_size))
        self.weight_d2c = Parameter(torch.Tensor(hidden_size_u, input_size))
        self.weight_u2f = Parameter(torch.Tensor(hidden_size_u, hidden_size_u))
        self.weight_u2i = Parameter(torch.Tensor(hidden_size_u, hidden_size_u))
        self.weight_u2o = Parameter(torch.Tensor(hidden_size_u, hidden_size_u))
        self.weight_u2c = Parameter(torch.Tensor(hidden_size_u, hidden_size_u))
        self.weight_w2f = Parameter(torch.Tensor(hidden_size_s))
        self.weight_w2i = Parameter(torch.Tensor(hidden_size_s))
        self.weight_w2o = Parameter(torch.Tensor(hidden_size_s))
        self.weight_w2c = Parameter(torch.Tensor(hidden_size_s))

        # TODO: Combine bias_d2f, bias_u2f, and bias_w2f into bias_2f.
        # TODO: Combine bias_d2i, bias_u2i, and bias_w2i into bias_2i.
        # TODO: Combine bias_d2o, bias_u2o, and bias_w2o into bias_2o.
        # TODO: Combine bias_d2c, bias_u2c, and bias_w2c into bias_2c.

        self.bias_d2f = Parameter(torch.Tensor(hidden_size_u))
        self.bias_d2i = Parameter(torch.Tensor(hidden_size_u))
        self.bias_d2o = Parameter(torch.Tensor(hidden_size_u))
        self.bias_d2c = Parameter(torch.Tensor(hidden_size_u))
        self.bias_u2f = Parameter(torch.Tensor(hidden_size_u))
        self.bias_u2i = Parameter(torch.Tensor(hidden_size_u))
        self.bias_u2o = Parameter(torch.Tensor(hidden_size_u))
        self.bias_u2c = Parameter(torch.Tensor(hidden_size_u))
        self.bias_w2f = Parameter(torch.Tensor(hidden_size_u))
        self.bias_w2i = Parameter(torch.Tensor(hidden_size_u))
        self.bias_w2o = Parameter(torch.Tensor(hidden_size_u))
        self.bias_w2c = Parameter(torch.Tensor(hidden_size_u))

        self.reset_parameters()

    def forward(self, input, hidden_u, hidden_s, hidden_c):
        # input (batch, embed_d), hidden_u (batch, hidden_d), hidden_s (batch, hidden_d, hidden_d), hidden_c (batch, hidden_d)

        if len(input.shape) == 1:
            input = input.unsqueeze(0)    # (batch, embed_d)

        if len(hidden_u.shape) == 1:
            hidden_u = hidden_u.unsqueeze(0)    # (batch, hidden_d)

        if len(hidden_s.shape) == 2:
            hidden_s = hidden_s.unsqueeze(0)    # (batch, hidden_d, hidden_d)

        if len(hidden_c.shape) == 1:
            hidden_c = hidden_c.unsqueeze(0)    # (batch, hidden_d)

        # TODO: Combine bias_d2f, bias_u2f, and bias_w2f into bias_2f.
        # TODO: Combine bias_d2i, bias_u2i, and bias_w2i into bias_2i.
        # TODO: Combine bias_d2o, bias_u2o, and bias_w2o into bias_2o.
        # TODO: Combine bias_d2c, bias_u2c, and bias_w2c into bias_2c.

        hidden_s_dot_w2f = torch.matmul(hidden_s, self.weight_w2f)                      # (batch, hidden_d)
        d2f_dot_input    = torch.matmul(self.weight_d2f, input.t()).permute(1, 0)       # (batch, hidden_d)
        u2f_dot_hidden_u = torch.matmul(self.weight_u2f, hidden_u.t()).permute(1, 0)    # (batch, hidden_d)

        forgetGate = torch.sigmoid(hidden_s_dot_w2f - d2f_dot_input + u2f_dot_hidden_u
                                   + self.bias_w2f + self.bias_d2f + self.bias_u2f)     # (batch, hidden_d)

        hidden_s_dot_w2i = torch.matmul(hidden_s, self.weight_w2i)                      # (batch, hidden_d)
        d2i_dot_input    = torch.matmul(self.weight_d2i, input.t()).permute(1, 0)       # (batch, hidden_d)
        u2i_dot_hidden_u = torch.matmul(self.weight_u2i, hidden_u.t()).permute(1, 0)    # (batch, hidden_d)

        inputGate = torch.sigmoid(hidden_s_dot_w2i - d2i_dot_input + u2i_dot_hidden_u
                                  + self.bias_w2i + self.bias_d2i + self.bias_u2i)      # (batch, hidden_d)

        hidden_s_dot_w2o = torch.matmul(hidden_s, self.weight_w2o)                      # (batch, hidden_d)
        d2o_dot_input    = torch.matmul(self.weight_d2o, input.t()).permute(1, 0)       # (batch, hidden_d)
        u2o_dot_hidden_u = torch.matmul(self.weight_u2o, hidden_u.t()).permute(1, 0)    # (batch, hidden_d)

        outputGate = torch.sigmoid(hidden_s_dot_w2o - d2o_dot_input + u2o_dot_hidden_u
                                  + self.bias_w2o + self.bias_d2o + self.bias_u2o)      # (batch, hidden_d)

        hidden_s_dot_w2c = torch.matmul(hidden_s, self.weight_w2c)                      # (batch, hidden_d)
        d2c_dot_input    = torch.matmul(self.weight_d2c, input.t()).permute(1, 0)       # (batch, hidden_d)
        u2c_dot_hidden_u = torch.matmul(self.weight_u2c, hidden_u.t()).permute(1, 0)    # (batch, hidden_d)

        gate = torch.tanh(hidden_s_dot_w2c - d2c_dot_input + u2c_dot_hidden_u
                          + self.bias_w2c + self.bias_d2c + self.bias_u2c)              # (batch, hidden_d)

        cellState = (forgetGate * hidden_c) + (inputGate * gate)    # (batch, hidden_d)

        out_u = outputGate * torch.tanh(cellState)    # (batch, hidden_d)

        return out_u, cellState

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size_u)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_layers):  # for number of layers times
            self.layer_dict['conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters, padding=1,
                                                             bias=self.use_bias)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = F.relu(out)  # apply relu
            print(out.shape)
            if self.dim_reduction_type == 'strided_convolution':  # if dim reduction is strided conv, then add a strided conv
                self.layer_dict['dim_reduction_strided_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=2,
                                                                                       dilation=1)

                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # use strided conv to get an output
                out = F.relu(out)  # apply relu to the output
            elif self.dim_reduction_type == 'dilated_convolution':  # if dim reduction is dilated conv, then add a dilated conv, using an arbitrary dilation rate of i + 2 (so it gets smaller as we go, you can choose other dilation rates should you wish to do it.)
                self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=1,
                                                                                       dilation=i + 2)
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](
                    out)  # run dilated conv on input to get output
                out = F.relu(out)  # apply relu on output

            elif self.dim_reduction_type == 'max_pooling':
                self.layer_dict['dim_reduction_max_pool_{}'.format(i)] = nn.MaxPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                self.layer_dict['dim_reduction_avg_pool_{}'.format(i)] = nn.AvgPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

            print(out.shape)
        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out,
                                        2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_layers):  # for number of layers

            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            out = F.relu(out)  # pass conv outputs through ReLU
            if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # pass previous outputs through a strided convolution indexed i
                out = F.relu(out)  # pass strided conv outputs through ReLU

            elif self.dim_reduction_type == 'dilated_convolution':
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                out = F.relu(out)

            elif self.dim_reduction_type == 'max_pooling':
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, fine_tune=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune)

    def get_encoder_type(self):
        return "resnet"

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DensenetEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune=True):
        super(DensenetEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        densenet = torchvision.models.densenet121(pretrained=True)
        
        # Remove linear layer, pool layer cannot be found
        modules = list(densenet.children())[:-1]
        self.densenet = nn.Sequential(*modules)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune)
    
    def get_encoder_type(self):
        return "densenet"
    
    def forward(self, images):
        out = self.densenet(images)    # (batch_size, 1024, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 1024, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 1024)
        return out
        
    def fine_tune(self, fine_tune=True):
        for p in self.densenet.parameters():
            p.requires_grad = False
            
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.densenet.children()):    # Densenet has 1 Sequential child that contains the all the layers
            for b in list(c.children())[6:]:
                for p in b.parameters():
                    p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        
    def get_decoder_type(self):
        return "lstm"

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings.to(torch.float))

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word predicion scores and alphas
        # Cannot use register_buffer here because model.forward is not called yet when model.to(device) is called.
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.init_h.weight.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.init_h.weight.device)
        
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

# See TPGN paper for notation 
class TpgnDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048):
        super(TpgnDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim   = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size  = vocab_size

        self.embedding   = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.lstm_cell_s = LstmCellS(embed_dim, decoder_dim, decoder_dim)
        self.lstm_cell_u = LstmCellU(embed_dim, decoder_dim, decoder_dim)
        self.init_h_s    = nn.Linear(encoder_dim, decoder_dim * decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_s    = nn.Linear(encoder_dim, decoder_dim * decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.tanh        = nn.Tanh()
        self.unbind      = nn.Linear(decoder_dim, decoder_dim ** 2)
        self.fc          = nn.Linear(decoder_dim ** 2, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def get_decoder_type(self):
        return "tpgn"

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.init_h_s.weight.data.uniform_(-0.1, 0.1)
        self.init_h_s.bias.data.fill_(0)
        self.init_c_s.weight.data.uniform_(-0.1, 0.1)
        self.init_c_s.bias.data.fill_(0)
        self.unbind.weight.data.uniform_(-0.1, 0.1)
        self.unbind.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings.to(torch.float))

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        batch_size               = encoder_out.size(0)
        mean_encoder_out         = encoder_out.mean(dim=1)
        average_mean_encoder_out = mean_encoder_out.mean(dim=1)
        reshaped_h_s = self.init_h_s(mean_encoder_out - average_mean_encoder_out.unsqueeze(1))    # (batch_size, decoder_dim * decoder_dim)
        h_s          = reshaped_h_s.view(batch_size, self.decoder_dim, -1)                        # (batch_size, decoder_dim, decoder_dim)
        reshaped_c_s = self.init_c_s(mean_encoder_out - average_mean_encoder_out.unsqueeze(1))    # (batch_size, decoder_dim * decoder_dim)
        c_s          = reshaped_c_s.view(batch_size, self.decoder_dim, -1)                        # (batch_size, decoder_dim * decoder_dim)
        h_u          = torch.zeros(batch_size, self.decoder_dim).to(self.init_h_s.weight.device)
        c_u          = torch.zeros(batch_size, self.decoder_dim).to(self.init_h_s.weight.device)

        return h_s, c_s, h_u, c_u

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        
        batch_size  = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size  = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels  = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LstmCellS and LstmCellU state
        h_s, c_s, h_u, c_u = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word predicion scores
        # Cannot use register_buffer here because model.forward is not called yet when model.to(device) is called.
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.init_h_s.weight.device)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h_s, c_s = self.lstm_cell_s(embeddings[:batch_size_t, t, :], h_s[:batch_size_t], h_u[:batch_size_t], c_s[:batch_size_t])
            h_u, c_u = self.lstm_cell_u(embeddings[:batch_size_t, t, :], h_u[:batch_size_t], h_s[:batch_size_t], c_u[:batch_size_t])
        
            encoded_sentence = torch.zeros(batch_size_t, self.decoder_dim ** 2, self.decoder_dim ** 2).to(self.init_h_s.weight.device)    # (batch, hidden_d ** 2, hidden_d **2)
            for d in range(self.decoder_dim):
                encoded_sentence[:, d * self.decoder_dim:d * self.decoder_dim + h_s.shape[1], d * self.decoder_dim:d * self.decoder_dim + h_s.shape[2]] += h_s
        
            unbinding_vector = self.tanh(self.unbind(h_u))    # (batch, hidden_d **2)
                        
            filler_vector = torch.matmul(encoded_sentence, unbinding_vector.unsqueeze(2)).squeeze(2)    # (batch, hidden_d ** 2)
            
            # Not using softmax because CrossEntropyLoss is expecting unnormalized scores
            preds = self.fc(filler_vector)    # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        
        return predictions, encoded_captions, decode_lengths, None, sort_ind