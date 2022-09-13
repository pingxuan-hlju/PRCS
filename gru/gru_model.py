import torch
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRUMulti(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRUMulti, self).__init__()

        # input_size = 320
        self.input_size = input_size

        # self.hidden_sizes = [64, 128, 320]
        # self.kernel_sizes = [3, 5, 3]
        # self.n_layers = 3

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            # 320->64，64->128，128->320
            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x
        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden

class EncodeConvGru(nn.Module):
    def __init__(self, input_size=6, in_dim=320):
        super(EncodeConvGru, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = [8, 16, input_size]
        self.kernel_sizes = [3, 5, 3]
        self.n_layers = 3
        self.gru = ConvGRUMulti(self.input_size, self.hidden_sizes, self.kernel_sizes, self.n_layers)
        self.conv = nn.Conv3d(in_channels=in_dim+in_dim // 2*2, out_channels=in_dim, kernel_size=1)
        self.jw_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.bn1 = nn.InstanceNorm3d(160)
        self.bn2 = nn.InstanceNorm3d(320)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x2 = self.jw_conv(x)
        x2 = self.relu(self.bn1(x2))
        x1 = x2.permute(1, 0, 3, 2, 4)
        num_seqs = x1.shape[0]

        res_forward = []
        res_backward = []
        h_forward_state = []
        h_backward_state = []
        h_forward_next = None
        h_backward_next = None
        for time in range(num_seqs):
            h_forward_next = self.gru(x1[time],h_forward_next)
            h_backward_next = self.gru(x1[num_seqs - time - 1],h_backward_next)
            h_forward_state.append(h_forward_next[self.n_layers-1].unsqueeze(dim=2))
            h_backward_state.append(h_backward_next[self.n_layers - 1].unsqueeze(dim=2))

        # 2D to 3D
        res_forward = reduce(lambda a, b: torch.cat((a, b), dim=2), h_forward_state)
        res_forward = res_forward.permute(0, 2, 3, 1, 4)
        res_backward = reduce(lambda a, b: torch.cat((a, b), dim=2), list(reversed(h_backward_state)))
        res_backward = res_backward.permute(0, 2, 3, 1, 4)
        res1 = torch.cat((x,res_forward,res_backward),1)
        res2 = self.conv(res1)
        res2 = self.relu(self.bn2(res2))
        return res2