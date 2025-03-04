import torch
import torch.nn as nn


class _ConvGRUCell(nn.Module):
    """
    ConvGRU Cell implementation

    Args:
        input_dim: Number of channels of input tensor
        hidden_dim: Number of channels of hidden state
        kernel_size: Size of the convolutional kernel
        bias:
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, init_hs='zero'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Reset gate parameters
        self.reset_gate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # Update gate parameters
        self.update_gate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # Output gate parameters
        self.out_gate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        assert init_hs in ['zero', 'learnable']
        self.init_hs = init_hs
        if init_hs == 'learnable':
            self._init_hs_para = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        else:
            self._init_hs_para = None


class ConvGRUCell(_ConvGRUCell):

    def forward(self, hs, x):
        """
        Forward pass of the ConvGRU cell

        Args:
            hs: (b, hidden_dim, h, w) Hidden state tensor
            x: (b, c, h, w) Input tensor

        Returns:
            hidden: (b, hidden_dim, h, w) Updated hidden state
        """
        if hs is None:
            B, _, H, W = x.size()
            if self._init_hs_para == 'zero':
                hs = torch.zeros(B, self.hidden_dim, H, W).to(x.device)
            elif self.init_hs == 'learnable':
                assert self._init_hs_para is not None
                hs = self._init_hs_para.expand(B, -1, H, W)

        # Concatenate input and hidden state
        x_hs = torch.cat([x, hs], dim=1)

        # Update gate
        z = torch.sigmoid(self.update_gate(x_hs))

        # Reset gate
        r = torch.sigmoid(self.reset_gate(x_hs))

        # Concatenate input tensor and reset-gated hidden state
        # Current memory content
        q = torch.tanh(self.out_gate(torch.cat([x, r * hs], dim=1)))

        # New hidden state
        next_hs = (1 - z) * hs + z * q

        return next_hs


class ContextConvGRUCell(_ConvGRUCell):

    def forward(self, hs, x, context):
        """"""
        assert context.shape[1] % 3 == 0
        cz, cr, cq = torch.chunk(context, chunks=3, dim=1)

        if hs is None:
            B, _, H, W = x.size()
            hs = torch.zeros(B, self.hidden_dim, H, W).to(x.device)

        # Concatenate input and hidden state
        x_hs = torch.cat([x, hs], dim=1)

        # Update gate
        z = torch.sigmoid(self.update_gate(x_hs) + cz)  # cz

        # Reset gate
        r = torch.sigmoid(self.reset_gate(x_hs) + cr)

        # Concatenate input tensor and reset-gated hidden state
        # Current memory content
        q = torch.tanh(self.out_gate(torch.cat([x, r * hs], dim=1)) + cq)
        # New hidden state
        next_hs = (1 - z) * hs + z * q

        return next_hs


class StackedConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, bias=True,
                 num_stacked=3):
        """

        Args:
            input_dim:
            hidden_dims:
            kernel_size:
            bias:
            num_stacked:
        """
        super().__init__()

        self.gru_cells = nn.ModuleList()
        assert num_stacked == len(hidden_dims)
        self.num_stacked = num_stacked
        for hidden_dim in hidden_dims:
            self.gru_cells.append(ConvGRUCell(input_dim, hidden_dim, kernel_size, bias))

    def forward(self, hd_states, x):
        """
        Forward pass of the Stacked ConvGRU

        Args:
            x: (B, C, H, W) Input tensor
            hd_states: List of hidden states for each scale, or None for initial state

        Returns:
            next_m_hs: List of updated hidden states for each scale
        """
        B, _, H, W = x.size()

        # Initialize hidden states if None
        if hd_states is None:
            hd_states = [None] * self.num_scales
        else:
            assert len(hd_states) == len(self.gru_cells)

        next_hd_states = []

        input_x = x
        for hd_state, gru_cell in zip(hd_states, self.gru_cells):
            next_hd_state = gru_cell(input_x, hd_state)
            next_hd_states.append(next_hd_state)
            input_x = next_hd_state
        return next_hd_states


