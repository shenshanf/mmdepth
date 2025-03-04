import torch
from torch import nn as nn

from .conv_gru import ConvGRUCell, ContextConvGRUCell
from mmdepth.models.modules.opts import Spatial2DInterpolAs, spatial_2d_maxpool_as


class CrossStackedGRUCell(nn.Module):
    """See RAFT-Stereo Algorithm"""

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

        # make cross connect gru layers
        for idx in range(num_stacked):
            if idx == 0:
                gru_cell = ConvGRUCell(hidden_dims[idx + 1], hidden_dims[idx], kernel_size, bias)
            elif idx == num_stacked - 1:
                gru_cell = ConvGRUCell(hidden_dims[idx - 1] + input_dim, hidden_dims[idx], kernel_size, bias)
            else:
                gru_cell = ConvGRUCell(hidden_dims[idx - 1] + input_dim[idx + 1], hidden_dims[idx], kernel_size, bias)
            self.gru_cells.append(gru_cell)

        self.up_sample = Spatial2DInterpolAs(mode='bilinear', align_corners=False)
        self.max_pool = spatial_2d_maxpool_as

    def _cross_connections(self, idx, hd_states, x, current_state):
        """Helper method to handle cross-scale connections"""
        if idx == 0:  # Coarsest level
            return self.max_pool(hd_states[idx + 1], current_state)
        elif idx == self.num_stacked - 1:  # Finest level
            up_features = self.up_sample(hd_states[idx - 1], current_state)
            return torch.cat((x, up_features), dim=1)
        else:  # Intermediate levels
            down_features = self.max_pool(hd_states[idx + 1], current_state)
            up_features = self.up_sample(hd_states[idx - 1], current_state)
            return torch.cat((down_features, up_features), dim=1)

    def forward(self, hd_states, x):
        """

        Args:
            hd_states: (from coarse to fine order)
            x:

        Returns:

        """
        assert len(hd_states) == len(self.gru_cells)
        next_hd_states = []

        for idx, (hd_state, gru_cell) in enumerate(zip(hd_states, self.gru_cells)):
            input_x = self._cross_connections(idx, hd_states, x, hd_state)
            next_hd_states.append(gru_cell(hd_state, input_x))
        return next_hd_states


class CrossStackedContextGRUCell(nn.Module):
    """See RAFT-Stereo Algorithm"""

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

        # make cross connect gru layers
        for idx in range(num_stacked):
            if idx == 0:
                gru_cell = ContextConvGRUCell(hidden_dims[idx + 1], hidden_dims[idx],
                                              kernel_size, bias)
            elif idx == num_stacked - 1:
                gru_cell = ContextConvGRUCell(hidden_dims[idx - 1] + input_dim, hidden_dims[idx],
                                              kernel_size, bias)
            else:
                gru_cell = ContextConvGRUCell(hidden_dims[idx - 1] + input_dim[idx + 1], hidden_dims[idx],
                                              kernel_size, bias)
            self.gru_cells.append(gru_cell)

        self.up_sample = Spatial2DInterpolAs(mode='bilinear', align_corners=False)
        self.max_pool = spatial_2d_maxpool_as

    def _cross_connections(self, idx, hd_states, x, current_state):
        """Helper method to handle cross-scale connections"""
        if idx == 0:  # Coarsest level
            return self.max_pool(hd_states[idx + 1], current_state)
        elif idx == self.num_stacked - 1:  # Finest level
            up_features = self.up_sample(hd_states[idx - 1], current_state)
            return torch.cat((x, up_features), dim=1)
        else:  # Intermediate levels
            down_features = self.max_pool(hd_states[idx + 1], current_state)
            up_features = self.up_sample(hd_states[idx - 1], current_state)
            return torch.cat((down_features, up_features), dim=1)

    def forward(self, hd_states, x, contexts):
        """

        Args:
            hd_states: (from coarse to fine order)
            x:
            contexts

        Returns:

        """
        assert len(hd_states) == len(self.gru_cells)
        assert len(contexts) == len(self.gru_cells)
        next_hd_states = []

        for idx, (hd_state, context, gru_cell) in enumerate(zip(hd_states, contexts, self.gru_cells)):
            input_x = self._cross_connections(idx, hd_states, x, hd_state)
            next_hd_states.append(gru_cell(hd_state, input_x, context))
        return next_hd_states
