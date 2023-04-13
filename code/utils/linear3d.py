"""Layer that takes as input as 2D vector, with 3D params."""
import math
import torch as th
from torch.nn import Parameter


def functional_linear3d(input, weight, bias=None):
    r"""
    Apply a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = input.transpose(0, 1).matmul(weight)
    if bias is not None:
        output += bias.unsqueeze(1)
    return output.transpose(0, 1)


class Linear3D(th.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(3, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, channels, in_features, out_features, batch_size=-1, bias=True, noise=False):
        super(Linear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        if noise:
            self.in_features += 1
        self.weight = Parameter(th.Tensor(channels, self.in_features, out_features))
        if bias:
            self.bias = Parameter(th.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        if noise:
            self.register_buffer("noise", th.Tensor(batch_size, channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix=None, permutation_matrix=None):

        input_ = [input]

        if input.dim() == 2:
            if permutation_matrix is not None:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, permutation_matrix.shape[1]]))
            elif hasattr(self, "noise"):
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features - 1 ]))
            else:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features]))

        if adj_matrix is not None and permutation_matrix is not None:
            input_.append((input_[-1].transpose(0, 1) @ (adj_matrix.t().unsqueeze(2) * permutation_matrix)).transpose(0, 1))
        elif adj_matrix is not None:
            input_.append(input_[-1] * adj_matrix.t().unsqueeze(0))
        elif permutation_matrix is not None:
            input_.append((input_[-1].transpose(0, 1) @ permutation_matrix).t())

        if hasattr(self, 'noise'):
            self.noise.normal_()
            input_.append(th.cat([input_[-1], self.noise], 2))

        return functional_linear3d(input_[-1], self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def apply_filter(self,permutation_matrix):
        transpose_weight = self.weight.transpose(1, 2) @ permutation_matrix
        self.weight = Parameter(transpose_weight.transpose(1, 2))



