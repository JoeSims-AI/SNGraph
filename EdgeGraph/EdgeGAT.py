import torch
import torch.nn.functional as F
from torch.nn import Parameter, Tanh
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def softmax(src, index, num_nodes):

    """
    Given a value tensor: 'src', this function first groups the values along the first dimension
    based on the indices specified ind: 'index', and then proceeds to compute the softmax individually for each group.
    """

    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    return out / (out_sum + 1e-16)


class Edge_GATConv(MessagePassing):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers.
    This is a modified version to account for edge features.

    Args:
        in_channels (int): Size of input node features.
        hidden_channels (list of ints): Size of hidden features in the
            two-layer perceptron.
        out_channels (int): Size of output node features.
        edge_dim (int): Size of edge features.
        heads (int, optional): The number of attention heads. (default: :obj:`1`)
        negative_slope (float, optional): The linear gradient for LeakyRelU
            in the negative axis. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability. If :obj: `0` then no
            dropout is used. (default: :obj:`0.`)
        bias (bool, optional): If set to :obj: `True`, bias is used.
            (default: :obj:`True`)

    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 edge_dim=1,
                 heads=1,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(Edge_GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight1 = Parameter(torch.Tensor(in_channels, heads * hidden_channels[0]))
        self.weight2 = Parameter(torch.Tensor(heads * hidden_channels[0], heads * hidden_channels[1]))
        self.weight3 = Parameter(torch.Tensor(heads * hidden_channels[1], heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        self.edge_update_w = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.weight3)
        glorot(self.att)
        glorot(self.edge_update_w)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        tanh1 = Tanh()
        x = tanh1(torch.mm(x, self.weight1))
        tanh2 = Tanh()
        x = tanh2(torch.mm(x, self.weight2))
        tanh3 = Tanh()
        x = tanh3(torch.mm(x, self.weight3).view(-1, self.heads, self.out_channels))

        if size is None and torch.is_tensor(x):
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(
            edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges],
                              dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr,
                              size=size)

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_update_w)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
