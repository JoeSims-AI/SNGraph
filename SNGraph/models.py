from SNGraph.EdgeGAT import Edge_GATConv
import torch.nn as nn
import torch
import torch_geometric.utils as tgu
from torch_scatter import scatter_add


class GraphConvMMP(nn.Module):  # Multiple Message Passing
    def __init__(self,
                 in_features=17,
                 out_features=4,
                 edge_dims=1,
                 n_message_passing=4,
                 hidden_features=64,
                 mlp_features=64):
        """
        :param in_features: The number of input node features. (default :obj:`17`).
        :type in_features: positive int
        :param out_features: The number of output node features/classes. (default :obj:`4`).
        :type out_features: positive int
        :param edge_dims: The number of edge features to include. (default :obj:`1`)
        :type edge_dims: positive int
        :param n_message_passing: The number of message passing steps. (default :obj:`4`).
        :type n_message_passing: positive int
        :param hidden_features: The number of output node features in the intermediate message passing steps.
            (default :obj:`64`).
        :type hidden_features: positive int
        :param mlp_features: The number of hidden features in the MLP within the model. (default :obj:`64`).
        :type mlp_features: positive int
        """
        super(GraphConvMMP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edge_dims = edge_dims
        # If the number of message passing steps is only 1 then the output features should be defined.
        self.hidden_features = hidden_features if n_message_passing > 1 else out_features
        # Store the graph message passing steps in a list.
        self.mp_list = nn.ModuleList()
        # Define the first message passing and append to he graph list.
        self.mp_list.add_module('in', Edge_GATConv(in_features,
                                                   [mlp_features, mlp_features],
                                                   self.hidden_features,
                                                   edge_dim=self.edge_dims,
                                                   heads=1))
        # The first message passing step is
        if n_message_passing > 1:
            for i in range(1, n_message_passing):
                if i + 1 != n_message_passing:
                    self.mp_list.add_module(f'hidden_{i}', Edge_GATConv(self.hidden_features,
                                                                        [mlp_features, mlp_features],
                                                                        self.hidden_features,
                                                                        edge_dim=self.edge_dims,
                                                                        heads=1))
                else:
                    self.mp_list.add_module(f'out', Edge_GATConv(self.hidden_features,
                                                                 [mlp_features, mlp_features],
                                                                 self.out_features,
                                                                 edge_dim=self.edge_dims,
                                                                 heads=1))

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):

        for mp in self.mp_list:
            x = mp(x, edge_index, edge_attr)

        # x = self.sigmoid(x)
        return x


class GraphAttSurv(nn.Module):

    def __init__(self,
                 in_features=17,
                 hidden_features=64,
                 out_graph_features=8,
                 out_features=1,
                 edge_dims=1,
                 n_message_passing=4,
                 mlp_features=64,
                 attention_features=128,
                 ):
        """
        :param in_features: The number of input node features. (default :obj:`17`).
        :type in_features: positive int
        :param hidden_features: The number of output features for the intermediate message passing steps.
           (default :obj:`64`).
       :type hidden_features: positive int
        :param out_graph_features: The number of features that the final message passing layers has.
            (default :obj:`8`).
        :type out_graph_features: positive int
        :param out_features: The number of features that the final model output has. (default :obj:`1`).
        :type out_features: positive int
        :param edge_dims: The number of edge features to include. (default :obj:`1`)
        :type edge_dims: positive int
        :param n_message_passing: The number of message passing steps. (default :obj:`4`).
        :type n_message_passing: positive int
        :param mlp_features: The number of hidden features in the MLP within the model. (default :obj:`64`).
        :type mlp_features: positive int
        :param attention_features: There is hidden layer between the output of the graph and the final output. This
            is part of the attention mechanism. (default :obj:`128`).
        :type attention_features: positive int
        """

        super(GraphAttSurv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out_graph_features = out_graph_features
        # If the number of message passing steps is only 1 then the output features should be defined.
        self.hidden_features = hidden_features if n_message_passing > 1 else out_graph_features
        # Store the graph message passing steps in a list.
        self.mp_list = nn.ModuleList()
        # Define the first message passing and append to the graph list.
        self.mp_list.add_module('in', Edge_GATConv(in_features,
                                                   [mlp_features, mlp_features],
                                                   self.hidden_features,
                                                   edge_dim=edge_dims,
                                                   heads=1))
        # The first message passing step is
        if n_message_passing > 1:
            for i in range(1, n_message_passing):
                if i + 1 != n_message_passing:
                    self.mp_list.add_module(f'hidden_{i}', Edge_GATConv(self.hidden_features,
                                                                        [mlp_features, mlp_features],
                                                                        self.hidden_features,
                                                                        edge_dim=edge_dims,
                                                                        heads=1))
                else:
                    self.mp_list.add_module('out', Edge_GATConv(self.hidden_features,
                                                                [mlp_features, mlp_features],
                                                                self.out_graph_features,
                                                                edge_dim=edge_dims,
                                                                heads=1))
        self.L = out_graph_features
        self.D = attention_features
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, out_features),
            nn.Sigmoid()
        )

    def forward(self, data):

        batch = data.batch
        x = data.x

        for mp in self.mp_list:
            x = mp(x, data.edge_index, data.edge_attr)

        A = self.attention(x)
        A = torch.transpose(A, 1, 0)
        A = tgu.softmax(A.flatten(), batch)
        M = torch.mul(x.permute(1, 0), A)
        M = scatter_add(M.permute(1, 0), batch, dim=0)
        Y_hat = self.classifier(M)
        return Y_hat, A