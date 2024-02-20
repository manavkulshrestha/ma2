import math
from torch_geometric.nn import GATv2Conv
from torch.nn import Module, LogSoftmax, LeakyReLU, BCEWithLogitsLoss, Linear
import torch
import torch_scatter


# class ActionNet(Module):
#     def __init__(self, heads=32, concat=False):
#         super().__init__()
#         self.conv1 = GATv2Conv(4, 256, heads=heads, concat=concat)
#         # self.conv2 = GATv2Conv(128, 256, heads=heads, concat=concat)
#         self.conv2 = GATv2Conv(256, 1024, heads=heads, concat=concat)
#         # self.conv4 = GATv2Conv(512, 1024, heads=heads, concat=concat)
#         self.conv3 = GATv2Conv(1024, 256, heads=heads, concat=concat)
#         # self.conv6 = GATv2Conv(512, 256, heads=heads, concat=concat)
#         # self.conv7 = GATv2Conv(256, 128, heads=heads, concat=concat)
#         self.conv4 = GATv2Conv(256, 128, heads=heads, concat=concat)
#         self.conv5 = GATv2Conv(128, 2, heads=heads, concat=concat)

#         # self.logsoftmax = LogSoftmax(1) #TODO CHECK if 1 is correct

#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv1(x, edge_index, edge_attr).relu()
#         x = self.conv2(x, edge_index, edge_attr).relu()
#         x = self.conv3(x, edge_index, edge_attr).relu()
#         x = self.conv4(x, edge_index, edge_attr).relu()
#         x = self.conv5(x, edge_index, edge_attr)
#         # x = self.conv6(x, edge_index).relu()
#         # x = self.conv7(x, edge_index).relu()
#         # x = self.conv8(x, edge_index).sigmoid()
#         # x = self.logsoftmax(x)

#         return x


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class IntConv(pyg.nn.MessagePassing): # SchInteractionNetwork class
    """Interaction Network as proposed in this paper: 
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html
    SchNet as proposed in this paper:
    https://arxiv.org/abs/1706.08566"""
    
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature, node_dist):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature, node_dist=node_dist)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature, node_dist):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, node_dist, dim_size=None):
        out = torch_scatter.scatter(torch.mul(inputs, node_dist), index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


gnn_type = "SchInteractionNetwork" # choose from {"SchInteractionNetwork", "GatNetwork", "EGatNetwork"} as defined in the previous section

class LearnedSimulator(torch.nn.Module):

    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10,                                                           # number of GNN layers
        num_particle_types=9,
        particle_type_dim=16,                                                     # embedding dimension of particle types
        dim=2,                                                                    # dimension of the world, typical 2D or 3D
        window_size=5,                                                            # the model looks into W frames before the frame to be predicted
        heads = 3                                                                 # number of attention heads in GAT and EGAT
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in = MLP(particle_type_dim + dim * (window_size + 2), hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        if gnn_type == "SchInteractionNetwork":
          self.layers = torch.nn.ModuleList([IntConv(
              hidden_size, 3
          ) for _ in range(n_mp_layers)])
        # elif gnn_type == "GatNetwork":
        #   self.layers = torch.nn.ModuleList([GatNetwork(
        #       hidden_size, hidden_size, heads
        #   ) for _ in range(n_mp_layers)])
        # elif gnn_type == "EGatNetwork":
        #   self.layers = torch.nn.ModuleList([EGatNetwork(
        #       hidden_size, hidden_size, hidden_size, heads, 3, get_attn = False
        #   ) for _ in range(n_mp_layers-1)])
        #   self.layers.append(EGatNetwork(
        #       hidden_size, hidden_size, hidden_size, heads, 3, get_attn = False, use_F = False))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        # pre-processing
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            if gnn_type == "SchInteractionNetwork":
                node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)
            # elif gnn_type == "GatNetwork":
            #     node_feature = self.layers[i](node_feature, data.edge_index)
            # elif gnn_type == "EGatNetwork":
            #     node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        # post-processing
        out = self.node_out(node_feature)
        return out


class ANet(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, **kwargs):
        super().__init__()
        self.encoder = ANEncoder(node_dim, edge_dim, **kwargs)
        self.decoder = ANDecoder(**kwargs)
        self.loss = BCEWithLogitsLoss()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)
        out = self.decoder(z)

        return out
    

class ANEncoder(torch.nn.Module):
        def __init__(self, node_dim, edge_dim, heads=32, concat=False):
            super().__init__()
            kwargs = {'heads': heads, 'concat': concat, 'edge_dim': edge_dim}
            self.conv1 = GATv2Conv(node_dim, 128, **kwargs)
            self.conv2 = GATv2Conv(128, 512, **kwargs)
            self.conv3 = GATv2Conv(512, 1028, **kwargs)

            self.activation = LeakyReLU()

        def forward(self, x, edge_index, edge_attr):
            x = self.conv1(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.conv2(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.conv3(x, edge_index, edge_attr)

            return x
    

class ANDecoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear1 = Linear(1028, 512)
        self.linear2 = Linear(512, 128)
        self.linear3 = Linear(128, 2)
        
        self.activation = LeakyReLU()

    def forward(self, z):
        z = self.linear1(z)
        z = self.activation(z)
        z = self.linear2(z)
        z = self.activation(z)
        z = self.linear3(z)

        return z.view(-1)

# x = F.dropout(x, p=0.5, training=self.training)