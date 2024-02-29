import math
import torch_scatter
import torch
import torch_geometric as pyg
import torch.nn.functional as F
import os
import numpy as np
import json
from neuralop.models import FNO
from torch import nn
from neuraloperator.neuralop.layers.mlp import MLP as NeuralOpMLP
from neuraloperator.neuralop.layers.embeddings import PositionalEmbedding
from neuraloperator.neuralop.layers.integral_transform import IntegralTransform
from neuraloperator.neuralop.layers.neighbor_search import NeighborSearch
import random
from random import randint
np.random.seed(42)
random.seed(42)
def generate_noise(position_seq, noise_std):
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise


def preprocess(particle_type, position_seq, target_position, metadata, noise_std):
    """Preprocess a trajectory and construct the graph"""
    # apply noise to the trajectory
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # calculate the velocities of particles
    recent_position = position_seq[:, -1]
    
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    
    
    # construct the graph based on the distances between particles
    n_particle = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=True, max_num_neighbors=n_particle)
    
    # node-level features: velocity, distance to the boundary
    normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0)

    # edge-level features: displacement, distance
    dim = recent_position.size(-1)
    edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
    edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # ground truth for training
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
    else:
        acceleration = None

    # return the graph with features

    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        node_dist=edge_distance,
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1),
        recent_pos = recent_position
    )
    return graph


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data_path=None, split='valid', window_length=7, noise_std=0.0, return_pos=False, random_sampling = False):
        super().__init__()

        # load dataset from the disk
        #self.data_path = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
        self.data_path = data_path
        with open(os.path.join('/home/hviswan/Documents/new_dataset/WaterDropSmall/', "metadata.json")) as f:
            self.metadata = json.load(f)
        #with open(os.path.join(data_path, f"{split}_offset.json")) as f:
        #    self.offset = json.load(f)
        #self.offset = {int(k): v for k, v in self.offset.items()}
        self.window_length = window_length
        self.noise_std = noise_std
        self.return_pos = return_pos

        dataset = torch.load(data_path)
        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']
        self.random_sampling = random_sampling
        #self.particle_type = np.memmap(os.path.join(data_path, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        #self.position = np.memmap(os.path.join(data_path, f"{split}_position.dat"), dtype=np.float32, mode="r")
        self.dim = self.position[0].shape[2]
        #for traj in self.offset.values():
        #    self.dim = traj["position"]["shape"][2]
        #    break

        # cut particle trajectories according to time slices
        #self.windows = []
        #for traj in self.offset.values():
        #    size = traj["position"]["shape"][1]
        #    length = traj["position"]["shape"][0] - window_length + 1
        #    for i in range(length):
        #        desc = {
        #            "size": size,
        #            "type": traj["particle_type"]["offset"],
        #            "pos": traj["position"]["offset"] + i * size * self.dim,
        #        }
        #        self.windows.append(desc)

    def len(self):
        return len(self.position)

    def get(self, idx):
        # load corresponding data for this time slice
        #window = self.windows[idx]
        #size = window["size"]

        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        target_position = torch.from_numpy(self.outputs[idx])

        #particle_type = self.particle_type[window["type"]: window["type"] + size].copy()
        #particle_type = torch.from_numpy(particle_type)
        #position_seq = self.position[window["pos"]: window["pos"] + self.window_length * size * self.dim].copy()
        #position_seq.resize(self.window_length, size, self.dim)
        #position_seq = position_seq.transpose(1, 0, 2)
        #target_position = position_seq[:, -1]
        #position_seq = position_seq[:, :-1]
        #target_position = torch.from_numpy(target_position)
        #position_seq = torch.from_numpy(position_seq)
        #print("Position_seq shape = ", position_seq.shape, " Target pos shape = ", target_position.shape, " particle type shape = ", particle_type.shape)

        if(self.random_sampling == True):
            
            mesh_size =  np.random.randint(int(0.25*particle_type.shape[0]), int(0.35*particle_type.shape[0]))
            while(mesh_size %10 !=0):
                mesh_size += 1
            #mesh_size=50
            #points = sorted(list(random.sample(range(0, particle_type.shape[0]), mesh_size)))
            points = sorted(list(range(0, particle_type.shape[0], 3)))
            particle_type = particle_type[points]
            position_seq = position_seq[points]
            target_position = target_position[points]
            #print("Position_seq shape = ", position_seq.shape, " Target pos shape = ", target_position.shape, " particle type shape = ", particle_type.shape)
            #points = list(range(input_x.shape[1]))
        
        # construct the graph
        with torch.no_grad():
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph


class RolloutDataset(pyg.data.Dataset):
    def __init__(self, data_path, split, window_length=7, random_sample=False):
        super().__init__()
        
        # load data from the disk
        self.data_path = data_path
        with open(os.path.join('/home/hviswan/Documents/new_dataset/WaterDropSmall/', "metadata.json")) as f:
            self.metadata = json.load(f)
        #with open(os.path.join(data_path, f"{split}_offset.json")) as f:
        #    self.offset = json.load(f)
        #self.offset = {int(k): v for k, v in self.offset.items()}
        self.window_length = window_length
        self.random_sample = random_sample
        #self.particle_type = np.memmap(os.path.join(data_path, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        #self.position = np.memmap(os.path.join(data_path, f"{split}_position.dat"), dtype=np.float32, mode="r")
        dataset = torch.load(data_path)
        self.particle_type = dataset['particle_type']
        self.position = dataset['position']
        self.n_particles_per_example = dataset['n_particles_per_example']
        self.outputs = dataset['output']

        if(self.random_sample == True):
            mesh_size =  np.random.randint(int(0.25*250), int(0.35*250))
            mesh_size = 140
            while(mesh_size %10 !=0):
                mesh_size += 1
            
            points = sorted(random.sample(range(0, 250), mesh_size))
            self.points = points
        #for traj in self.offset.values():
        #    self.dim = traj["position"]["shape"][2]
        #    break
        self.dim = self.position[0].shape[2]
    def len(self):
        return len(self.position)
    
    def get(self, idx):
        #traj = self.offset[idx]
        #size = traj["position"]["shape"][1]
        #time_step = traj["position"]["shape"][0]
        #particle_type = self.particle_type[traj["particle_type"]["offset"]: traj["particle_type"]["offset"] + size].copy()
        #particle_type = torch.from_numpy(particle_type)
        #position = self.position[traj["position"]["offset"]: traj["position"]["offset"] + time_step * size * self.dim].copy()
        #position.resize(traj["position"]["shape"])
        #position = torch.from_numpy(position)

        particle_type = torch.from_numpy(self.particle_type[idx])
        position_seq = torch.from_numpy(self.position[idx])
        position_seq = torch.permute(position_seq, dims=(1,0,2))
        
        target_position = torch.from_numpy(self.outputs[idx])
        if(self.random_sample):
            
            
            particle_type = particle_type[self.points]
            position_seq = position_seq.permute(1,0,2)
            position_seq = position_seq[self.points]
            position_seq = position_seq.permute(1,0,2)
            target_position = target_position[self.points]
        data = {"particle_type": particle_type, "position": position_seq}
        return data

import matplotlib.pyplot as plt
import networkx as nx
dataset_sample = OneStepDataset(data_path='/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDrop_train.pt', return_pos=True, random_sampling=True)
graph, position = dataset_sample[0]
print(f"The first item in the valid set is a graph: {graph}")
print(f"This graph has {graph.num_nodes} nodes and {graph.num_edges} edges.")
print(f"Each node is a particle and each edge is the interaction between two particles.")
print(f"Each node has {graph.num_node_features} categorial feature (Data.x), which represents the type of the node.")
print(f"Each node has a {graph.pos.size(1)}-dim feature vector (Data.pos), which represents the positions and velocities of the particle (node) in several frames.")
print(f"Each edge has a {graph.num_edge_features}-dim feature vector (Data.edge_attr), which represents the relative distance and displacement between particles.")
print(f"The model is expected to predict a {graph.y.size(1)}-dim vector for each node (Data.y), which represents the acceleration of the particle.")

# remove directions of edges, because it is a symmetric directed graph.
nx_graph = pyg.utils.to_networkx(graph).to_undirected()
# remove self loops, because every node has a self loop.
nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
plt.figure(figsize=(7, 7))
nx.draw(nx_graph, pos={i: tuple(v) for i, v in enumerate(position)}, node_size=50)
plt.savefig('graph.png')


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
    
class SchInteractionNetwork(pyg.nn.MessagePassing): # SchInteractionNetwork class
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

import torch_geometric.utils as pyg_utils

class GatNetwork(pyg.nn.MessagePassing): # GAT Class

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GatNetwork, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(self.in_channels, self.heads * self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = torch.nn.Parameter(torch.randn(self.heads, self.out_channels))
        self.att_r = torch.nn.Parameter(torch.randn(self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l,x_r), alpha=(alpha_l,alpha_r), size=size).reshape(-1, H*C)
        out = out.reshape(-1, H, C).mean(dim=1)
        return out
    
    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr is not None:
          alpha = pyg.utils.softmax(alpha, ptr)
        else:
          alpha = pyg.utils.softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout)
        out = x_j * alpha
        return out

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out

from torch_sparse import SparseTensor, matmul

class EGatNetwork(pyg.nn.MessagePassing):  # EGAT class

    def __init__(self, in_node_channels, in_edge_channels, out_channels, heads = 3, layers = 3, bias = True, get_attn = False, use_F = True,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(EGatNetwork, self).__init__(node_dim=0, **kwargs)

        self.in_node_channels = in_node_channels
        self.in_edge_channels = in_edge_channels
        self.out_channels = out_channels
        self.heads = heads
        self.get_attn = get_attn
        self.use_F = use_F
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.node_out = None
        self.edge_out = None
        self.attn_weights = None
        
        # linear transformation layers for node and edge features
        # self.lin_node = torch.nn.Linear(self.in_node_channels, self.out_channels, bias=True)
        # self.lin_edge = torch.nn.Linear(self.in_edge_channels, self.out_channels, bias=True)
        self.lin_node_i = torch.nn.Linear(self.in_node_channels, self.heads * self.out_channels, bias=False)
        self.lin_node_j = torch.nn.Linear(self.in_node_channels, self.heads * self.out_channels, bias=False)
        self.lin_edge_ij = torch.nn.Linear(self.in_edge_channels, self.heads * self.out_channels, bias=False)

        # attention MLP to multiply with transformed node and edge features 
        self.attn_A = MLP(3*self.heads*self.out_channels, self.heads*self.out_channels, self.heads*self.out_channels, layers)
        #self.attn_A = torch.nn.Linear(3*self.heads*self.out_channels, self.heads*self.out_channels)

        # attention layer to multiply with new edge feature to get unnormalized attention weights
        self.attn_F =  torch.nn.Parameter(torch.FloatTensor(size=(1, self.heads, self.out_channels)))

        # MLPS for compressing multi-head node and edge features
        self.node_mlp = MLP(self.heads * self.out_channels, self.out_channels, self.out_channels, layers)
        self.edge_mlp = MLP(self.heads * self.out_channels, self.out_channels, self.out_channels, layers)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.lin_node.weight)
        # nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_node_i.weight)
        nn.init.xavier_uniform_(self.lin_node_j.weight)
        nn.init.xavier_uniform_(self.lin_edge_ij.weight)
        nn.init.xavier_uniform_(self.attn_F)

    def forward(self, h, edge_index, edge_feature, size=None):
        H, C = self.heads, self.out_channels
        h_prime_i = self.lin_node_i(h)                                           # shape [N,H*C]
        h_prime_j = self.lin_node_j(h)                                           # shape [N,H*C]
        f_ij = self.lin_edge_ij(edge_feature)                                    # shape [E,H*C]
        node_out = self.propagate(edge_index, x=(h_prime_i,h_prime_j), size=size, f_ij=f_ij)   # new multi-head node features
        self.node_out = self.node_mlp(node_out.reshape(-1,H*C))
        self.edge_out = self.edge_mlp(self.edge_out.reshape(-1,H*C)) #self.lin_edge(edge_feature) + self.edge_mlp(self.edge_out.reshape(-1,H*C))
        #self.node_out = self.lin_node(h) + self.node_out
        if self.get_attn:
          return self.node_out, self.edge_out, self.attn_weights
        else:
          return self.node_out, self.edge_out
    
    def message(self, x_i, x_j, index, ptr, size_i, f_ij):
        f_prime_ij = torch.cat([x_i, f_ij, x_j], dim=-1)                         # shape [E,H*C]
        f_prime_ij = self.attn_A(f_prime_ij)
        f_prime_ij = F.leaky_relu(f_prime_ij, negative_slope=self.negative_slope).reshape(-1, self.heads, self.out_channels) 
        self.edge_out = f_prime_ij                                               # new multi-head edge features
        eps = (f_prime_ij * self.attn_F) if self.use_F else f_prime_ij
        eps = eps.sum(dim=-1).unsqueeze(-1)                                      # unnormalized attention weights
        alpha = pyg_utils.softmax(eps, index, ptr, size_i)                       # normalized attention weights
        alpha = F.dropout(alpha, p=self.dropout) 
        self.attn_weights = alpha                                                # shape [E,H,1]
        out = x_j.reshape(-1,self.heads,self.out_channels) * alpha 
        return out

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out



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
        self.fno_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        if gnn_type == "SchInteractionNetwork":
          self.layers = torch.nn.ModuleList([SchInteractionNetwork(
              hidden_size, 3
          ) for _ in range(n_mp_layers)])
          self.layers2 = torch.nn.ModuleList([SchInteractionNetwork(
              hidden_size, 3
          ) for _ in range(4)])
        elif gnn_type == "GatNetwork":
          self.layers = torch.nn.ModuleList([GatNetwork(
              hidden_size, hidden_size, heads
          ) for _ in range(n_mp_layers)])
        elif gnn_type == "EGatNetwork":
          self.layers = torch.nn.ModuleList([EGatNetwork(
              hidden_size, hidden_size, hidden_size, heads, 3, get_attn = False
          ) for _ in range(n_mp_layers-1)])
          self.layers.append(EGatNetwork(
              hidden_size, hidden_size, hidden_size, heads, 3, get_attn = False, use_F = False))
          
        
        self.fno = FNO(n_modes = (24,24),
        hidden_channels=64,
        in_channels=14,
        out_channels=14,
        lifting_channels=256,
        projection_channels=256,
        n_layers=1, 
        use_mlp=False,
        stabilizer=None,
        non_linearity=F.gelu,
        preactivation=True
        )
        self.fno_mapper = FNO(n_modes = (24,24),
        hidden_channels=64,
        in_channels=2,
        out_channels=hidden_size,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4, 
        use_mlp=True,
        stabilizer='tanh',
        non_linearity=F.gelu,
        preactivation=True
        )
        self.reset_parameters()

        self.gno_radius = 0.0004
        fno_hidden_channels = 3
        self.gno_coord_embed_dim = 2
        self.gno_mlp_hidden_layers = [32, 64]
        kernel_in_dim = 1 * self.gno_coord_embed_dim
        kernel_in_dim += fno_hidden_channels 
        self.gno_mlp_hidden_layers.insert(0, kernel_in_dim)
        self.gno_mlp_hidden_layers.append(fno_hidden_channels)

        projection_channels=256
        self.pos_embed = PositionalEmbedding(self.gno_coord_embed_dim)
        self.nb_search_out = NeighborSearch(use_open3d=False)
        self.gno = IntegralTransform(
                    mlp_layers=self.gno_mlp_hidden_layers,
                    mlp_non_linearity=F.gelu,
                    transform_type='nonlinear_kernelonly' 
        )
        self.projection = NeuralOpMLP(in_channels=fno_hidden_channels, 
                        out_channels=128, 
                        hidden_channels=projection_channels, 
                        n_layers=1, 
                        n_dim=1, 
                        non_linearity=F.gelu)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        # pre-processing

        #fno_input = torch.unsqueeze(data.pos, dim=0)
        #fno_input = torch.permute(fno_input, dims=(2, 1, 0))
        #fno_input = torch.unsqueeze(fno_input, dim=0)
        #pos_embedding = self.fno(fno_input)

        #pos_embedding = torch.squeeze(pos_embedding, dim=0)
        #pos_embedding = torch.permute(pos_embedding, dims=(2, 1, 0))
        #pos_embedding = torch.squeeze(pos_embedding, dim=0)

        #neighbor_map = self.nb_search_out(data.recent_pos, data.recent_pos, self.gno_radius)
        
        #print("POS EMBEDDING SHAPE = ", pos_embedding.shape)
        #print("Graph Latent X SHape = ", graph_latent.shape)
        
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            if gnn_type == "SchInteractionNetwork":
                node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)
            elif gnn_type == "GatNetwork":
                node_feature = self.layers[i](node_feature, data.edge_index)
            elif gnn_type == "EGatNetwork":
                node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        # post-processing
        out = self.node_out(node_feature)
        #print("OUT SHAPE = ", out.shape)
        fno_input = torch.unsqueeze(out, dim=0)
        fno_input = torch.permute(fno_input, dims=(2, 1, 0))
        fno_input = torch.unsqueeze(fno_input, dim=0)
        acc_embedding = self.fno_mapper(fno_input)
        acc_embedding = torch.squeeze(acc_embedding, dim=0)
        acc_embedding = torch.permute(acc_embedding, dims=(2, 1, 0))
        node_feature = torch.squeeze(acc_embedding, dim=0)
        #neighbor_map = self.nb_search_out(data.recent_pos, data.recent_pos, self.gno_radius)
        #print("POS EMBEDDING = ", pos_embedding.shape)
        #print("RECENT POS = ", data.recent_pos.shape)
        #print("OUT = ", out.shape)
        
        for i in range(4):
            if gnn_type == "SchInteractionNetwork":
                node_feature, edge_feature = self.layers2[i](node_feature, data.edge_index, edge_feature=edge_feature, node_dist=data.node_dist)

        out = self.node_out(node_feature)
        #graph_latent = self.gno(y = data.recent_pos, neighbors=neighbor_map,  f_y=out)
        #graph_latent = graph_latent.unsqueeze(0).permute(0,2,1)
        #out = self.projection(graph_latent).squeeze(0).permute(1, 0)
        #out = self.node_out(out)
        #graph_latent = self.gno(y = pos_embedding, neighbors=neighbor_map,  f_y=out)
        #graph_latent = graph_latent.unsqueeze(0).permute(0,2,1)
        #out = self.projection(graph_latent).squeeze(0).permute(1, 0)

        #print("data.x shape = ", data.x.shape)
        #print("Data recentPOS shape = ", data.recent_pos.shape)
        #print("OUT SHAPE = ", out.shape)
        #exit()
        return out

params = {
    "epoch": 1000,
    "batch_size": 4,
    "lr": 1e-4,
    "noise": 3e-4,
    "save_interval": 1000,
    "eval_interval": 10000,
    "rollout_interval": 5000,
}

def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(0)
    #print("Total Time = ", total_time)
    
    traj = data["position"][:window_size]
    #print("TRAJ SHAPE = ", traj.shape)
    traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]


    for time in range(total_time - window_size):
        with torch.no_grad():
            graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)
    return traj


def oneStepMSE(simulator, dataloader, metadata, noise):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2).cuda()
        for data in dataloader:
            data = data.cuda()
            pred = simulator(data)
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count


def rolloutMSE(simulator, dataset, noise):
    total_loss = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        for rollout_data in dataset:
            #points = sorted(list(range(0, rollout_data['particle_type'].shape[0], 3)))
            #rollout_data['particle_type'] = rollout_data['particle_type'][points]
            #print("ROLLOUT POSITION = ", rollout_data['position'].shape)
            #pos = rollout_data['position'].permute(1, 0, 2)
            #pos = pos[points].permute(1, 0, 2)
            #rollout_data['position'] = pos
            
            rollout_out = rollout(simulator, rollout_data, dataset.metadata, noise)
            rollout_out = rollout_out.permute(1, 0, 2)
            loss = (rollout_out - rollout_data["position"]) ** 2
            loss = loss.sum(dim=-1).mean()
            print("ROLLOUT LOSS = ", loss.item())
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


from tqdm import tqdm

def train(params, optimizer, scheduler, ckpt, simulator, train_loader, valid_loader=None, valid_rollout_dataset=None):
    loss_fn = torch.nn.MSELoss()
    

    # recording loss curve
    train_loss_list = []
    loss_list = []
    eval_loss_list = []
    eval_mse_list = []
    rollout_mses = []
    onestep_mse_list = []
    rollout_mse_list = []
    x_axis = []
    total_step = 0
    highest_loss = 100000
    for i in range(ckpt, params["epoch"]):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1
            train_loss_list.append((total_step, loss.item()))
            if(batch_count%100 == 0):
                loss_list.append(total_loss/batch_count)
                #print("DATA Y SGAOE = ", len(train_dataset))
                x_axis.append((i*len(train_dataset)//params["batch_size"])+batch_count)


            # evaluation
            if total_step % params["eval_interval"] == 0:
                simulator.eval()
                
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, valid_dataset.metadata, params["noise"])
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                eval_mse_list.append(onestep_mse)
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            # do rollout on valid set
            if total_step % params["rollout_interval"] == 0:
                simulator.eval()
                #valid_rollout_dataset = train_dataset
                rollout_mse = rolloutMSE(simulator, valid_rollout_dataset, params['noise'])
                rollout_mse_list.append((total_step, rollout_mse))
                rollout_mses.append(rollout_mse)
                tqdm.write(f"\nEval: Rollout MSE: {rollout_mse}")
                simulator.train()

            # save model
            #if total_step % params["save_interval"] == 0:
            if(highest_loss > loss.item()):
                print(f'Loss improved from {highest_loss} to {loss.item()} saving weights!')
                highest_loss = loss.item()
                torch.save(
                   {
                       "model": simulator.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(),
                       "epoch":i
                   },
                   os.path.join('/home/hviswan/Documents/Neural Operator', f"EGATNO_rand_dense.pt")
                )
            if(batch_count%100 == 0):
                #x_axis = list(range(len(loss_list)))
                y_axis = loss_list
                fig = plt.figure()
                plt.plot(x_axis, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Train MSE Loss')
                fig.savefig('EGATNO_dense_train.png')
                plt.close(fig)

                y_axis = eval_mse_list
                xa = list(range(len(y_axis)))

                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Eval MSE Loss')
                fig.savefig('EGATNO_dense_eval.png')
                plt.close(fig)

                y_axis = rollout_mses
                xa = list(range(len(y_axis)))
                fig = plt.figure()
                plt.plot(xa, y_axis)
                plt.xlabel('Epochs')
                plt.ylabel('Average Rollout Loss')
                fig.savefig('EGATNO_dense_rollout.png')
                plt.close(fig)
    return train_loss_list

if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = OneStepDataset('/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDrop_train.pt', "train", noise_std=params["noise"], random_sampling=True)
    valid_dataset = OneStepDataset('/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_test.pt', "train", noise_std=params["noise"], random_sampling=True)
    rollout_dataset = RolloutDataset('/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_rollout.pt', "train", random_sample=True)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False)


    simulator = LearnedSimulator()
    ckpt = torch.load('EGATNO_rand_dense.pt')
    weights = torch.load('EGATNO_rand_dense.pt')['model']

    model_dict = simulator.state_dict()
    ckpt_dict = {}
    
    #print(simulator.keys())
    model_dict = dict(model_dict)

    for k, v in weights.items():
        k2 = k[0:]
        #print(k2)
        if k2 in model_dict:
            #print(k2)
            if model_dict[k2].size() == v.size():
                ckpt_dict[k2] = v
            else:
                print("Size mismatch while loading! %s != %s Skipping %s..."%(str(model_dict[k2].size()), str(v.size()), k2))
                mismatch = True
        else:
            print("Model Dict not in Saved Dict! %s != %s Skipping %s..."%(2, str(v.size()), k2))
            mismatch = True
    if len(simulator.state_dict().keys()) > len(ckpt_dict.keys()):
        mismatch = True
    model_dict.update(ckpt_dict)
    simulator.load_state_dict(model_dict)
    

    #simulator.load_state_dict(weights['model'])
    simulator = simulator.cuda()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    #ckpt = {}
    #ckpt['epoch'] = 0
    train_loss_list = train(params, optimizer, scheduler, ckpt['epoch'], simulator, train_loader, valid_loader, valid_rollout_dataset=rollout_dataset)

