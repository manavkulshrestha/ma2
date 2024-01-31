from torch_geometric.nn import GATv2Conv
from torch.nn import Module, LogSoftmax, LeakyReLU, BCEWithLogitsLoss, Linear
import torch


class ActionNet(Module):
    def __init__(self, heads=32, concat=False):
        super().__init__()
        self.conv1 = GATv2Conv(4, 256, heads=heads, concat=concat)
        # self.conv2 = GATv2Conv(128, 256, heads=heads, concat=concat)
        self.conv2 = GATv2Conv(256, 1024, heads=heads, concat=concat)
        # self.conv4 = GATv2Conv(512, 1024, heads=heads, concat=concat)
        self.conv3 = GATv2Conv(1024, 256, heads=heads, concat=concat)
        # self.conv6 = GATv2Conv(512, 256, heads=heads, concat=concat)
        # self.conv7 = GATv2Conv(256, 128, heads=heads, concat=concat)
        self.conv4 = GATv2Conv(256, 128, heads=heads, concat=concat)
        self.conv5 = GATv2Conv(128, 2, heads=heads, concat=concat)

        self.logsoftmax = LogSoftmax(1) #TODO CHECK if 1 is correct

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)
        # x = self.conv6(x, edge_index).relu()
        # x = self.conv7(x, edge_index).relu()
        # x = self.conv8(x, edge_index).sigmoid()
        x = self.logsoftmax(x)

        return x
    

class ANet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ANEncoder(**kwargs)
        self.decoder = ANDecoder(**kwargs)
        self.loss = BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z)

        return out
    

class ANEncoder(torch.nn.Module):
        def __init__(self, heads=32, concat=False):
            super().__init__()
            self.conv1 = GATv2Conv(4, 128, heads=heads, concat=concat)
            self.conv2 = GATv2Conv(128, 512, heads=heads, concat=concat)
            self.conv3 = GATv2Conv(512, 1028, heads=heads, concat=concat)

            self.activation = LeakyReLU()

        def forward(self, x, edge_index):
            x = self.layer1(x, edge_index)
            x = self.activation(x)
            x = self.layer2(x, edge_index)
            x = self.activation(x)
            x = self.layer3(x, edge_index)

            return x
    

class ANDecoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear1 = Linear(1028, 512)
        self.linear2 = Linear(512, 128)
        self.linear2 = Linear(128, 2)
        
        self.activation = LeakyReLU()

    def forward(self, z):
        z = self.linear1(z)
        z = self.activation(z)
        z = self.linear2(z)
        z = self.activation(z)
        z = self.linear3(z)

        return z.view(-1)

# x = F.dropout(x, p=0.5, training=self.training)