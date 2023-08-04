import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, out_dim):
        super().__init__()

        self.layer_num = layer_num
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        
        x = inputs
        for _ in range(self.layer_num):
            x = self.hidden(x)
            x = self.relu(x)
        x = torch.cat([inputs, x], axis = 1)

        out = self.out_layer(x)

        return out

class TwoTowerModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.user_encoder = Encoder()
        self.business_encoder = Encoder()

    def forward(self, user_inputs, business_inputs):
        
        x1 = self.user_encoder(user_inputs)
        x2 = self.business_encoder(business_inputs)
        out = torch.dot(x1, x2)

        return out
        