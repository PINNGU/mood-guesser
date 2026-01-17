import torch
import torch.nn as nn
import torch.nn.functional as F


class MoodClassifier(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=5):
        super(MoodClassifier, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # fc1 -> bn1 -> relu -> dropout1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # fc2 -> bn2 -> relu -> dropout2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # fc3 (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def get_state_dict(self):
        return self.state_dict()
    
    def load_state_dict_from_dict(self, state_dict):
        self.load_state_dict(state_dict)


def create_model(input_dim=7, output_dim=5):
    return MoodClassifier(input_dim=input_dim, output_dim=output_dim)
