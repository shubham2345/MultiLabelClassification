import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # First hidden layer with 768 neurons
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first hidden layer
        self.fc2 = nn.Linear(512, 256)         # Second hidden layer with 256 neurons
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second hidden layer
        self.fc3 = nn.Linear(256, 128)         # Third hidden layer with 128 neurons
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout layer after the third hidden layer
        self.fc4 = nn.Linear(128, output_dim)          # Output layer for 19 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))            # Activation for the first layer
        x = self.dropout1(x)                    # Apply dropout after the first layer
        x = torch.relu(self.fc2(x))            # Activation for the second layer
        x = self.dropout2(x)                    # Apply dropout after the second layer
        x = torch.relu(self.fc3(x))            # Activation for the third layer
        x = self.dropout3(x)                    # Apply dropout after the third layer
        x = torch.sigmoid(self.fc4(x))         # Sigmoid for multi-label output
        return x