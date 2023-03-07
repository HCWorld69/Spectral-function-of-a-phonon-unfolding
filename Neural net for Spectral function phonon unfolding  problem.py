# Here's my example PyTorch code for training a neural network to predict the spectral function of a phonon unfolding problem:
# this is just a basic example, and please contact me if you  may need to modify the code to suit your specific problem. In particular, you'll need to define your own data loading and preprocessing functions to convert your data into PyTorch tensors.

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network architecture
class PhononUnfoldingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PhononUnfoldingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the hyperparameters
input_size = 100
hidden_size = 50
output_size = 10
learning_rate = 0.01
batch_size = 32
num_epochs = 100

# Load the data and split into training and validation sets
x_train = torch.randn(1000, input_size)
y_train = torch.randn(1000, output_size)
x_valid = torch.randn(100, input_size)
y_valid = torch.randn(100, output_size)
train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize the neural network and optimizer
net = PhononUnfoldingNet(input_size, hidden_size, output_size)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Define the loss function
criterion = nn.MSELoss()

# Train the neural network
for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    net.train()
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)
    valid_loss /= len(valid_dataset)
    print('Epoch: {:02d}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, train_loss, valid_loss))
