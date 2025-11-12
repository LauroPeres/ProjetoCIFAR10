import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    
    def __init__(self, inputSize, hiddenSize, numClasses):
        super().__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hiddenSize, numClasses)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 canais de entrada, 32 de saída, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Camadas de pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 * 4 * 4 = 2048
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 classes no CIFAR10
        
        # Camadas de dropout para regularização
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Camadas convolucionais com ReLU e pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32x3 -> 16x16x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x16x32 -> 8x8x64
        x = self.pool(F.relu(self.conv3(x)))  # 8x8x64 -> 4x4x128
        
        # Achatar para camadas fully connected
        x = x.view(-1, 128 * 4 * 4)
        
        # Camadas fully connected com dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
