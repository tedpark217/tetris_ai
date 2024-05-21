import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models

class RLModel(nn.Module):
  def __init__(self, input_size):
    super(RLModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=6)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
    self.maxpool1 = nn.MaxPool2d(kernel_size=(13, 3))
    
    self.conv3 = nn.Conv2d(in_channels=input_size, out_channels=128, kernel_size=4)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
    self.maxpool2 = nn.MaxPool2d(kernel_size=(15, 5))

    self.dense1 = nn.Linear(110, 64)
    self.dense2 = nn.Linear(64, 128)
    self.dense3 = nn.Linear(128, 1)

    self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

  def forward(self, input1, input2):
    a = F.relu(self.conv1(input1))
    a = F.relu(self.conv2(a))
    a = self.maxpool1(a)
    a = a.view(a.size(0),-1)

    b = F.relu(self.conv3(input1))
    b = F.relu(self.conv4(b))
    b = self.maxpool2(b)
    b = b.view(b.size(0),-1)

    x = torch.cat([a, b, input2], dim=1)
    x = F.relu(self.dense1(x))
    x = F.relu(self.dense2(x))
    output = self.dense3(x)

    return output

# TODO
class NewModel(nn.Module):
    def __init__(self, input_size):
        super(NewModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=6)

    def forward(self, input1, input2):
        a = F.relu(self.conv1(input1))
        return a