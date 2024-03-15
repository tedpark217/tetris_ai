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

    print("printing output.shape")
    print(input1.shape[0])
    print(output.shape)

    return output

# TODO
class NewModel(nn.Module):
    
  verbose = False
    
  def __init__(self, input_size):
    super(NewModel, self).__init__()
    
    print("INIT new model")
    print(input_size)
    
    
    self.fc1 = nn.Linear(200, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(174, 128)
    self.fc5 = nn.Linear(128, 64)
    self.fc6 = nn.Linear(64,1)
    
    # self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=6)
    # self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
    # self.maxpool1 = nn.MaxPool2d(kernel_size=(13, 3))
    
    # self.conv3 = nn.Conv2d(in_channels=input_size, out_channels=128, kernel_size=4)
    # self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
    # self.maxpool2 = nn.MaxPool2d(kernel_size=(15, 5))

    # self.dense1 = nn.Linear(110, 64)
    # self.dense2 = nn.Linear(64, 128)
    # self.dense3 = nn.Linear(128, 1)

    self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

  def forward(self, input1, input2):
    # shape0 = input1.shape[0]
    # if(shape0 != 18):
    #     self.verbose = False
      
    # if(self.verbose):
    #     print("forward NEW MODEL")
    #     print(input1.shape)
    #     print(input2.shape)

    
    my_result = F.relu(self.fc1(input1.view(-1, 200)))
    my_result = F.relu(self.fc2(my_result))
    my_result = F.relu(self.fc3(my_result))
    
    my_result = torch.cat((my_result, input2), dim=1)
    
    my_result = F.relu(self.fc4(my_result))
    my_result = F.relu(self.fc5(my_result))
    
    
    output = F.relu(self.fc6(my_result))
    
    # print("printing output.shape")
    # print(input1.shape[0])
    # print(output.shape)
    
    # if(self.verbose):
    #     print("printing my result and input2 shape:")
    #     print(my_result.shape)
    #     print(input2.shape)
    
    
    # if(self.verbose):
    #     print("before")
    #     print(input1.shape)
    # a = F.relu(self.conv1(input1))
    # if(self.verbose):
    #     print("after")
    #     print(a.shape)
    
    
    
    # a = F.relu(self.conv2(a))
    # a = self.maxpool1(a)
    # a = a.view(a.size(0),-1)

    # b = F.relu(self.conv3(input1))
    # b = F.relu(self.conv4(b))
    # b = self.maxpool2(b)
    # b = b.view(b.size(0),-1)

    # x = torch.cat([a, b, input2], dim=1)
    # x = F.relu(self.dense1(x))
    # x = F.relu(self.dense2(x))
    # output = self.dense3(x)
    
    # if(self.verbose):
    # print("printing output shape")
    # print(input1.shape[0])
    # print(output.shape)
    

    return output