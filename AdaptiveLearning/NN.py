import torch
import torch.nn as nn   #Neura Network library


### Defining a dense layer of my model###
# x: input to the layer
class OurDenseLayer(nn.Module):
    def __init__(self, input_nodes, output_nodes):
        super(OurDenseLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_nodes, output_nodes)) #initializing the weigths 
        self.bias = nn.Parameter(torch.randn(output_nodes)) #initializing the weigths 

    def forward(self, x):
        y = torch.sigmoid(torch.matmul(x, self.W) + self.bias)  #y = sigmoid(xW + b)
        return y



### Defining a model using subclassing###
class LinearSubclassing(nn.Module):
    def __init__(self, input_nodes, output_nodes):
        super(LinearSubclassing, self).__init__()
        self.linear = nn.Linear(input_nodes, output_nodes)  #linear layer
        self.activation = nn.Sigmoid()                      #sigmopid activation

    def forward(self, x):
        linear_output = self.linear(x)
        output = self.activation(linear_output)
        return output



### Custom behavior with subclassing nn.Module ###
class LinearButSometimesIdentity(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearButSometimesIdentity, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    ##Implement the behavior where the network outputs the input, unchanged, under control of the isidentity argument.
    def forward(self, inputs, isidentity=False):
        if isidentity:
            return inputs
        else:
            return self.linear(inputs)
    # def forward(self, inputs, isidentity=False):