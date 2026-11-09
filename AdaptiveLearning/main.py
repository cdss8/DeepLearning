#Introducing ML with PyTorch

import torch  #tensorflow alternative
from NN import OurDenseLayer, LinearSubclassing, LinearButSometimesIdentity
import torch.nn as nn
import matplotlib.pyplot as plt


#Introducing tensors
#x0 dimesion
integer =  torch.tensor(1234)
pi = torch.tensor(3.1415926535)
print(f"`integer`------- is a {integer.dim} with {integer.ndim}-dimension Tensor:{integer}")
print(f"`pi`------------ is a {pi.dim} with {pi.ndim}-dimension Tensor:{pi}")

#x1 dimension
fibonacci = torch.tensor([1,1,2,3,5,8,13,21])
counter = torch.tensor(range(200))
print(f"\n`fibinacci`----- is a {fibonacci.ndim}-d Tensor with shape:{fibonacci.shape}")
print(f"`counter`------------ is a {counter.ndim}-d Tensor with shape:{counter.shape}")

#2x dimension and more 
matrix = torch.tensor([[3,4,5,6],[1,3,2,8]])
assert isinstance(matrix, torch.Tensor), "matrix must be a torch Tensor Object"
assert matrix.ndim == 2
print(f"\n`matrix`----- is a {matrix.ndim}-d Tensor with shape:{matrix.shape}")
print(matrix.type)

#Above 2x dimensiion
# #For image processing and computer vission, we often use 4 dimenssions: batch size, range of colors, image weigth and image height
image = torch.zeros((10, 3, 256, 256))  #matrix of zeros
assert isinstance(image, torch.Tensor)
assert image.ndim == 4
assert image.shape == (10, 3, 256, 256)
print(f"\n`Image of zeros with`-----has a {image.ndim}-d  and Tensor with shape:{image.shape}")

#simple operations
row_vector1 = matrix[0]
row_vector2 = image[1] #it should print a 4 dimensional matrix of zeros in 3 vectors, if I not mistaken :)) 
column_vector = matrix[:, 0] # prints the 0-i position of each vector of the matrix
scalar = matrix[0, 1] # prints a concrete number of the matrix

print(f"`column_vector`: {column_vector}")
print(f"`row_vector1` ----- printing the 0-i vector of the matrix: {row_vector1}")
print(f"`row_vector2` ----- printing the 1-i vector of the image : {row_vector2}")
print(f"`scalar`----------- {scalar}")


#nodes in the graph
a = torch.tensor(1.8)
b = torch.tensor(7)

sum = torch.add(a, b)
sum2 = a + b  #checking if + operand works on tensors
print(f"`sum`  ------------- {sum}")
print(f"`sum2` ------------- {sum2}")

#nodes of multiple operation, where a nd b are independent variables
#Tree operationtopology
def func(a, b):
    c = torch.add(a, b) 
    d = torch.subtract(b, 1) 
    e = torch.multiply(c, d)  
    return e

e = func(a,b)
print(f"`function e` -------- {e}")



# Defining layers of our NN.py and testing the output!
num_inputs = 2
num_outputs = 4
layer = OurDenseLayer(num_inputs, num_outputs)
x_input = torch.tensor([[1, 2.1]])
y1 = layer(x_input)
print(f"\nLinear testing")
print(f"`input shape`-------- {x_input.shape}")
print(f"`output shape` ------ {y1.shape}")
print(f"`output result`------- {y1}")

#Sequencial testing
model = nn.Sequential(nn.Linear(num_inputs, num_outputs),nn.Sigmoid())
y2 = model(x_input)
print(f"\nSequencial testing")
print(f"`input shape`-------- {x_input.shape}")
print(f"`output shape` ------ {y2.shape}")
print(f"`output result`------- {y2}")

#Subclasing testing
model2 = LinearSubclassing(num_inputs, num_outputs)
y3 = model2(x_input)
print(f"\nSubclasing testing")
print(f"`input shape`-------- {x_input.shape}")
print(f"`output shape` ------ {y3.shape}")
print(f"`output result`------- {y3}")


# Test the IdentityModel
model3 = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
y4 = model3(x_input)  
y4_identity = model3(x_input, isidentity=True) 
print(f"\nIdentity Model")
print(f"`input`----------{x_input}")
print("`Network linear output`------{}; \n`network identity output`-------{}".format(y4, y4_identity))
print(f"`Network Linear results`------- {y4}")



# Gradient computation
# y = x2^2 with example: x2 = 3.0
x2 = torch.tensor(3.0, requires_grad=True)
y = x2 ** 2
y.backward()  # Compute the gradient

dy_dx = x2.grad  #.grad differenciates(similar to derivates) the function, which is 2 * x2
#method is useful for backpropragration 

print("\n\ndiferencial of dy/dx of y=x^2 at x2 = 3.0 is--------- ", dy_dx)
assert dy_dx == 6.0



# Function minimization with autograd and gradient descent
# Initialize a random value for our intial x
x = torch.randn(1)
print(f"Initializing x ---------- x={x.item()}")

learning_rate = 1e-2  # Learning rate
history = []
x_f = 4  # Target value


# We will run gradient descent for a number of iterations. At each iteration, we compute the loss, compute the derivative of the loss with respect to x, and perform the update.
for i in range(500):
    x = torch.tensor([x], requires_grad=True)

    
    loss = (x - x_f) ** 2                       # Compute the loss as the square of the difference between x and x_f
    loss.backward()                             # Backpropagate through the loss to compute gradients
    x = x.item() - learning_rate * x.grad       # Update x with gradient descent

    history.append(x.item())

# Plot the evolution of x as we optimize toward x_f!
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.savefig("my_plot.png", bbox_inches="tight")
plt.show()
plt.close()