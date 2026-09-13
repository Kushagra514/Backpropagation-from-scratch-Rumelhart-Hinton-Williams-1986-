import torch
import torch.nn as nn

#same dataset
x_torch = torch.tensor([
    [2.0,3.0],
    [1.0,1.0],
    [3.0,2.0],
    [0.0,1.0]
])

y_torch = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

model = nn.Sequential(
    nn.Linear(2,2),
    nn.Sigmoid(),
    nn.Linear(2,1),
    nn.Sigmoid()
)

#same initial parameters
with torch.no_grad():
    model[0].weight.copy_(torch.tensor([
        [0.5,0.2],
        [0.1,0.4]
    ]))

    model[0].bias.copy_(torch.tensor([
        0.1,
        0.2 
    ]))
    model[2].weight.copy_(torch.tensor([
        [0.3,0.6]
    ]))
    model[2].bias.copy_(torch.tensor([
        0.1
    ]))


def loss_function(y_hat,y):
    return 0.5 * torch.mean((y_hat-y) ** 2)

#forward pass
y_hat_torch = model(x_torch)

loss_torch = loss_function(y_hat_torch,y_torch)

#backpropogation 
loss_torch.backward()

print("Pytorch Loss:",loss_torch.item())

print("\nPytorch gradients:")
print("dw1:")
print(model[0].weight.grad)

print("\ndb1:")
print(model[0].bias.grad)

print("\ndw2:")
print(model[2].weight.grad)

print("\ndb2:")
print(model[2].bias.grad)

