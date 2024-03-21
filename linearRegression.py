import tensorflow as tf
from IPython.display import display, Math
import torch

a = torch.tensor([-2.0],requires_grad=True)
b = torch.tensor([3.0],requires_grad=True)
c = torch.tensor([10.0],requires_grad=True)
x = 3*(a**2) + 2*(b**2)
z = b**2 + c

z.retain_grad()
y = 5*(x**2) + 2*z

    y.backward()
    print(display(Math(fr'\frac{{\partial e}}{{\partial a}} = {a.grad.item()}')))
    print()
    print(display(Math(fr'\frac{{\partial e}}{{\partial b}} = {b.grad.item()}')))
    print()
    print(display(Math(fr'\frac{{\partial e}}{{\partial c}} = {c.grad.item()}')))
    print()
