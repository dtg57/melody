from jaxtyping import Float
from torch import *
from check_dimensions import CheckDimensions

num1 = 2
num2 = 5

@CheckDimensions([num1, num2])
def a(x):
    return x.transpose(0, 1)

print(a(Tensor([[3, 1], [1, 2], [0, 8]])))