import torch
import numpy as np

# exploration of random tensors
shapeOne = (7, 7)
randomTensor0 = torch.rand(shapeOne)

shapeTwo = (1, 7)
randomTensor1 = torch.rand(shapeTwo)
randomTensor1_tranposed = torch.transpose(randomTensor1, 0, 1)
torch.matmul(randomTensor0, randomTensor1_tranposed)


# exploration of random tensors w/ seeds
randomSeed0 = 0

torch.manual_seed(seed = randomSeed0)

randomTensor2 = torch.rand(7, 7)
randomTensor3 = torch.rand(1, 7)
randomTensor3_transposed = torch.transpose(randomTensor3, 0, 1)

torch.matmul(randomTensor2, randomTensor3_transposed)

# GPU exploration + tensor aggregation
device = torch.device("mps")
torch.manual_seed(1234)

randomTensor4 = torch.rand(2, 3).to(device)
randomTensor5 = torch.rand(2, 3).to(device)

ProductFourFive = torch.matmul(randomTensor4, torch.transpose(randomTensor5, 0, 1))

ProductFourFive.min()
ProductFourFive.max()

ProductFourFive.argmax() 
ProductFourFive.argmin()

# experimenting with different dimensions
randomSeed1 = 7
torch.manual_seed(randomSeed1)
randomTensorSix = torch.rand(1, 1, 1, 10)

torch.manual_seed(randomSeed1)
randomTensorSeven = torch.rand(1, 1, 1, 10)
randomTensorSeven_squeezed = torch.squeeze(randomTensorSeven)