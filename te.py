import torch

mask = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1]])
print(torch.sum(mask,dim=0)>0)
