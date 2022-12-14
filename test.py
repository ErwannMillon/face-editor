import torch
import torchvision
mask = torch.load("attn_mask.pt")
# print(mask.shape)
# mask = torchvision.transforms.ToTensor()(mask)
x = torchvision.transforms.functional.resize(mask, (10, 10))
print(x.shape)