import matplotlib.pyplot as plt
import numpy as np
import torch
x = torch.randn((128, 128)).numpy()
plt.imshow(x, cmap="Blues")
plt.show()
