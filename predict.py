import scipy.stats as stats
from PIL import Image
import torch.nn as nn
import torch

import numpy as np

bmd=np.load("bmd820.npy")
ref_bmd = 0.9747
std = 0.1185
tsc = (bmd - ref_bmd)/std
label = (tsc<-2.5).astype(int)
print(label)
CT = np.load('input2.npy').reshape(-1,1,256,256)
CT = torch.Tensor(CT)

device='cuda' if torch.cuda.is_available() else 'cpu'

CT = CT.to(device).float()
model = torch.load('predictSick.pt')
criterion = nn.MSELoss()

pred = model(CT)
pred = pred.cpu()
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
pred = pred.detach().cpu().numpy().squeeze().astype(np.uint8)
np.save('result.npy', pred)
print(pred)