import torch
from models.student import VggSmall
import matplotlib.pyplot as plt

model = VggSmall.Bypass2_vgg_small_1w1a()
checkpoint = torch.load("save_by2/Mesh2_Vggsmall_1w1a_0.99KD-5T_l12-1.5e-4.th")
state_dict = checkpoint["state_dict"]
for name,value in state_dict.items():
    if "dropout9.w1" in name:
        data = value.view(-1).cpu()


plt.grid(linestyle='--')
plt.bar(range(len(data)),data,color='r')
plt.ylim(-0.25,0.25)
plt.ylabel("Weights")
plt.xlabel("Channels")
plt.show()