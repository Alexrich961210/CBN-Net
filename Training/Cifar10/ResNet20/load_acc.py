import torch

checkpoint = torch.load("save_by2/Fp_MeshNet20_l12-3e-4.th")
print("the model accuracy is {:.2f}".format(checkpoint['best_prec1']))