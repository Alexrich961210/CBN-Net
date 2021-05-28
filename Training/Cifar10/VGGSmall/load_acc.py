import torch

checkpoint = torch.load("save_by2/Mesh2_Vggsmall_1w1a_0.99KD-5T_l12-2e-4.th")
print("the model accuracy is {:.2f}".format(checkpoint['best_prec1']))