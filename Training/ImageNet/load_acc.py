import torch

checkpoint = torch.load('ImageNet_Save/Imagenet_CBN2_Bireal18_ReAct_kd_wd0_l12-2e-4_pca_sgd_pretrained_3model_best.pth.tar')
# model_student.load_state_dict(checkpoint['state_dict'])
print('the best prec1 acc is {:.2f}'.format(checkpoint['best_top1_acc']))
