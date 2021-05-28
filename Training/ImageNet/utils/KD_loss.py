# Code is modified from MEAL (https://arxiv.org/abs/1812.02425) and Label Refinery (https://arxiv.org/abs/1805.02641).

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.
    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""
    def __init__(self,T):
        super(DistributionLoss, self).__init__()
        self.T = T
    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output/self.T, dim=1)
        real_output_soft = F.softmax(real_output/self.T, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

def loss_fn_kd(outputs, labels, teacher_outputs, a,T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = a
    T = T
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                          F.softmax(teacher_outputs/T, dim=1),) * (alpha * T * T) + \
    #           F.cross_entropy(outputs, labels) * (1. - alpha)
    KD_loss = F.kl_div(F.log_softmax(outputs / T, dim=1),
                       F.softmax(teacher_outputs/ T, dim=1)) * (T *T*alpha) +\
              F.cross_entropy(outputs, labels) * (1. - alpha)
    KD_loss = KD_loss *10
    # KD_loss.cuda()
    return KD_loss