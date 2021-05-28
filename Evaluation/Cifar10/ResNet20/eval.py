import torch
from models import ResNet20_Hardtanh
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import torch.nn as nn
import random
import numpy as np
import logging
import sys
import os

save_name = 'CBN-Net with ResNet20 on CIFAR10'
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format)
fh = logging.FileHandler(os.path.join('results/{}.txt'.format(save_name)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():

    # load from CBN-Net (K=2)
    checkpoint1 = torch.load("save/CBN-Net(LC&KD)_K-2_L21-1e-4_0.99a5T.th",map_location='cuda:1')
    model1 = ResNet20_Hardtanh.CBN20_K2().cuda()
    state_dict1 = checkpoint1['state_dict']
    model1.load_state_dict(state_dict1)

    # load from CBN-Net (K=5)
    checkpoint2 = torch.load("save/CBN-Net(LC&KD)_K-5_L21-1e-4_0.99a5T.th",map_location='cuda:1')
    model2 = ResNet20_Hardtanh.CBN20_K5().cuda()
    state_dict2 = checkpoint2['state_dict']
    model2.load_state_dict(state_dict2)

    # load dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=16)

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # test our model
    logging.info('Start evaluating CBN-Net (K=2):')
    acc1 = validate(val_loader, model1, criterion)
    logging.info("the best accuracy of CBN-Net (K=2) ResNet20 on Cifar-10 is {:.2f} ".format(acc1))

    logging.info('Start evaluating CBN-Net (K=5):')
    acc2 = validate(val_loader, model2, criterion)
    logging.info("the best accuracy of CBN-Net (K=5) ResNet20 on Cifar-10 is {:.2f}".format(acc2))

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                logging.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()