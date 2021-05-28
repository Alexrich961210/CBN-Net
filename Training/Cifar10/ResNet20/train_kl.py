import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging
import sys
from util import KD_loss
import matplotlib.pyplot as plt
from models.student import ResNet20
from models.teacher import Fp_ResNet20
from util.KD_loss import DistributionLoss
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--a', '--a_kd', default=0.99, type=float,
                    metavar='A', help='weight decay (default: 0.5)')
parser.add_argument('--T', '--temperature', default=5, type=float,
                    metavar='A', help='temperature (default: 1)')
parser.add_argument('--l12', '--l12_regularization', default=9e-4, type=float,
                    metavar='A', help='l12 weight (default: 3e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_by2', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0
arg1 = parser.parse_args()

save_name = 'CBNNet20_K2_l12-9e-4_kd'
if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format)
fh = logging.FileHandler(os.path.join('log/{}.txt'.format(save_name)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    global args, best_prec1
    args = parser.parse_args()

    start_t = time.time()
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.cuda.set_device(0)
    # load teacher model
    model_teacher = Fp_ResNet20.MeshNet20_K2().cuda()
    model_teacher.load_state_dict(torch.load("save_by2/Fp_MeshNet20_l12-3e-4.th")['state_dict'])
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    logging.info("model: ")

    #create the model
    model = ResNet20.MeshNet20_K2()
    logging.info(model)
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_kd = KD_loss.DistributionLoss()
    criterion_kd.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if 'dropout' not in pname :
            weight_parameters.append(p)
            print(pname)

    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.SGD(
        [{'params': weight_parameters, 'weight_decay': args.weight_decay},
         {'params':other_parameters}], lr=args.lr,momentum=args.momentum)
    # -----------------------cosine学习率退火
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)


    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    loss_train = []
    acc_train = []
    acc_val = []

    for epoch in range(args.start_epoch, args.epochs):


        # train for one epoch
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        trainloss,trainacc,weightlist = train(train_loader, model,model_teacher,optimizer, epoch,criterion,args.l12)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, '{}.th'.format(save_name)))
        loss_train.extend(trainloss)
        acc_train.extend(trainacc)
        acc_val.append(prec1)
        if (epoch+1) == 1:
            plt.figure(epoch)
            plt.bar(range(len(weightlist)), weightlist)
            plt.savefig("./results/lateral/{}_{}epoch.jpg".format(save_name, epoch + 1))
        if (epoch+1) % 40 == 0:
            plt.figure(epoch)
            plt.bar(range(len(weightlist)), weightlist)
            plt.savefig("./results/lateral/{}_{}epoch.jpg".format(save_name,epoch+1))
    training_time = (time.time() - start_t) / 3600
    logging.info('total training time = {} hours'.format(training_time))
    np.save("./save_train/{}_loss_train.npy".format(save_name), loss_train)
    np.save("./save_train/{}_acc_train.npy".format(save_name), acc_train)
    np.save("./save_train/{}_acc_val.npy".format(save_name), acc_val)
    plt.subplot(1,2,1)
    plt.plot(loss_train)
    plt.subplot(1,2,2)
    plt.plot(acc_train)
    plt.savefig("./results/{}.jpg".format(save_name))
    # plt.show()


def train(train_loader, model,model_teacher,optimizer, epoch,criterion,l12):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model_teacher.eval()

    end = time.time()
    loss_train = []
    acc_train = []
    weight_list = []
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        logits_student = model(input_var)
        logits_teacher = model_teacher(input_var)
        sim = 0
        for j in range(6):
            sim += torch.norm(model.layer1[j].dropout1.w1,p=2) +torch.norm(model.layer1[j].dropout2.w1,p=2)
            sim += torch.norm(model.layer2[j].dropout1.w1,p=2) +torch.norm(model.layer2[j].dropout2.w1,p=2)
            sim += torch.norm(model.layer3[j].dropout1.w1, p=2) + torch.norm(model.layer3[j].dropout2.w1, p=2)
        sim.cuda()

        loss = KD_loss.loss_fn_kd(logits_student,target_var,logits_teacher,args.a,args.T)\
               + l12 * sim
        # loss = criterion(logits_student, target_var) + l12 * sim
        # use this loss for any training statistics

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        logits_student = logits_student.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(logits_student.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    for i in range(6):
        weight_list.append(torch.norm(model.layer1[i].dropout1.w1, p=1))
        weight_list.append(torch.norm(model.layer1[i].dropout2.w1, p=1))
    for i in range(6):
        weight_list.append(torch.norm(model.layer2[i].dropout1.w1, p=1))
        weight_list.append(torch.norm(model.layer2[i].dropout2.w1, p=1))
    for i in range(6):
        weight_list.append(torch.norm(model.layer3[i].dropout1.w1, p=1))
        weight_list.append(torch.norm(model.layer3[i].dropout2.w1, p=1))
    loss_train.append(losses.val)
    acc_train.append(top1.val)

    return loss_train,acc_train,weight_list

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

            # if args.half:
            #     input_var = input_var.half()

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

            if i % args.print_freq == 0:
                logging.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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