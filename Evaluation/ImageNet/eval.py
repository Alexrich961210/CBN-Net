import torch
import torch.utils
import torch.utils.data.distributed
import torch.nn as nn
from torchvision import datasets, transforms
from util.utils import *
from models import birealnet
from torch.cuda.amp import autocast as autocast
import logging

save_name = 'CBN-Net_K2_Res18_kd_l21-2e-4'
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format)
fh = logging.FileHandler(os.path.join('results/{}.txt'.format(save_name)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():

    model_student = birealnet.Meshk2_resnet18()
    model_student = nn.DataParallel(model_student).cuda()
    checkpoint_react = torch.load(
        'save/CBN-Net_K2_Res18_kd_l21-2e-4model_best.pth.tar')
    acc = checkpoint_react['best_top1_acc']
    epoch = checkpoint_react['epoch']
    state_dict = checkpoint_react['state_dict']
    model_student.load_state_dict(state_dict)
    logging.info('Start evaluating CBN-Net (K=2) on ImageNet dataset: ')


    criterion = nn.CrossEntropyLoss().cuda()

    # valdir = os.path.join('data/', 'val')
    valdir = os.path.join('data','val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=8,pin_memory=True)
    _,top1,_ = validate(epoch, val_loader,model_student,criterion)
    print('the best accuracy of this model is',top1)

def validate(epoch,val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            with autocast():
                # compute output
                logits= model(images)
                loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        logging.info(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
