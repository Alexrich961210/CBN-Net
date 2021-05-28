#-*-coding:utf-8-*-
import torch.utils
import torch.utils.data.distributed
from utils.utils import *
from torchvision import datasets, transforms
# from resnet.step2 import birealnet
from torch.cuda.amp import autocast as autocast
import torchvision.models as models
from utils import KD_loss
import matplotlib.pyplot as plt
from models import birealnet

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
parser = argparse.ArgumentParser("birealnet18")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./ImageNet_Save', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', default='/home/cqdx/ImageNet',help='path to dataset')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--T', '-temp', default=1, type=int,help='temperature controls soft degree (default: 1)')
parser.add_argument('--a', default=0.9, type=float, help='balance loss weight')
parser.add_argument('--sim', default=0.1, type=float, help='balance loss weight')
parser.add_argument('--l2', default=1e-2, type=float, help='distance coefficient')
parser.add_argument('--l12', default=2e-4, type=float, help='L12 regularization coefficient')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--type',default='torch.cuda.FloatTensor',help='type of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--start_epoch',default=-1,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--warm_up',dest='warm_up',action='store_true',default=False,help='use warm up or not')
args = parser.parse_args()
CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

save_name = 'Imagenet_CBN2v1_Bireal18_ReAct_kd_wd0_l12-2e-4_pca_sgd_pretrained_3_lr-0.1_epoch-120'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/{}.txt'.format(save_name)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(args)
def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        cudnn.benchmark = True

    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    model_student = birealnet.Meshk2_resnet18()
    model_student = nn.DataParallel(model_student).cuda()
    checkpoint_react1 = torch.load('save/checkpoint.pth.tar')
    # # checkpoint_react2 = torch.load('ImageNet_Save/Imagenet_MeshBireal18_ReAct_kd_wd0_pca_l12-2e-4_sgdmodel_best.pth.tar')
    state_dict1 = checkpoint_react1['state_dict']
    # state_dict2 = checkpoint_react2['state_dict']
    # model_student.load_state_dict(state_dict)
    # print('the best top1 acc is {:.2f}% and top5 acc is 80.99% on epoch {}'.format(acc, epoch))
    dict_new = model_student.state_dict().copy()
    list_dict_new = list(dict_new.keys())
    for i in range(6):
        dict_new[list_dict_new[i]] = state_dict1[list_dict_new[i]]
    for k, v in dict_new.items():
        if 'block1' in k:
            k_new = k[:16] + k[23:]
            dict_new[k] = state_dict1[k_new]
        if 'block2' in k:
            k_new = k[:16] + k[23:]
            dict_new[k] = state_dict1[k_new]
    model_student.load_state_dict(dict_new)
    # logging.info('student:')
    # for layer in model_student.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         print(layer)
    #         layer.float()
    # model_student = nn.DataParallel(model_student).cuda()
    logging.info(model_student)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    criterion_kd = KD_loss.DistributionLoss(args.T)
    criterion_kd.cuda()

    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if 'dropout' not in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.SGD(
            [{'params' : other_parameters,'initial_lr':args.lr},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay,'initial_lr':args.lr}],
            lr=args.lr,momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up * 4, eta_min=0,
                                                              last_epoch=args.start_epoch - args.warm_up * 4)

    # 在训练最开始之前实例化一个GradScaler对象
    best_top1_acc= 0

    # start_epoch = args.start_epoch + 1
    checkpoint_tar = os.path.join(args.save, '{}checkpoint.pth.tar'.format(save_name))
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        args.start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    #adjust the learning rate according to the checkpoint
    for epoch in range(args.start_epoch+1):
        scheduler.step()

    # load training data
    traindir = os.path.join('/home/cqdx/ImageNet/', 'train')
    valdir = os.path.join('/home/cqdx/ImageNet/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #
    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        # PCANoise(0.1),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # train the model
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.start_epoch+1, args.epochs):
        #*warm up
        if args.warm_up and epoch <5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])
        end = time.time()
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_obj, train_top1_acc,  train_top5_acc, weightlist = train(epoch,scaler,train_loader, model_student,model_teacher,criterion_kd,optimizer,args.a,args.T,args.l12)
        #* adjust Lr
        if epoch >= 4 * args.warm_up:
            scheduler.step()
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch,val_loader, model_student, criterion)
        print("one epoch needs ", (time.time()-end))
        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_student.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            # 'amp': amp.state_dict(),
            }, is_best, args.save,save_name)
        #
        if (epoch+1) == 1:
            plt.figure(epoch)
            plt.bar(range(len(weightlist)), weightlist)
            plt.savefig("./results/lateral/{}_{}epoch.jpg".format(save_name, epoch + 1))
        if (epoch+1) % 3 == 0:
            plt.figure(epoch)
            plt.bar(range(len(weightlist)), weightlist)
            plt.savefig("./results/lateral/{}_{}epoch.jpg".format(save_name,epoch+1))

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch,scaler, train_loader, model_student,model_teacher,criterion,optimizer,a,T,l12):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    model_student.train()
    # model_teacher.eval()
    end = time.time()


    # for param_group in optimizer.param_groups:
    #     cur_lr = param_group['lr']
    weight_list = []
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        target_var = target
        # compute outputy
        with autocast():
            logits_student = model_student(images)
            logits_teacher = model_teacher(images)
            sim = 0
            for j in range(4):
                sim += torch.norm(model_student.module.layer1[j].dropout1.w1, p=2) + torch.norm(
                    model_student.module.layer1[j].dropout2.w1, p=2)
                sim += torch.norm(model_student.module.layer2[j].dropout1.w1, p=2) + torch.norm(model_student.module.layer2[j].dropout2.w1, p=2)
                sim += torch.norm(model_student.module.layer3[j].dropout1.w1, p=2) + torch.norm(model_student.module.layer3[j].dropout2.w1, p=2)
                sim += torch.norm(model_student.module.layer4[j].dropout1.w1, p=2) + torch.norm(
                    model_student.module.layer4[j].dropout2.w1, p=2)
            sim.cuda()
        # loss =  a* criterion_kd(logits_student,logits_teacher) + (1.-a)*criterion(logits_student, target_var)\
            loss = criterion(logits_student,logits_teacher)\
                    + sim * l12\
                   # + cos * sloss.mean()


        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # loss.backward()
        # Scales loss. 为了梯度放大.
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)

        # 准备着，看是否要增大scaler
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,top5=top5))

        # progress.display(i)
    for m in range(4):
        weight_list.append((torch.norm(model_student.module.layer1[m].dropout1.w1,p=1)).item())
        weight_list.append((torch.norm(model_student.module.layer1[m].dropout2.w1,p=1)).item())
    for n in range(4):
        weight_list.append((torch.norm(model_student.module.layer2[n].dropout1.w1,p=1)).item())
        weight_list.append((torch.norm(model_student.module.layer2[n].dropout2.w1,p=1)).item())
    for k in range(4):
        weight_list.append((torch.norm(model_student.module.layer3[k].dropout1.w1,p=1)).item())
        weight_list.append((torch.norm(model_student.module.layer3[k].dropout2.w1,p=1)).item())
    for l in range(4):
        weight_list.append((torch.norm(model_student.module.layer4[l].dropout1.w1,p=1)).item())
        weight_list.append((torch.norm(model_student.module.layer4[l].dropout2.w1,p=1)).item())

    return losses.avg, top1.avg, top5.avg,weight_list

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
            # images = images.cuda()
            # target = target.cuda()
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
