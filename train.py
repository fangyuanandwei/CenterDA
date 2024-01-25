from __future__ import print_function
import os
import argparse
import torch
import data_loader
from model import model
from util import to_cuda,print_learning_rate,test,save_ckpt
import torch.optim as optim
import math
import tqdm
import time
import numpy as np
from PIL import Image
import torch.nn.functional as F
import warnings
import platform
import torch.nn as nn
from functions import Pseudo_data,CenterLoss,CrossEntropyLabelSmooth
from trible_select import pdist_torch
from tensorboardX import SummaryWriter
from signal import signal,SIGPIPE,SIG_DFL
import Clustering
from model.mmd import MMD
from util import to_cuda,filter_samples,filter_class,to_onehot
signal(SIGPIPE,SIG_DFL)
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CDA')

parser.add_argument('--batch_size', type=int, default = 10, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default = 40, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')

parser.add_argument('--save_path', type=str, default="./save_ckpt/0510/",
                    help='the path to save the model')

parser.add_argument('--save_numpy', type=str, default="./save_numpy/0510/",
                    help='the path to save the numpy')

parser.add_argument('--root_path', type=str, default="/raid/huangl02/WGQ/DA_data/Office_31/",
                    help='the path to load the data')

parser.add_argument('--source_dir', type=str, default="amazon",
                    help='the name of the source dir')

parser.add_argument('--test_dir', type=str, default="dslr",
                    help='the name of the test dir')

parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')

parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')

parser.add_argument('--num_class', default=31, type=int,
                    help='the number of classes')

parser.add_argument('--gpu', default = 6, type=int)

parser.add_argument('--log',default = '0527.txt')

parser.add_argument('--min_class',default = 10)

args = parser.parse_args()
platform_ = platform.system()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data():
    source_dir = args.source_dir +'/'
    test_dir = args.test_dir + '/'
    source_train_loader = data_loader.load_training(args.root_path, source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, test_dir, args.batch_size, kwargs)

    source_test_loader = data_loader.load_testing(args.root_path, source_dir, args.batch_size, kwargs)
    target_test_loader  = data_loader.load_testing(args.root_path, test_dir, args.batch_size, kwargs)

    source_signal = data_loader.load_single(args.root_path, source_dir, args.batch_size, kwargs)
    target_signal = data_loader.load_single(args.root_path, test_dir, args.batch_size, kwargs)

    return source_train_loader, target_train_loader, source_test_loader,target_test_loader,source_signal,target_signal

def get_centers(model, dataloader, num_classes):
    centers = 0
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))

    for batch_idx,(data, gt) in tqdm.tqdm(enumerate(dataloader),total=len(source_train_loader), ncols=80,leave=False):
        data = to_cuda(data)
        gt = to_cuda(gt)

        feature = model(data,data)[2]
        feature = feature.data

        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)

    return centers

def filtering(clustered_target_samples,NUM_CLASSES):
    threshold = 1.0
    min_sn_cls = 3
    target_samples = clustered_target_samples

    # filtering the samples
    chosen_samples = filter_samples(target_samples, threshold=threshold)

    # filtering the classes
    filtered_classes = filter_class(chosen_samples['label'], min_sn_cls, NUM_CLASSES)

    print('The number of filtered classes: %d.' % len(filtered_classes))
    return chosen_samples, filtered_classes

def register_history(history,key, value, history_len):
    if key not in history:
        history[key] = [value]
    else:
        history[key] += [value]

    if len(history[key]) > history_len:
        history[key] = history[key][len(history[key]) - history_len:]

def complete_train(args,loop,history,clustered_target_samples):
    if loop > args.epochs:
        return True
    if 'target_centers' not in history or 'ts_center_dist' not in history or 'target_labels' not in history:
        return False
    if len(history['target_centers']) < 2 or len(history['ts_center_dist']) < 1 or len(history['target_labels']) < 2:
       return False

    # target centers along training
    target_centers = history['target_centers']
    get_dist = Clustering.DIST()
    eval1 = torch.mean(get_dist.get_dist(target_centers[-1], target_centers[-2])).item()

    # target-source center distances along training
    eval2 = history['ts_center_dist'][-1].item()

    # target labels along training
    path2label_hist = history['target_labels']
    paths = clustered_target_samples['data']
    num = 0
    for path in paths:
        pre_label = path2label_hist[-2][path]
        cur_label = path2label_hist[-1][path]
        if pre_label != cur_label:
            num += 1
    eval3 = 1.0 * num / len(paths)

    STOP_THRESHOLDS = (0.001, 0.001, 0.001)
    return (eval1 < STOP_THRESHOLDS[0] and eval2 < STOP_THRESHOLDS[1] and eval3 < STOP_THRESHOLDS[2])

def update_labels(model_train,source_train_loader,num_class,cluster):
    model_train.eval()
    inint_center = get_centers(model_train, source_train_loader, num_class)
    cluster.set_init_centers(inint_center)

def accuracy(preds, target):
    preds = torch.max(preds, dim=1)[1]
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)

def complete_training(loop,history):
    if loop >= args.epochs+1:
        return True

    if 'target_centers' not in history or \
            'ts_center_dist' not in history or \
            'target_labels' not in history:
        return False

    if len(history['target_centers']) < 2 or \
    len(history['ts_center_dist']) < 1 or \
    len(history['target_labels']) < 2:
       return False

    # target centers along training
    target_centers = history['target_centers']
    eval1 = torch.mean(Clustering.DIST.get_dist(target_centers[-1], target_centers[-2])).item()

    # target-source center distances along training
    eval2 = history['ts_center_dist'][-1].item()

    # target labels along training
    path2label_hist = history['target_labels']
    paths = clustered_target_samples['data']
    num = 0
    for path in paths:
        pre_label = path2label_hist[-2][path]
        cur_label = path2label_hist[-1][path]
        if pre_label != cur_label:
            num += 1
    eval3 = 1.0 * num / len(paths)
    STOP_THRESHOLDS = (0.001, 0.001, 0.001)
    return (eval1 < STOP_THRESHOLDS[0] and \
            eval2 < STOP_THRESHOLDS[1] and \
            eval3 < STOP_THRESHOLDS[2])

def source_sample(model,source_data):
    source_feat, source_gt, source_paths = [], [], []
    samples = {}
    model.eval()
    transform_signal = source_data.transform
    for example in tqdm.tqdm(source_data.imgs,total=len(source_data.imgs), ncols=80,leave=False):
        source_path = example[0]
        img = Image.open(source_path).convert('RGB')
        if transform_signal is not None:
            img = transform_signal(img)
        img = to_cuda(img)
        img = torch.unsqueeze(img, 0)

        out = model(img, img)
        feature_share = out[2]

        source_paths += [source_path]
        source_gt += [to_cuda(torch.tensor([example[1]]))]
        source_feat += [feature_share.data]

    samples['data'] = source_paths
    samples['gt'] = torch.cat(source_gt, dim=0) if len(source_gt) > 0 else None
    samples['feature'] = torch.cat(source_feat, dim=0)
    return samples

def split_samples_classwise(samples,numberclass):
    data = samples['data']
    label = samples['label']
    gt = samples['gt']
    samples_list = []

    for c in range(numberclass):
        mask = (label == c)
        data_c = [data[k] for k in range(mask.size(0)) if mask[k].item() == 1]
        label_c = torch.masked_select(label, mask)
        gt_c = torch.masked_select(gt, mask) if gt is not None else None
        samples_c = {}
        samples_c['data'] = data_c
        samples_c['label'] = label_c
        samples_c['gt'] = gt_c
        samples_list.append(samples_c)

    return samples_list

def construct_categorical_dataloader(samples, filtered_classes,numberclass,train_data):
    target_classwise = split_samples_classwise(samples, numberclass)
    dataloader = train_data['categorical']['loader']
    classnames = dataloader.classnames

    dataloader.class_set = [classnames[c] for c in filtered_classes]
    dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] for c in filtered_classes}

    dataloader.num_selected_classes = min(args.min_class, len(filtered_classes))
    dataloader.construct()

def init_data(dataloader):
    train_data = {key: dict() for key in dataloader if key != 'test'}

    for key in train_data.keys():
        if key not in dataloader:
            continue
        cur_dataloader = dataloader[key]
        train_data[key]['loader'] = cur_dataloader
        train_data[key]['iterator'] = None

    return train_data

def get_samples(train_data,data_name):
    assert(data_name in train_data)
    assert('loader' in train_data[data_name] and 'iterator' in train_data[data_name])

    data_loader = train_data[data_name]['loader']
    data_iterator = train_data[data_name]['iterator']
    assert data_loader is not None and data_iterator is not None, 'Check your dataloader of %s.' % data_name

    try:
        sample = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        sample = next(data_iterator)
        train_data[data_name]['iterator'] = data_iterator
    return sample

def CAS(train_data):
    samples = get_samples(train_data,'categorical')

    source_samples = samples['Img_source']
    source_sample_paths = samples['Path_source']
    source_nums = [len(paths) for paths in source_sample_paths]

    target_samples = samples['Img_target']
    target_sample_paths = samples['Path_target']
    target_nums = [len(paths) for paths in target_sample_paths]

    source_sample_labels = samples['Label_source']
    selected_classes = [labels[0].item() for labels in source_sample_labels]
    assert (selected_classes == [labels[0].item() for labels in samples['Label_target']])

    return source_samples, target_samples, samples

def train_class_wise(args,epoch_index,model,train_data,filtered_classes,
                     loss_func,crierion_cent,writer):

    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch_index - 1) / args.epochs), 0.75)
    center_mmd = MMD(num_layers=1, kernel_num=[5], kernel_mul=[2])
    xent = CrossEntropyLabelSmooth(num_class)
    if args.diff_lr:
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.bottleneck.parameters(),'lr': LEARNING_RATE},
            {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)

        optimizer_center = torch.optim.SGD(crierion_cent.parameters(), lr=0.5,
                                           momentum=args.momentum, weight_decay=args.l2_decay)

    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum, weight_decay=args.l2_decay)
        optimizer_center = torch.optim.SGD(crierion_cent.parameters(), lr=0.1)

    iters_per_loop = int(len(train_data['categorical']['loader'])) * 1.0
    train_data['categorical']['iterator'] = iter(train_data['categorical']['loader'])
    stop = False
    train_iter = 0

    model.train()

    while not stop:
        optimizer.zero_grad()
        optimizer_center.zero_grad()


        source_samples_cls, target_samples_cls, cas_sample = CAS(train_data)

        source_samples_cls_lable = cas_sample['Label_source']
        source_samples_cls_data_cat = to_cuda(torch.cat([to_cuda(source_data) for source_data in source_samples_cls], dim=0))
        source_samples_cls_lable_cat = to_cuda(torch.cat([to_cuda(source_label) for source_label in source_samples_cls_lable], dim=0))

        target_samples_cls_lable = cas_sample['Label_target']
        target_samples_cls_data_cat = to_cuda(torch.cat([to_cuda(target_data) for target_data in target_samples_cls], dim=0))
        target_samples_cls_lable_cat = to_cuda(torch.cat([to_cuda(target_label) for target_label in target_samples_cls_lable], dim=0))

        #run model
        out_center = model_class_wise(source_samples_cls_data_cat, target_samples_cls_data_cat)

        #class loss
        loss_cls = xent(out_center[0],source_samples_cls_lable_cat)

        #center loss
        s_feature,t_feature = out_center[2],out_center[3]
        center_feature = torch.cat([s_feature,t_feature],dim=0)
        center_label = torch.cat([source_samples_cls_lable_cat,target_samples_cls_lable_cat],dim=0)
        weight_cent = 0.005
        center_loss = 0.005 * crierion_cent(center_feature, center_label)

        #MMD
        mmd_weight = 1
        mmd_loss = mmd_weight * center_mmd.forward([s_feature], [t_feature])['mmd']

        #total loss
        loss = loss_cls + mmd_loss + center_loss
        loss.backward()
        optimizer.step()

        for param in crierion_cent.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_center.step()

        if (train_iter + 1)%(max(1,iters_per_loop//6))==0:
            print('source:{},target:{},Epoch:{}/{},iter:{},loss_cls:{},mmd_loss:{},center_loss:{},learn:{:.4f}'.
                  format(args.source_dir, args.test_dir, epoch_index, args.epochs, train_iter + 1,
                         loss_cls.item(), mmd_loss.item(),center_loss.item(),LEARNING_RATE))
        train_iter += 1
        if train_iter >= iters_per_loop:
            stop = True
        else:
            stop = False


def make_loss_with_center(num_class,feat_dim,label_smooth):
    center_criterion = CenterLoss(num_class,feat_dim)
    xent = CrossEntropyLabelSmooth(num_class)
    def loss_function(source_feature,source_label,center_feature,center_label):
        if label_smooth:
            return xent(source_feature, source_label) + center_loss_weight * center_criterion(center_feature, center_label)
        else:
            return F.cross_entropy(source_feature,source_label) + center_loss_weight*center_criterion(center_feature,center_label)

    return loss_function,center_criterion


if __name__ == '__main__':

    #Image_CLEF
    # args.root_path = '/raid/huangl02/WGQ/DA_data/Image_CLEF/'
    # source_list = ['i_split','c_split','p_split']
    # target_list = ['i_split','c_split','p_split']

    #Offfice_home
    args.root_path = '/raid/huangl02/WGQ/DA_data/OfficeHome/'
    source_list = ['Art','Clipart','Product','Real World']
    target_list = ['Art','Clipart','Product','Real World']

    #Offfice_31
    # args.root_path = '/raid/huangl02/WGQ/DA_data/Office_31/'
    # source_list = ['Art','Clipart','Product','Real World']
    # target_list = ['Art','Clipart','Product','Real World']

    args.min_class = 8

    for source_index in source_list:
        for target_index in target_list:
            if source_index == target_index:
                continue

            args.source_dir = source_index
            args.test_dir = target_index
            args.epochs = 50

            print('Source:{} To Target:{}'.format(args.source_dir, args.test_dir))
            log_dir = './OfficeHome_0709/log_{}_{}'.format(args.source_dir, args.test_dir)
            comment = '0603_{}_{}'.format(args.source_dir, args.test_dir)
            writer = SummaryWriter(log_dir=log_dir, comment=comment)
            source_train_loader, target_train_loader, source_test_loader,target_test_loader, source_data, target_data = load_data()
            args.num_class = len(source_train_loader.dataset.classes)

            #model and center loss
            model_class_wise = to_cuda(model.CDANet(number_classes = args.num_class, base_net='ResNet50'))
            loss_func,center_criterion = make_loss_with_center(args.num_class,256,label_smooth=False)

            max_correct = 0
            max_epoch = 0
            center = None
            target_peodual = None

            dataloaders = {}
            dataset_type = 'CategoricalSTDataset'
            source_batch_size = 3
            target_batch_size = 3
            dataroot_S = os.path.join(args.root_path, args.source_dir)
            dataroot_T = os.path.join(args.root_path, args.test_dir)

            train_transform = source_train_loader.dataset.transform
            classes = source_train_loader.dataset.classes

            from class_wise_data import ClassAwareDataLoader
            dataloaders['categorical'] = ClassAwareDataLoader(
                dataset_type=dataset_type,
                source_batch_size=source_batch_size,
                target_batch_size=target_batch_size,
                source_dataset_root=dataroot_S,
                transform = train_transform,
                classnames = classes,
                num_workers=1,
                drop_last=True, sampler='RandomSampler')

            train_data = init_data(dataloaders)
            num_class = len(source_train_loader.dataset.classes)
            cluster = Clustering.Clustering()

            for epoch_index in range(1,args.epochs+1):
                print('Epoch:{}/{}'.format(epoch_index, args.epochs))
                with torch.no_grad():
                    history = {}
                    update_labels(model_class_wise, source_train_loader, num_class, cluster)
                    cluster.feature_clustering(model_class_wise, target_data)

                    source_samples = source_sample(model_class_wise, source_data)

                    clustered_target_samples = cluster.samples
                    target_centers = cluster.centers
                    center_change = cluster.center_change
                    path2label = cluster.path2label

                    register_history(history,'target_centers', target_centers,2)
                    register_history(history,'ts_center_dist', center_change, 2)
                    register_history(history,'target_labels', path2label,     2)

                    if clustered_target_samples is not None and clustered_target_samples['gt'] is not None:
                        preds = to_onehot(clustered_target_samples['label'], num_class)
                        gts = clustered_target_samples['gt']
                        res = accuracy(preds, gts)
                        print('Clustering %s: %.4f' % ('accuracy', res))
                        writer.add_scalar('Clustering_acc',res,epoch_index)

                    # check if meet the stop condition
                    stop = complete_training(epoch_index,history)
                    print('Stop:{}'.format(stop))
                    if stop: break

                    # filtering the clustering results
                    target_filter, target_filtered_classes = filtering(clustered_target_samples,num_class)
                    construct_categorical_dataloader(target_filter, target_filtered_classes,num_class,train_data)

                train_class_wise(args,epoch_index,model_class_wise,train_data, target_filtered_classes,
                                 loss_func,center_criterion,writer)

                source_correct,source_loss = test(epoch_index, args, model_class_wise, source_test_loader)
                target_correct,target_loss = test(epoch_index, args, model_class_wise, target_test_loader)
                source_acc = source_correct.item()/len(source_test_loader.dataset)
                target_acc = target_correct.item()/len(target_test_loader.dataset)

                print('epoch:{},source_acc:{},source_loss:{}'.format(epoch_index,source_acc,source_loss))
                print('epoch:{},target_acc:{},target_loss:{}\n'.format(epoch_index,target_acc,target_loss))


                writer.add_scalar('source_acc', source_acc, epoch_index)
                writer.add_scalar('target_acc', target_acc, epoch_index)

            save_path = './OfficeHome_0709/{}_{}_model_class_wise.ckpt'.format(args.source_dir,args.test_dir)
            save_ckpt(model_class_wise,args.epochs,save_path)
            model_class_wise = None

