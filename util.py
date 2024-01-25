import os
import torch
import logging
import numpy as np
import torch.nn.functional as F


def filter_samples(samples, threshold=0.05):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    mask = min_dist < threshold
    # mask_label = (samples['label']==samples['predictlabel'])
    # mask = (mask_dist&mask_label)


    filtered_data = [samples['data'][m]	for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples

def filter_class(labels, num_min, num_classes):
    filted_classes = []
    for c in range(num_classes):
        mask = (labels == c)
        count = torch.sum(mask).item()
        if count >= num_min:
            filted_classes.append(c)

    return filted_classes

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k is 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)

def save_ckpt(model,epoch,save_path):
    state = {'net': model.state_dict(), 'epoch': epoch}
    torch.save(state, save_path)

def save_numpy(model,save_path,epoch,test_loader,save_name):
    model.eval()
    with torch.no_grad():
        all_features = []
        all_labels = []
        for data, target in test_loader:
            data, target = to_cuda(data), to_cuda(target)
            out = model(data, data, target)
            feature_share = out[3]

            if torch.cuda.is_available():
                all_features.append(feature_share.data.cpu().numpy())
                all_labels.append(target.data.cpu().numpy())
            else:
                all_features.append(feature_share.data.numpy())
                all_labels.append(target.data.numpy())

        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)

        all_features_file = os.path.join(save_path, save_name+ '_all_features.npy')
        all_label_file = os.path.join(save_path, save_name + '_all_labels.npy')

        np.save(all_features_file,all_features)
        np.save(all_label_file, all_labels)

def get_cluster_center(feature, labels,number_class, centers):

    num_classes = number_class
    if centers is None:
        centers = 0
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))

    start = 0
    end = 0

    while end < len(feature):
        iter_number = 100
        end = start + iter_number
        if end > len(feature):
            end = len(feature)

        feature_iter = feature[start:end]
        culster_gt = labels[start:end]

        if torch.cuda.is_available():
            mask = (culster_gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        else:
            mask = (culster_gt == refs).unsqueeze(2).type(torch.FloatTensor)

        feature_iter = feature_iter.unsqueeze(0)
        centers += torch.sum(feature_iter * mask, dim=1)
        start = end

    if torch.cuda.is_available():
        label_numpy = labels.cpu().numpy()
        centers = centers.cpu().numpy()
    else:
        label_numpy = labels.numpy()
        centers = centers.numpy()

    for class_index in range(number_class):
        class_number = np.sum(label_numpy==class_index)
        centers[class_index] = centers[class_index] / class_number
    centers = torch.from_numpy(centers)
    return to_cuda(centers)

def Cos_Distance_feature_center(feature, center, cross=False):
    pointA = F.normalize(feature, dim=1)
    pointB = F.normalize(center, dim=1)
    if not cross:
        return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
    else:
        assert (pointA.size(1) == pointB.size(1))
        return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))

def test(epoch,args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = to_cuda(data), to_cuda(target)
            out = model(data, data)
            s_output = out[0]
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item() # sum up batch loss
            pred = s_output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)

        # print( 'Epoch:{}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        #     epoch,test_loss, correct, len(test_loader.dataset),
        #     100. * correct.item() / len(test_loader.dataset)))

    return correct,test_loss

