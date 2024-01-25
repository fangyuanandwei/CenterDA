import Clustering
import torch
from util import to_cuda,to_onehot

def complete_training(loop,args,history,clustered_target_samples,clustering):
    if loop >= args.epochs:
        return True

    if 'target_centers' not in history or 'ts_center_dist' not in history or 'target_labels' not in history:
        return False

    if len(history['target_centers']) < 2 or \
            len(history['ts_center_dist']) < 1 or \
            len(history['target_labels']) < 2:
        return False

    # target centers along training
    target_centers = history['target_centers']
    eval1 = torch.mean(clustering.Dist.get_dist(target_centers[-1],
                                                     target_centers[-2])).item()

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

def get_centers(net, dataloader, num_classes, key='feat'):
    centers = 0
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))
    for data,label in dataloader:
        data = to_cuda(data)
        gt = to_cuda(label)
        output = net(data,data)

        feature = output[4].data

        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)

        centers += torch.sum(feature * mask, dim=1)

    return centers

def update_labels(net,source_dataloader,target_dataloader,clustering):
    net.eval()
    num_classes = len(source_dataloader.dataset.classes)
    source_centers = get_centers(net,source_dataloader,num_classes)
    init_target_centers = source_centers

    clustering.set_init_centers(init_target_centers)
    clustering.feature_clustering(net,target_dataloader)

def register_history(history, key, value, history_len):
    if key not in history:
        history[key] = [value]
    else:
        history[key] += [value]

    if len(history[key]) > history_len:
        history[key] = history[key][len(history[key]) - history_len:]

def model_eval(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)

def filter_samples(samples, threshold=0.05):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    mask = min_dist < threshold

    filtered_data = [samples['data'][m]
		for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) \
                     if samples['gt'] is not None else None

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

def filtering(clustered_target_samples,NumberClass):
    threshold = 1.0
    min_sn_cls = 3
    target_samples = clustered_target_samples

    # filtering the samples
    chosen_samples = filter_samples(target_samples, threshold=threshold)

    # filtering the classes
    filtered_classes = filter_class(chosen_samples['label'], min_sn_cls, NumberClass)

    print('The number of filtered classes: %d.' % len(filtered_classes))
    return chosen_samples, filtered_classes


def get_target_cluster_label(args,model,source_dataloader,target_dataloader,clustering):
    stop = False
    history = {}
    loop = 0

    num_classes = len(source_dataloader.dataset.classes)
    while True:
        with torch.no_grad():
            update_labels(model,source_dataloader,target_dataloader,clustering)

            clustered_target_samples = clustering.samples
            target_centers = clustering.centers
            center_change = clustering.center_change
            path2label = clustering.path2label

            register_history(history,'target_centers', target_centers, 2)
            register_history(history,'ts_center_dist', center_change, 2)
            register_history(history,'target_labels', path2label, 2)

            if clustered_target_samples is not None and clustered_target_samples['gt'] is not None:
                preds = to_onehot(clustered_target_samples['label'], num_classes)
                gts = clustered_target_samples['gt']
                res = model_eval(preds, gts)
                print('Clustering %s: %.4f' % ('accuracy', res))

            stop = complete_training(loop,args,history,clustered_target_samples,clustering)
            print('Stop:{}'.format(stop))
            if stop: break

            target_hypt, filtered_classes = filtering(clustered_target_samples,num_classes)











