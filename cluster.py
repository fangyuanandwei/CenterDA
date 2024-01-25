import torch
import Clustering
from util import to_cuda,filter_samples,filter_class

def get_centers(model, dataloader, num_classes):
    centers = 0
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))
    for data, gt in dataloader:

        data = to_cuda(data)
        gt = to_cuda(gt)

        feature = model(data,data)[4]
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
