import os
from PIL import Image
from torch.utils.data import Dataset
import random
from math import ceil
import torch

def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate

def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_with_labels(dir, classnames):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels

def make_dataset_classwise(dir, category):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname != category:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class CategoricalSTDataset(Dataset):
    def __init__(self):
        super(CategoricalSTDataset, self).__init__()

    def initialize(self, source_root, target_paths,
                   classnames, class_set,
                   source_batch_size,
                   target_batch_size, seed=None,
                   transform=None, **kwargs):

        self.source_root = source_root
        self.target_paths = target_paths

        self.transform = transform
        self.class_set = class_set

        self.data_paths = {}
        self.data_paths['source'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source'][cid] = make_dataset_classwise(self.source_root, c)
            cid += 1

        self.data_paths['target'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['target'][cid] = self.target_paths[c]
            cid += 1

        self.seed = seed
        self.classnames = classnames

        self.batch_sizes = {}

        for d in ['source', 'target']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = source_batch_size if d == 'source' else target_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1

    def __getitem__(self, index):
        data = {}
        for d in ['source', 'target']:
            cur_paths = self.data_paths[d]
            if self.seed is not None:
                random.seed(self.seed)

            inds = random.sample(range(len(cur_paths[index])), self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_' + d] = path
            assert (len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_' + d not in data:
                    data['Img_' + d] = [img]
                else:
                    data['Img_' + d] += [img]

            data['Label_' + d] = [self.classnames.index(self.class_set[index])] * len(data['Img_' + d])
            data['Img_' + d] = torch.stack(data['Img_' + d], dim=0)

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSTDataset'

class ClassAwareDataLoader(object):
    def name(self):
        return 'ClassAwareDataLoader'

    def __init__(self, source_batch_size, target_batch_size,
                 source_dataset_root="", target_paths=[],
                 transform=None, classnames=[],
                 class_set=[], num_selected_classes=0,
                 seed=None, num_workers=0, drop_last=True,
                 sampler='RandomSampler', **kwargs):

        # dataset type
        self.dataset = CategoricalSTDataset()

        # dataset parameters
        self.source_dataset_root = source_dataset_root
        self.target_paths = target_paths
        self.classnames = classnames
        self.class_set = class_set
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source_root=self.source_dataset_root,
                                target_paths=self.target_paths,
                                classnames=self.classnames, class_set=self.class_set,
                                source_batch_size=self.source_batch_size,
                                target_batch_size=self.target_batch_size,
                                seed=self.seed, transform=self.transform,
                                **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_sampler=batch_sampler,
                                                      collate_fn=collate_fn,
                                                      num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // \
                         self.dataset.batch_sizes[d][cid] for d in ['source', 'target']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len