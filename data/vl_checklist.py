import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
import torch.distributed as dist

class vl_checklist_dataset(Dataset):
    def __init__(self, transform, json_file, split_dict: dict, dataset_pass_dict, split='train', config=None):
        '''
        image_root (string): Root directory of images
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        with open(json_file, 'r') as fp:
            self.annotation = json.load(fp)

        self.vg_root = config['vg_root']
        self.haik_root = config['haik_root']
        self.swig_root = config['swig_root']
        self.transform = transform

        self.train_perc = dataset_pass_dict.get('training_data_sample',1)

        # sample training data
        labels_map =  {'train': 0, 'val': 1, 'test': 2}
        label = labels_map[split]
        mod_ann = []
        for ann in self.annotation:
            if split_dict[ann[0]] == label:
                for p,n in zip(ann[1]['POS'], ann[1]['NEG']):
                    mod_ann.append((ann[0], {'POS': p, 'NEG': n}))
        self.annotation = mod_ann

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        if ann[0].startswith('VG'):
            img_root = self.vg_root
        elif os.path.exists(os.path.join(self.haik_root,ann[0])):
            img_root = self.haik_root
        elif os.path.exists(os.path.join(self.swig_root, ann[0])):
            img_root = self.swig_root
        else:
            raise ValueError(f'Could not find file {ann[0]} in any image root!')

        image0_path = os.path.join(img_root, ann[0])
        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0)


        pos_sentence = pre_caption(ann[1]['POS'], 40)
        neg_sentence = pre_caption(ann[1]['NEG'], 40)


        return image0, pos_sentence, neg_sentence, index

class vl_checklist_dataset_zsl(Dataset):
    def __init__(self, transform, json_file, dataset_pass_dict, config=None):
        '''
        image_root (string): Root directory of images
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        with open(json_file, 'r') as fp:
            self.annotation = json.load(fp)

        self.vg_root = config['vg_root']
        self.haik_root = config['haik_root']
        self.swig_root = config['swig_root']
        self.transform = transform

        self.train_perc = dataset_pass_dict.get('training_data_sample',1)

        # sample training data
        mod_ann = []
        for ann in self.annotation:
            for p,n in zip(ann[1]['POS'], ann[1]['NEG']):
                mod_ann.append((ann[0], {'POS': p, 'NEG': n}))
        self.annotation = mod_ann

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        if ann[0].startswith('VG'):
            img_root = self.vg_root
        elif os.path.exists(os.path.join(self.haik_root,ann[0])):
            img_root = self.haik_root
        elif os.path.exists(os.path.join(self.swig_root, ann[0])):
            img_root = self.swig_root
        else:
            raise ValueError(f'Could not find file {ann[0]} in any image root!')

        image0_path = os.path.join(img_root, ann[0])
        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0)


        pos_sentence = pre_caption(ann[1]['POS'], 40)
        neg_sentence = pre_caption(ann[1]['NEG'], 40)


        return image0, pos_sentence, neg_sentence, index

def gen_split_new(json_file, split_dict, split=(.8,.1,.1), seed=0):
    to_save = False
    if dist.get_rank() == 0:
        rng = random.Random(seed)
        with open(json_file, 'r') as fp:
            json_data = json.load(fp)
        for d in json_data:
            if d[0] not in split_dict.keys():
                to_save = True
                r = rng.random()
                if r<split[0]:
                    s = 0
                elif r < split[0]+split[1]:
                    s = 1
                else:
                    s = 2
                split_dict[d[0]] = s
    dist.barrier()
    return to_save

