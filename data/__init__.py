import glob
import os.path
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment
import data.vl_checklist as vl_checklist


def create_dataset(dataset, config, dataset_pass_dict=None, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'vl-checklist':
        json_files = []
        split_file = config['split_file']
        if os.path.exists(split_file):
            with open(split_file, 'rb') as fp:
                split_dict = pickle.load(fp)
        else:
            split_dict = {'json_files': [], 'image_splits': {}}
        for json_file in config['json_files']:
            if os.path.isfile(json_file):
                json_files.append(json_file)
            else:
                glob_files = glob.glob(json_file, recursive=True)
                if len(glob_files) > 0:
                    json_files.extend(glob_files)
                else:
                    raise ValueError(f'Could not resolve files with: "{json_file}"')
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for json_file in json_files:

            if not json_file in split_dict['json_files']:
                print(f'Generating split for {json_file}')
                save_split = vl_checklist.gen_split_new(json_file, split_dict['image_splits'])
                if save_split:
                    split_dict['json_files'].append(json_file)
                    with open(config['split_file'], 'wb') as fp:
                        pickle.dump(split_dict, fp, pickle.HIGHEST_PROTOCOL)
                vl_checklist.dist.barrier()
                if not save_split:
                    with open(config['split_file'], 'rb') as fp:
                        split_dict = pickle.load(fp)


            train_datasets.append(
                vl_checklist.vl_checklist_dataset(transform_train, json_file, split_dict['image_splits'],
                                                  split='train', config=config,
                                                  dataset_pass_dict=dataset_pass_dict))
            val_datasets.append(
                vl_checklist.vl_checklist_dataset(transform_test, json_file, split_dict['image_splits'], split='val',
                                                  config=config,
                                                  dataset_pass_dict=dataset_pass_dict))

            test_datasets.append(
                vl_checklist.vl_checklist_dataset(transform_test, json_file, split_dict['image_splits'], split='test',
                                                  config=config,
                                                  dataset_pass_dict=dataset_pass_dict))

        return torch.utils.data.ConcatDataset(train_datasets), torch.utils.data.ConcatDataset(val_datasets), \
               torch.utils.data.ConcatDataset(test_datasets)

def create_zsl_dataset(dataset, config, dataset_pass_dict=None, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'vl-checklist':
        json_files = []
        for json_file in config['zsl_json_files']:
            if os.path.isfile(json_file):
                json_files.append(json_file)
            else:
                glob_files = glob.glob(json_file, recursive=True)
                if len(glob_files) > 0:
                    json_files.extend(glob_files)
                else:
                    raise ValueError(f'Could not resolve files with: "{json_file}"')
        train_datasets = []

        for json_file in json_files:
            train_datasets.append(
                vl_checklist.vl_checklist_dataset_zsl(transform_train, json_file, config=config,
                                                  dataset_pass_dict=dataset_pass_dict))


        return torch.utils.data.ConcatDataset(train_datasets),torch.utils.data.ConcatDataset(train_datasets),torch.utils.data.ConcatDataset(train_datasets)

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
