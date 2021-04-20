import copy
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.manipulate import SubDataset, FeatureDataset



def get_dataset(name, type='train', download=True, capacity=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|test]-dataset for chosen dataset.'''

    dataset_class = AVAILABLE_DATASETS[name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


##-------------------------------------------------------------------------------------------------------------------##


def get_experiment(name, tasks=1, data_dir="./store/datasets", normalize=False, augment=False, verbose=False,
                   exception=False, only_config=False, per_class=False):
    '''Load, organize and return train- and test-dataset(s) for requested experiment.'''

    # Define data-type
    if name == "MNIST":
        data_type = 'mnist'
    elif name == "CIFAR10":
        data_type = 'cifar10'
    elif name == "CIFAR100":
        data_type = 'cifar100'
    elif name == "CORe50":
        data_type = 'core50'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[data_type+"_denorm"]
    # check for number of tasks
    if tasks > config['classes']:
        raise ValueError("Experiment '{}' cannot have more than {} tasks!".format(name, config['classes']))
    # -how many classes per epoch?
    if not per_class:
        classes_per_task = int(np.floor(config['classes'] / tasks))
        config['classes'] = classes_per_task * tasks
        config['classes_per_task'] = classes_per_task
    # -if only config-dict is needed, return it
    if only_config:
        return config

    # Prepare permutation to shuffle label-ids (to create different class batches for each random seed)
    classes = config['classes']
    permuted_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))

    # Load train and test datasets with all classes
    if not name in ("CORe50"):
        target_transform = transforms.Lambda(lambda y, p=permuted_class_list: int(p[y]))
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               normalize=normalize, augment=augment, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform,
                              normalize=normalize, augment=augment, verbose=verbose)

    # Split the testset, and possible also the trainset, up into separate datasets for each task/class
    labels_per_task = [[label] for label in range(classes)] if per_class else [
        list(np.array(range(classes_per_task)) + classes_per_task*task_id) for task_id in range(tasks)
    ]
    train_datasets = []
    test_datasets = []
    for labels in labels_per_task:
        if name in ("CORe50"):
            # -training data
            class_datasets = []
            for label in labels:
                class_id = permuted_class_list[label]
                class_ids = list(range(class_id * 5, (class_id + 1) * 5))  #-> for each category, there are 5 objects
                object_datasets = []
                for class_id in class_ids:
                    path_name = os.path.join(data_dir, 'core50_features', 'train_object{}.pt'.format(class_id))
                    feature_tensor = torch.load(path_name)
                    object_datasets.append(FeatureDataset(feature_tensor.view(-1, 512, 1, 1), label))
                class_datasets.append(ConcatDataset(object_datasets))
            task_dataset = ConcatDataset(class_datasets)
            train_datasets.append(task_dataset)
            # -test data
            class_datasets = []
            for label in labels:
                class_id = permuted_class_list[label]
                class_ids = list(range(class_id * 5, (class_id + 1) * 5))  #-> for each category, there are 5 objects
                object_datasets = []
                for class_id in class_ids:
                    path_name = os.path.join(data_dir, 'core50_features', 'test_object{}.pt'.format(class_id))
                    feature_tensor = torch.load(path_name)
                    object_datasets.append(FeatureDataset(feature_tensor.view(-1, 512, 1, 1), label))
                class_datasets.append(ConcatDataset(object_datasets))
            task_dataset = ConcatDataset(class_datasets)
            test_datasets.append(task_dataset)
        else:
            train_datasets.append(SubDataset(trainset, labels, target_transform=None))
            test_datasets.append(SubDataset(testset, labels, target_transform=None))

    # Return tuple of data-sets and config-dictionary
    return ((train_datasets, test_datasets), config)
