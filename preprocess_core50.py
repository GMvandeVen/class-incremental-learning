#!/usr/bin/env python3
import numpy as np
import os
import urllib.request
import zipfile
import torch
import argparse
from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.models import resnet18


## NOTE: The loading of the COR50e-dataset is based on: https://github.com/Continvvm/continuum (retrieved 18 Dec 2020)


####################################################################################################################

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, url.split("/")[-1])

    if os.path.exists(file_name):
        print("Dataset already downloaded at {}.".format(file_name))
    else:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Wget/1.20.3 (linux-gnu)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, file_name, ProgressBar().update)

    return file_name


def unzip(path):
    directory_path = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(directory_path)


class ProgressBar:
    '''Basic progress bar, based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console'''

    def __init__(self):
        self.count = 0

    def update(self, tmp, block_size, total_size):
        self.count += block_size

        percent = "{}".format(int(100 * self.count / total_size))
        filled_length = int(100 * self.count // total_size)
        pbar = "#" * filled_length + '-' * (100 - filled_length)

        print("\r|%s| %s%%" % (pbar, percent), end="\r")
        if self.count == total_size:
            print()


class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=label_squeezing_collate_fn, drop_last=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )


class COR50eDataset(Dataset):
    '''Create dataset from <np.array> with paths to images saved on disk.'''

    def __init__(self, x, y):
        self._x = x    # <np.array> with path to images
        self._y = y    # <np.array> with labels

    def __len__(self):
        return self._x.shape[0]

    def get_sample(self, index):
        '''Returns a Pillow image corresponding to the given `index`.'''
        x = self._x[index]
        x = Image.open(x).convert("RGB")
        return x

    def __getitem__(self, index):
        '''Method used by PyTorch's DataLoaders to query a sample and its target.'''
        # -get image & label
        img = self.get_sample(index)
        y = self._y[index]
        # -transform image to tensor
        img = transforms.ToTensor()(img)
        # -return the tuple
        return img, y

####################################################################################################################


## Function for specifying input options
def handle_inputs():
    filename = 'preprocess_core50.py'
    description = 'Download and preprocess (with ResNet18 pretrained on ImageNet) the CORe50 dataset.'
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--batch', type=int, default=512, help="batch-size for pre-processing")
    args = parser.parse_args()
    return args


## Function for downloading and pre-processing the CORe50 dataset
def download_and_preprocess(data_dir='./store/datasets', batch_size=512):

    # Create folders, if necessary
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(os.path.join(data_dir, 'core50_features')):
        os.mkdir(os.path.join(data_dir, 'core50_features'))

    # Location where data and CSV file with IDs of training samples can be downloaded
    data_url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    train_ids_url = "https://vlomonaco.github.io/core50/data/core50_train.csv"

    # Download the data
    if os.path.exists(os.path.join(data_dir, "core50_128x128")):
        print("CORe50-dataset already downloaded.")
    else:
        print("Downloading the CORe50-dataset...")
        path = download(data_url, data_dir)
        print("Unzipping the downloaded dataset...")
        unzip(path)

    # Download the CSV file containing the IDs of the training vs. test samples
    split_path = os.path.join(data_dir, "core50_train.csv")
    if os.path.exists(split_path):
        train_image_ids_path = split_path
        print("CSV-file with default train/test split already downloaded.")
    else:
        print("Downloading CSV-file with default train/test split...")
        train_image_ids_path = download(train_ids_url, data_dir)

    # Read the CSV file containing the IDs of the training vs. test samples
    train_image_ids = set()
    with open(train_image_ids_path, "r") as f:
        for line in f:
            image_id = line.split(",")[0].split(".")[0]
            train_image_ids.add(image_id)

    # Prepare for collecting the paths (and corresponding labels) of the images
    x_train, y_train = [], []
    x_test, y_test = [], []

    # Cycle through the 10 domains:
    for domain_id in range(10):
        # -folder of this domain
        domain_folder = os.path.join(data_dir, "core50_128x128", "s{}".format(domain_id + 1))

        # In each domain, there are 50 different objects. Cycle through them:
        for object_id in range(50):
            # -folder for this object
            object_folder = os.path.join(domain_folder, "o{}".format(object_id + 1))

            # Cycle through all images of the current object in the current domain:
            for path in os.listdir(object_folder):
                # -get ID of image (needed to check whether in train- or test-set)
                image_id = path.split(".")[0]
                # -add the path to this images to train- or test-set
                in_train_set = (image_id in train_image_ids)
                if in_train_set:
                    x_train.append(os.path.join(object_folder, path))
                    y_train.append(object_id)
                else:
                    x_test.append(os.path.join(object_folder, path))
                    y_test.append(object_id)

    # Convert to numpy-arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Create a separate train- and test-dataset for each object
    train_datasets = []
    test_datasets = []
    for object_id in range(50):
        # -training data
        selection = np.isin(y_train, [object_id])
        dataset = COR50eDataset(x=x_train[selection], y=y_train[selection])
        train_datasets.append(dataset)
        # -test data
        selection = np.isin(y_test, [object_id])
        dataset = COR50eDataset(x=x_test[selection], y=y_test[selection])
        test_datasets.append(dataset)

    # Use cuda for the pre-processing?
    cuda = torch.cuda.is_available()# and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Specify and load pretrained model to extract the features with
    print("Loading ResNet18-model, pretrained on ImageNet...")
    feature_extractor = resnet18(pretrained=True)
    feature_extractor.fc = Identity()      #-> remove final layer

    # Move feature extractor to correct device and set to evaluation-mode
    feature_extractor.to(device)
    feature_extractor.eval()
    num_features = 512

    # Cylce through all 50 objects
    for object_id in range(50):
        print("Extracting features for object '{}'...".format(object_id))

        # -training data
        loader = get_data_loader(train_datasets[object_id], batch_size=batch_size, drop_last=False, cuda=cuda)
        all_features = torch.empty((len(loader.dataset), num_features)) #-> allocate tensor, to be filled slice-by-slice
        count = 0
        for x, _ in loader:
            with torch.no_grad():
                x = feature_extractor(x.to(device)).cpu()
            all_features[count:(count + x.shape[0])] = x.view(-1, num_features)
            count += x.shape[0]
        # -store tensor with all extrated features for this object's training data
        path_name = os.path.join(data_dir, 'core50_features', 'train_object{}.pt'.format(object_id))
        torch.save(all_features, path_name)

        # -testing data
        loader = get_data_loader(test_datasets[object_id], batch_size=batch_size, drop_last=False, cuda=cuda)
        all_features = torch.empty((len(loader.dataset), num_features))
        count = 0
        for x, _ in loader:
            with torch.no_grad():
                x = feature_extractor(x.to(device)).cpu()
            all_features[count:(count + x.shape[0])] = x.view(-1, num_features)
            count += x.shape[0]
        # -store tensor with all extrated features for this object's test data
        path_name = os.path.join(data_dir, 'core50_features', 'test_object{}.pt'.format(object_id))
        torch.save(all_features, path_name)



if __name__ == '__main__':
    args = handle_inputs()
    download_and_preprocess(data_dir=args.d_dir, batch_size=args.batch)
