import glob
import os
import time

import kaggle
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

# np.random.seed(0)

### CUDA
use_cuda = torch.cuda.is_available()
devices = [torch.device("cuda:0" if use_cuda else "cpu")]
device0 = devices[0]
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch_type = "torch.cuda"
    map_location = lambda storage, loc: storage.cuda()
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    torch_type = "torch"
    map_location = 'cpu'


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def savefig(fname, dpi=None):
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi, format='png')


def inf_train_gen(trainloader):
    while True:
        for images, targets in trainloader:
            yield (images, targets)


def generate_sample(dataset, iter, netG, fix_noise, save_dir, device, num_classes=10, img_w=28, img_h=28):
    if dataset in ['mnist', 'fashionmnist']:
        generate_image(iter, netG, fix_noise, save_dir, device, num_classes, img_w, img_h)
    elif dataset in ['ptb', 'mitbih']:
        generate_time_sample(iter, netG, fix_noise, save_dir, device, num_classes, img_w=img_w, img_h=img_h)
    else:
        print(F"SAMPLE GENERATION FOR {dataset} NOT SUPPORTED")


def generate_time_sample(iter, netG, fix_noise, save_dir, device, num_classes, img_w=28, img_h=28):
    batchsize = fix_noise.size()[0]
    nrows = 3
    ncols = num_classes * batchsize
    figsize = (1.2 * ncols, 1.2 * nrows)

    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    sample_list = []
    z_dim = 10
    for i in range(nrows):
        fix_noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim)
        noise = fix_noise.to(device)
        batchsize = fix_noise.size()[0]

        for j in range(num_classes):
            class_id = j
            label = torch.full((num_classes,), class_id).to(device)
            sample = netG(noise, label)
            sample = sample.view(batchsize, img_w * img_h, 1)
            sample = sample.cpu().data.numpy()
            sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w * img_h, 1])

    fig = plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.plot(np.arange(0, img_w * img_h), samples[i])
        plt.axis('off')
        for xpos in np.arange(0, img_w * img_h, img_w):
            plt.axvline(x=xpos, linewidth=0.2, color='gray')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))
    plt.show()
    plt.close()

    del label, noise, sample
    torch.cuda.empty_cache()


def generate_image(iter, netG, fix_noise, save_dir, device, num_classes=10,
                   img_w=28, img_h=28):
    batchsize = fix_noise.size()[0]
    nrows = num_classes  # TODO set back to 10 for mnist?
    ncols = num_classes
    figsize = (ncols, nrows)
    noise = fix_noise.to(device)

    sample_list = []
    for class_id in range(num_classes):
        label = torch.full((nrows,), class_id).to(device)
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w, img_h])

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))
    plt.close()

    del label, noise, sample
    torch.cuda.empty_cache()


def get_device_id(id, num_discriminators, num_gpus):
    partitions = np.linspace(0, 1, num_gpus, endpoint=False)[1:]
    device_id = 0
    for p in partitions:
        if id <= num_discriminators * p:
            break
        device_id += 1
    return device_id


def to_trainset(samples, labels=None):
    if labels is None:
        labels = np.zeros(len(samples)).astype('int64')
    dataset = torch.utils.data.TensorDataset(torch.tensor(samples, dtype=torch.float),
                                             torch.tensor(labels, dtype=torch.int64))
    return dataset, np.bincount(labels)


def load_digits_mnist(data_root, transform_train):
    dataloader = datasets.MNIST
    trainset = dataloader(root=os.path.join(data_root, 'MNIST'), train=True, download=True,
                          transform=transform_train)
    return trainset


def load_fashion_mnist(data_root, transform_train):
    dataloader = datasets.FashionMNIST
    trainset = dataloader(root=os.path.join(data_root, 'FashionMNIST'), train=True, download=True,
                          transform=transform_train)
    return trainset


def load_ecg_heartbeat(data_root, dataset, train=True, num_samples=None, pad=True, plot=False):
    samples, labels = load_ecg_heartbeat_samples_labels(data_root, dataset, train, num_samples, pad)

    if plot:
        plt.figure(figsize=(5, 5))
        for i in range(0, 25):
            plt.subplot(5, 5, i + 1)
            plt.plot(np.arange(0, len(samples[0])), samples[i])
            plt.axis('off')

        plt.show()
        plt.close()

        plot_dataset(samples, labels, imshow=False)

    return samples, labels


def pad_data(data):
    sample_len = len(data[0])
    pad_length = int(np.ceil(np.sqrt(sample_len)) ** 2 - sample_len)
    pad = np.zeros((len(data), pad_length))
    data = np.concatenate((data, pad), axis=1)
    return data


def load_ecg_heartbeat_samples_labels(data_root, dataset, train=True, num_samples=None, pad=True):
    # if pad is true pads the samples to a the smallest square possible for
    download_dir = os.path.join(data_root, "ecg")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shayanfazeli/heartbeat', path=download_dir, unzip=True)

    if dataset == 'mitbih':
        filenames = ['mitbih_train.csv'] if train else ['mitbih_test.csv']
    elif dataset == 'ptb':
        filenames = ['ptbdb_abnormal.csv', 'ptbdb_normal.csv']
    else:
        raise ValueError(f"Dataset {dataset} not part of kaggle shayanfazeli/heartbeat")

    subsets = []
    for file in filenames:
        subsets.append(pd.read_csv(os.path.join(download_dir, file), delimiter=',').to_numpy(dtype=np.float))
    data = np.concatenate(subsets)

    np.random.shuffle(data)

    if dataset == 'ptb':
        # split into train and test 80/20
        split = int(0.8 * len(data))
        train_samples = data[:split, :]
        test_samples = data[split:, :]

        data = train_samples if train else test_samples

    if num_samples and len(data) > num_samples:
        data = data[:num_samples]

    samples = data[:, :-1]
    labels = data[:, -1:].astype('int64')

    if pad:
        samples = pad_data(samples)

    return samples, labels.ravel()


def load_dataset(dataset, data_root, save_dir=False):
    transform_train = transforms.ToTensor()

    samples, labels = [], []
    if dataset == 'mnist':
        num_classes, width, height = 10, 28, 28
        classes_split = np.ones(10) * 6000
        dataset = load_digits_mnist(data_root, transform_train)
    elif dataset == 'fashionmnist':
        num_classes, width, height = 10, 28, 28
        classes_split = np.ones(10) * 6000
        dataset = load_fashion_mnist(data_root, transform_train)
    elif dataset == 'mitbih':
        samples, labels = load_ecg_heartbeat(data_root, dataset, pad=True)
        dataset, classes_split = to_trainset(samples, labels)
        num_classes, width, height = len(classes_split), 14, 14
    elif dataset == 'ptb':
        samples, labels = load_ecg_heartbeat(data_root, dataset, pad=True)
        dataset, classes_split = to_trainset(samples, labels)
        num_classes, width, height = len(classes_split), 14, 14
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    if save_dir:
        np.savez(f"{save_dir}/samples_real.npz", samples=samples, labels=labels)

    return dataset, num_classes, width, height, classes_split


def plot_samples(samples, labels, save_file=None, show=False, imshow=True):
    nrows = int(np.sqrt(len(samples)))
    ncols = nrows
    plt.figure(figsize=(nrows, ncols))

    if imshow and len(samples.shape) < 4:
        width = int(np.sqrt(samples.shape[1]))
        samples = np.reshape(samples, (-1, width, width, 1))

    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)

        if imshow:
            plt.imshow(samples[i], cmap='gray')
        else:
            plt.plot(np.arange(0, len(samples[i])), samples[i])
        plt.axis('off')

    if save_file:
        savefig(save_file)

    if show:
        plt.show()

    plt.close()


def plot_dataset(samples, labels, imshow=False):
    uniq_labels = np.unique(labels)
    uniq_labels.sort()

    to_plot = []
    plot_labels = []
    for label in uniq_labels:
        j = 0
        for i, sample in enumerate(samples):
            if labels[i] == label:
                j += 1
                to_plot.append(sample)
                plot_labels.append(label)

            if j == len(uniq_labels):
                break

    to_plot = np.array(to_plot)
    plot_samples(to_plot, labels, show=True, imshow=imshow)

def generate_samples(generator, n, max_samples_per_iter=10, device=device0,
                     img_w=28, img_h=28, shuffle=True,
                     flatten=False, classes_split=None):
    """
    Generates n images using the generator
     and outputs them into a .mat file with tensor if write_mat_locatino is provided
     flattens into size [n, 784] (784 = flattened MNIST image dimension) if flatten is true
    TODO fix the number of samples that are generated, does not work with e.g. n=1280
    """
    print(f"\tgenerating {n} samples using the given generator")
    start_time = time.time()

    num_classes = generator.num_classes
    if classes_split is None:
        classes_split = np.ones(num_classes) * (n / num_classes)

    z_dim = generator.z_dim
    samples_per_iter = max_samples_per_iter

    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    fix_noise = bernoulli.sample((samples_per_iter, z_dim)).view(samples_per_iter, z_dim)
    noise = fix_noise.to(device)
    batchsize = fix_noise.size()[0]

    samples = []
    labels = []
    for class_label in range(num_classes):
        print(f"\rgenerating images for label {class_label + 1} / {num_classes}", end="")
        n_remaining = classes_split[class_label]
        # split into blocks of x samples to fit into memory
        while n_remaining > 0:
            n_remaining -= samples_per_iter
            # generate n_class samples with label: class_label
            label = torch.full((samples_per_iter,), class_label).to(device)
            sample = generator(noise, label)
            if not flatten:
                sample = sample.view(batchsize, img_w, img_h)

            sample = sample.cpu().data.numpy()
            samples.extend([x for x in sample])

            label = label.cpu().data.numpy()
            labels.extend(label)

    samples = np.array(samples)
    labels = np.array(labels)

    if shuffle:
        indices = np.arange(samples.shape[0])
        np.random.shuffle(indices)

        samples = samples[indices]
        labels = labels[indices]

    print(f"\tSample generation took {round(time.time() - start_time)} seconds")
    return samples, labels


def train_classifier_mitbih(train_samples, train_labels, test_samples, test_labels):
    print(f"\t\t Calculating downstream classifier accuracies, "
          f"{len(train_samples)} training samples, {len(test_samples)} test samples")

    # From https://github.com/astorfi/differentially-private-cgan
    sc = MinMaxScaler()

    X_train = sc.fit_transform(train_samples)
    X_test = sc.transform(test_samples)

    n_estimator = 100
    if len(np.unique(train_labels)) > 2 or len(np.unique(test_labels)) > 2:
        # make binary for mitbih
        print("making labels binary")
        train_labels[train_labels != 0] = 1
        test_labels[test_labels != 0] = 1
    cls = GradientBoostingClassifier(n_estimators=n_estimator)

    cls.fit(X_train, train_labels)
    y_pred = cls.predict(X_test)
    score = metrics.accuracy_score(test_labels, y_pred)

    y_pred = cls.predict_proba(X_test)[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(test_labels, y_pred)
    AUROC = metrics.auc(fpr_rf_lm, tpr_rf_lm)

    precision, recall, thresholds = metrics.precision_recall_curve(test_labels, y_pred)
    AUPRC = metrics.auc(recall, precision)

    y_pred_labels = cls.predict(X_test)
    f1 = metrics.f1_score(test_labels, y_pred_labels)
    print(metrics.confusion_matrix(test_labels, y_pred_labels))
    print({"accuracy": score, "AUROC": AUROC, "AUPRC": AUPRC, "F1": f1})
    return {"accuracy": score, "AUROC": AUROC, "AUPRC": AUPRC, "F1": f1}


def train_mitbih_classifier_baseline():
    print("training mitbih")
    train_samples, train_labels = load_ecg_heartbeat_samples_labels('../../resources/data', dataset='mitbih',
                                                                    train=True)
    test_samples, test_labels = load_ecg_heartbeat_samples_labels('../../resources/data', dataset='mitbih', train=False)
    train_classifier_mitbih(train_samples, train_labels, test_samples, test_labels)


def train_ptb_classifier_baseline():
    print("training ptb")
    train_samples, train_labels = load_ecg_heartbeat_samples_labels('../../resources/data', dataset='ptb', train=True)
    test_samples, test_labels = load_ecg_heartbeat_samples_labels('../../resources/data', dataset='ptb', train=False)
    train_classifier_mitbih(train_samples, train_labels, test_samples, test_labels)

