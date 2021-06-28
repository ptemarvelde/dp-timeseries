import argparse
import os
import pprint
import sys
import time

import tensorflow as tf
import torchvision.datasets as datasets
from scipy.io import savemat
import matplotlib.pyplot as plt

datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]

if os.path.exists("/resources/"):
    RESOURCES_ROOT = "/resources/"
else:
    RESOURCES_ROOT = "../../resources/"

MNIST_MODULE = 'https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1'
FASHION_MNIST_PATH = RESOURCES_ROOT + "eval_models/own_inception_fashionmnist/"
MNIST_PATH = RESOURCES_ROOT + "eval_models/own_inception_mnist/"

sys.path.append("../../GS-WGAN-main/source")

from fid import get_fid  # comment to avoid loading all activations
from models import *
from util import mnist_score
from utils import generate_image, load_ecg_heartbeat_samples_labels
# from Base_DP_CGAN import sample_Z, generator, xavier_init
from evaluation_networks import model_from_checkpoint, \
    get_accuracies, get_accuracies_ecg
from MLP_for_Inception import inception

# set tensorflow logging level
tf.get_logger().setLevel('ERROR')

EVALUATE_ITER = 2000

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


def get_value(tensor):
    """
    Gets the real value of a tensor regardless of whether eager execution is enabled.
    """
    try:
        res = tensor.eval()
    except NotImplementedError:  # happens when eager execution is enabled
        try:
            res = tensor.numpy()
        except:
            print("Inception score calc failing since inception_score.numpy() and inception_score.eval() both fail")
            res = None
    return res


def to_tensor(data):
    return tf.constant(data.reshape((*data.shape, 1)))


def get_classifier_for_inception_score(dataset, classifier_path=None):
    classifier = None
    if dataset == 'mnist':
        classifier_path = MNIST_PATH if not classifier_path else classifier_path
    elif dataset == 'fashionmnist':
        classifier_path = FASHION_MNIST_PATH if not classifier_path else classifier_path
    else:
        raise ValueError(f"{dataset} not a supported dataset for inception score")

    print(f"getting classifier from {classifier_path}")
    classifier = model_from_checkpoint(classifier_path, inception()) if not classifier else classifier

    return classifier


def calc_inception_score(samples, dataset, num_batches=10, classifier=None):
    # since giving a num_batches > 1 to the mnist_score function does not give standard deviation the batches
    # are handled manually here.
    print(f"\t\tCalculating inception score in {num_batches} batches for {len(samples)} samples")
    start_time = time.time()

    # shuffle samples
    np.random.shuffle(samples)
    # create tensor of images and reshape for compatibality
    image_tensor = to_tensor(samples)
    scores = []
    batch_start_time = 0
    if not classifier or type(classifier) == str:
        classifier = get_classifier_for_inception_score(dataset, classifier_path=classifier)

    for i, batch in enumerate(tf.split(image_tensor, num_batches)):
        print(
            f"\r\t\tCalculating score for batch {i + 1} / {num_batches}, last batch took {round(time.time() - batch_start_time)} seconds",
            end="")
        batch_start_time = time.time()
        inception_score = mnist_score(batch, classifier, dataset)

        scores.append(get_value(inception_score))
        print(f"{scores[-1]}", end='')

    print(f"\n\t\tInception score calculation took {round(time.time() - start_time)} seconds")
    del classifier
    return np.mean(scores), np.std(scores)


def to_rgb_and_reshape(imgs):
    tensor = to_tensor(imgs)
    rgb = tf.image.grayscale_to_rgb(tensor)

    rgb = get_value(rgb)
    rgb = rgb.transpose([0, 3, 1, 2])
    return rgb


def calc_frechet_distance_inception(real_data, samples, num_batches):
    assert len(real_data) == len(samples), "inputs not of equal lengths"
    assert len(real_data) % num_batches == 0, "data input length should be multiple of num_batches"

    print(
        f"\n\t\tCalculating frechet inception distance using inceptionV1 in {num_batches} batches for {len(samples)} samples")
    start_time = time.time()

    real_arr = to_rgb_and_reshape(real_data) * 255
    gen_arr = to_rgb_and_reshape(samples) * 255

    fid = []
    batch_start_time = 0
    for i, batch in enumerate(np.array_split(list(zip(real_arr, gen_arr)), num_batches)):
        print(
            f"\r\t\tCalculating score for batch {i + 1}, last batch took {round(time.time() - batch_start_time)} seconds",
            end="")
        batch_start_time = time.time()
        real = batch[:, 0, :, :].astype('uint8')
        gen = batch[:, 1, :, :].astype('uint8')
        fid.append(get_fid(real, gen))

    print(f"\n\t\tCalculation took {round(time.time() - start_time)} seconds")
    return np.mean(fid), np.std(fid)


def quantitative_evaluation(gen_samples, gen_labels, real_samples, real_labels, dataset, gen_data_size=10000,
                            num_batches=10,
                            metric_dict=None,
                            inception_classifier=None):
    """
    @param generator Generator
    @param gen_data_size size of the to be generated (and evaluated) dataset.
    @param real_data actual samples used during evaluation


    Takes as input a generator and outputs the following metrics:
    --Sample quality--
    Inception Score https://github.com/ChunyuanLI/MNIST_Inception_Score
    Frechet inception distance

    --downstream tasks--
    Multi Layer Perceptron accuracy
    Convolutional Neural Network accuracy
    Average accuracy of all downstream classifiers
    Average accuracy of all downstream classifiers normalized by accuracy when trained on real data
    """
    if metric_dict is None:
        metric_dict = {}
    print(f"\tStarting metric calculation")
    results = {}

    if metric_dict.get('IS') is not False:
        inception_score = calc_inception_score(gen_samples[:gen_data_size], dataset, num_batches, inception_classifier)
        print(f"\tinception score: {inception_score}")
        results['Inception Score'] = inception_score

    if metric_dict.get('FID') is not False:
        frechet_distance = calc_frechet_distance_inception(real_samples[:gen_data_size], gen_samples[:gen_data_size],
                                                           num_batches=num_batches)
        print(f"\tfrechet distance: {frechet_distance}")
        results['Frechet Inception Distance'] = frechet_distance

    if metric_dict.get('classifier_acc') is not False:
        if dataset == 'ptb':
            accuracies = get_accuracies_ecg(gen_samples, gen_labels, real_samples, real_labels)
        else:
            accuracies = get_accuracies(gen_samples, gen_labels, real_samples, real_labels)
        print(f"\tdownstream classifier accuracies: {accuracies}")
        results['Downstream Classifier Accuracy'] = accuracies

    print(f"\n\n\tquantitative eval results: {results}")

    del gen_labels, gen_samples
    return results


class Evaluator:
    """
    Class used to calculate quantitative metrics during training of GS-WGAN
    """

    def __init__(self, gen_data_size,
                 dataset,
                 save_location=None,
                 generator_path=None,
                 generator_dir=None,
                 generator=None,
                 num_batches=10,
                 random_seed=1,
                 args=None):

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if (not generator_path and not generator_dir) or (generator_path and generator_dir):
            raise ValueError("Please provide a generator_path OR generator_dir")

        if generator_path:
            if generator_path[-4:] == ".pth":
                self.generator = load_generator_gswgan(generator_path)
                self.generator_dir = None
            else:
                raise ValueError(f"generator_path should point to a '.pth' file but was {generator_path}")
        else:
            # generator dir
            self.generator = None
            self.generator_dir = generator_dir

        if not gen_data_size % num_batches == 0:
            # raise gen_data_size to be a multiple of num_batches
            gen_data_size = gen_data_size + (num_batches - gen_data_size % num_batches)
            print("gen_data_size not a multiple of num_batches, setting gen_data_size to {gen_data_size}")

        if generator_path:
            self.gen_samples, self.gen_labels = get_data(generator_path, gen_data_size)

        self.dataset = dataset
        self.gen_data_size = gen_data_size
        self.save_location = save_location
        self.real_data = None
        self.real_labels = None
        self.num_batches = num_batches
        if args is None:
            self.metric_dict = {"IS": True, "FID": True, "classifier_acc": True}
        else:
            self.metric_dict = {"IS": not args.skip_IS,
                                "FID": not args.skip_FID,
                                "classifier_acc": not args.skip_classifier_acc
                                }

    def evaluate(self, generator=None, iteration=None):
        if generator:
            self.generator = generator

        if self.real_data is None:
            self.real_data, self.real_labels = load_images(self.dataset)[:self.gen_data_size]

        if self.generator_dir:
            res = evaluate_multiple(self.generator_dir, self.real_data, self.real_labels, self.dataset,
                                    self.gen_data_size,
                                    self.num_batches, self.metric_dict)
        elif self.generator:
            res = quantitative_evaluation(self.gen_samples, self.gen_labels, self.real_data, self.real_labels
                                          , self.dataset, self.gen_data_size,
                                          self.num_batches, metric_dict=self.metric_dict)
        else:
            raise ValueError("no generator_dir and no self.generator")

        if self.save_location:
            os.makedirs(self.save_location, exist_ok=True)
            save_file = os.path.join(self.save_location, "results.txt")
            with open(save_file, 'a') as f:
                if iteration:
                    f.write(f"{iteration}, {str(res)}\n")
                else:
                    f.write(pprint.pformat(res))
        return res


def show_images(generator_dir):
    """
    Takes as input a generator file and outputs a figure with generated images.
    """
    z_dim = 10
    model_dim = 64
    gen = GeneratorResNet(z_dim=10, model_dim=64, num_classes=10, torch_type=torch_type)
    gen.load_state_dict(torch.load(generator_dir, map_location=map_location))
    gen.to(device0)

    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)

    generate_image(iter="final", netG=gen, fix_noise=fix_noise, save_dir=None, device=device0)


def plot_images(samples):
    """
    Takes samples and shows 10 from every len(samples)/10 samples
    ie shows 10 samples for each label (assumes samples are sorted)
    """
    samples = [samples[x] for x in range(len(samples)) if x % (len(samples) / 10) < 10]

    nrows = ncols = int(np.sqrt(len(samples)))
    plt.figure(figsize=(ncols, nrows))
    for i in range(nrows * ncols):
        sample = np.reshape(samples[i], (28, 28))
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(sample, cmap='gray')
        plt.axis('off')

    try:
        plt.show()
    except:
        print("cannot show plot")

    plt.close()


def load_images(dataset, save_path='../data/samples/', train=False):
    """
    Load real mnist or fashionmnist data
    @param dataset should be 'mnist' or 'fashionmnist'
    @param save_path path to save dataset to
    @param train
    """
    if dataset == 'mnist':
        dataloader = datasets.MNIST
    elif dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
    elif dataset == 'ptb':
        samples, labels = load_ecg_heartbeat_samples_labels(save_path, 'ptb', train=False, pad=False)
    else:
        raise ValueError("dataset must be 'mnist' or 'fashionmnist' but was ", dataset)

    if dataset in ['mnist', 'fashionmnist']:
        trainset = dataloader(root=os.path.join(save_path, dataset), train=train, download=True)
        samples, labels = trainset.data.numpy().astype('float32'), trainset.targets.numpy()

    return samples, labels


def generate_samples(generator, n, max_samples_per_iter=100, device=device0,
                     img_w=28, img_h=28, shuffle=True, write_mat_location=None,
                     flatten=False, plot=True):
    """
    Generates n images using the generator
     and outputs them into a .mat file with tensor if write_mat_locatino is provided
     flattens into size [n, 784] (784 = flattened MNIST image dimension) if flatten is true
    TODO fix the number of samples that are generated, does not work with e.g. n=1280
    """
    print(f"\tgenerating {n} samples using the given generator")
    start_time = time.time()

    num_classes = generator.num_classes
    n_class = int(n / num_classes)
    z_dim = generator.z_dim
    samples_per_iter = max_samples_per_iter if n_class > max_samples_per_iter else n_class

    p = 0.5

    samples = []
    labels = []
    for class_label in range(num_classes):
        print(f"\rgenerating images for label {class_label + 1} / {num_classes}", end="")
        n_remaining = n_class
        # split into blocks of x samples to fit into memory
        while n_remaining > 0:
            bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
            fix_noise = bernoulli.sample((samples_per_iter, z_dim)).view(samples_per_iter, z_dim)
            noise = fix_noise.to(device)
            batchsize = fix_noise.size()[0]

            n_remaining -= samples_per_iter
            # generate n_class samples with label: class_label
            label = torch.full((samples_per_iter,), class_label).to(device)
            sample = generator(noise, label)
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

    if plot:
        plot_images(samples)

    if flatten:
        samples = np.array([x.flatten() for x in samples])

    if write_mat_location:
        mdic = {"images": samples, "labels": labels}
        savemat(write_mat_location, mdic)

    print(f"\tSample generation took {round(time.time() - start_time)} seconds")
    return samples, labels


def load_generator_gswgan(generator_dir, z_dim=10, model_dim=64, num_classes=10, samples_dim=(28, 28)):
    gen = GeneratorResNet(z_dim=z_dim, model_dim=model_dim, num_classes=num_classes, torch_type=torch_type,
                          samples_dim=samples_dim)
    gen.load_state_dict(torch.load(generator_dir, map_location=map_location))
    gen.to(device0)
    return gen


def get_data_sources(adir):
    """
    Get list of all files containing data from @param dir
    i.e. get all files ending with .npz or starting with netGS
    return list of these files, type (either 'npz' or 'pth')
    """
    generators = [file for file in os.listdir(adir) if file.startswith("netGS")]
    numpy_files = [file for file in os.listdir(adir) if file.endswith(".npz")]

    if len(generators) > 0 and len(numpy_files) > 0:
        raise ValueError(f"{adir} contains both files starting with netGS and "
                         f"ending with .npz, should only contain one of the two")

    res = generators if len(generators) > 0 else numpy_files
    return res


def shuffle(samples, labels):
    """
    Randomly shuffle samples and labels the same way.
    """
    indices = np.arange(samples.shape[0])
    np.random.shuffle(indices)

    return samples[indices], labels[indices]


def get_data(path: str, num_samples, flatten=False, width=28, num_classes=10):
    """
    Retrieve samples from generator (generate samples) or from .npz (load)
    """
    atype = path[-3:]
    if atype == "pth":
        generator = load_generator_gswgan(os.path.join(path), num_classes=num_classes, samples_dim=(width, width))
        samples, labels = generate_samples(generator, num_samples,
                                           flatten=flatten, shuffle=True, plot=False, img_w=width, img_h=width)
    elif atype == "npz":
        data = np.load(path)
        samples = data['data']
        labels = data['labels']

        # transform labels from (n, 10) to (n, 1) array
        labels = np.where(labels == 1)[1]

        if not flatten:
            samples = np.reshape(samples, (-1, width, width))
    else:
        raise ValueError(f"Type {atype} not recognized")

    samples, labels = shuffle(samples, labels)
    return samples[:num_samples], labels[:num_samples]


def get_iter_from_data_source(filename: str):
    """
    Gets the iteration as int from filename of format netGS_iter.pth or dp-cgan...._iter.npz
    """
    underscore_index = filename.find("_")
    try:
        iteration = int(filename[underscore_index + 1:-4]) if underscore_index != -1 else -1
    except:
        iteration = -1
    return iteration


def evaluate_multiple(generator_dir, real_data, real_labels, dataset, gen_data_size, num_batches, metric_dict):
    """
    Runs the quantitative evaluation with all generators in a directory
    generator file names should be of format: 'netGS_{iter}.pth' or '...._{iter}.npz'
    """
    result_list = []
    generators = get_data_sources(generator_dir)

    iterations = [get_iter_from_data_source(gen) for gen in generators]

    zipped = sorted(zip(iterations, generators), key=lambda k: k[0])

    to_calc_list = []
    print(zipped)
    for iteration, gen in zipped:
        if iteration % 2000 != 0 and not iteration == -1:
            print(f"skipping {gen}, iter {iteration}")
        else:
            to_calc_list.append([iteration, gen])

    # for dpcgan .npz's find latest and fix iteration
    if to_calc_list[0][0] == -1:
        to_calc_list[0][0] = to_calc_list[-1][0] + (to_calc_list[-1][0] - to_calc_list[-2][0])

    to_calc_list = sorted(to_calc_list, key=lambda k: k[0])
    to_calc_list.reverse()
    n_gens = len(to_calc_list)
    # load inception classifier
    inception_classifier = get_classifier_for_inception_score(dataset) if metric_dict["IS"] else None

    for i, (iteration, gen) in enumerate(to_calc_list):
        path = os.path.join(generator_dir, gen)
        print(f"Starting generator evaluation for generator {gen}, {i + 1}/{n_gens}")
        start_time = time.time()
        gen_samples, gen_labels = get_data(path, gen_data_size,
                                           flatten=False, width=28, num_classes=10
                                           )

        if dataset in ['ptb', 'mitbih']:
            gen_samples = gen_samples[:, :187]

        result = quantitative_evaluation(gen_samples, gen_labels, real_data, real_labels,
                                         dataset, gen_data_size=gen_data_size,
                                         num_batches=num_batches, metric_dict=metric_dict,
                                         inception_classifier=inception_classifier)

        print(f"Finished generator evaluation for generator {gen}, {i + 1}/{n_gens} "
              f"in {round(time.time() - start_time)} seconds")
        result_list.append([iteration, result])

    print(result_list)
    return np.array(result_list)


def evaluate_single_generator(gen_path, gen_data_size, dataset, metric_dict=None):
    gen_samples, gen_labels = get_data(gen_path, gen_data_size)
    real_samples, real_labels = load_images(dataset, train=False)

    inception_classifier = get_classifier_for_inception_score(dataset)
    num_batches = 5
    result = quantitative_evaluation(gen_samples, gen_labels, real_samples, real_labels,
                                     dataset, gen_data_size=gen_data_size,
                                     num_batches=num_batches, metric_dict=metric_dict,
                                     inception_classifier=inception_classifier)
    print(result)
    return result


def main(args):
    generator_path = args.generator_path
    generator_dir = args.generator_dir
    gen_data_size = args.gen_data_size
    dataset = args.dataset
    save_location = args.save_location
    num_batches = args.num_batches
    EVALUATE_ITER = args.eval_iter

    evaluator = Evaluator(gen_data_size=gen_data_size,
                          dataset=dataset,
                          generator_path=generator_path,
                          generator_dir=generator_dir,
                          save_location=save_location,
                          num_batches=num_batches,
                          args=args)

    evaluator.evaluate()


def evaluate_real_data():
    dataset = 'mnist'
    gensize = 10000
    real, labels = load_images(dataset, train=True)
    real, labels = shuffle(real, labels)
    samples = real[:gensize]
    samples_labels = labels[:gensize]

    test_samples, test_labels = load_images(dataset, train=False)
    fid = calc_frechet_distance_inception(shuffle(real, labels)[0], test_samples, 10)
    inception = calc_inception_score(test_samples, dataset)
    acc = get_accuracies(samples, samples_labels, test_samples, test_labels)
    print(f"Inception score: {inception} \n"
          f"FID: {fid}\n"
          f"classifier acc from real samples {acc}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, default=None,
                        help='Path to the generator that should be evaluated')
    parser.add_argument('--generator_dir', type=str, default=None,
                        help='Path to the directory containing multiple generators that should be evaluated')
    parser.add_argument('--gen_data_size', type=int, default=10000,
                        help='Number of samples to create and use for evaluation')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashionmnist', 'ptb'], default='mnist',
                        help=' dataset name')
    parser.add_argument('--save_location', type=str, default="../results",
                        help='Directory to save output')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of batches to run for inception score & frechet inception distance')
    parser.add_argument('--eval_iter', type=int, default=2000,
                        help='only evaluates save files if the iteration is divisible by this factor')
    parser.add_argument('--skip_IS', default=False, action='store_true',
                        help='Include flag to skip Inception Score calculation')
    parser.add_argument('--skip_FID', default=False, action='store_true',
                        help='Include flag to skip Frechet Inception Distance calculation')
    parser.add_argument('--skip_classifier_acc', default=False, action='store_true',
                        help='Include this flag to skip calculation of accuracy of'
                             ' classifiers trained on generated data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
