import copy
import datetime
import random

import torch.optim as optim
import torch.utils.data as data
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from models import *
from ops import exp_mov_avg
from privacy_analysis import get_epsilon
from utils import *

CLIP_BOUND = 1.

SENSITIVITY = 2.
DATA_ROOT = '../../resources/data/'

##########################################################
### hook functions
##########################################################
def master_hook_adder(module, grad_input, grad_output):
    '''
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def dp_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global noise_multiplier
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    ### add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])

    return tuple(grad_input_new)


def download_pretrained_discriminators_from_google_drive(download_path, dataset):
    # skip if loaddir alraedy has contents
    if os.path.exists(download_path) and len(os.listdir(download_path)) > 1:
        print(f"Skipping pretrained discriminator download since {download_path} is not empty")
        return

    id = None
    if dataset == 'mnist':
        id = '1mgDuj5dhuF9zR5F4sMdy4lVAJO0NiveG'
    elif dataset == 'fashionmnist':
        id = '1HNSDN3JDoVWLjgVoA57J-CshonUrPg4H'

    if id is None:
        raise ValueError(f"No support for {dataset} pretrained discriminator download")
    download_path = "/".join(download_path.split("/")[:-1])  # remove last directory, gets added by download
    zipfile = os.path.join(download_path, "discriminators.zip")
    gdd.download_file_from_google_drive(file_id=id, dest_path=zipfile,
                                        unzip=True, showsize=False, overwrite=False)
    # delete zip, files where extracted already
    # for some reason the file isn't properly closed after unzipping so we do that here.
    with open(zipfile, 'r') as f:
        f.close()
    os.remove(zipfile)


##########################################################
### main
##########################################################
def main(args):
    ### config
    global noise_multiplier
    dataset = args.dataset
    dataset_name = args.dataset
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    load_dir = args.load_dir
    save_dir = args.save_dir
    if_dp = (noise_multiplier > 0.)
    gen_arch = args.gen_arch
    num_gpus = args.num_gpus
    download_discriminators = args.pretrain_download
    save_iters = args.save_iterations


    ### CUDA
    use_cuda = torch.cuda.is_available()
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch_type = "torch.cuda"
        map_location = lambda storage, loc: storage.cuda()
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        torch_type = "torch"
        map_location = 'cpu'

    ### Download pretrained discriminators
    if download_discriminators:
        download_pretrained_discriminators_from_google_drive(download_path=load_dir, dataset=dataset)

    ### Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### data loading
    trainset, NUM_CLASSES, width, height, classes_split = load_dataset(dataset_name, DATA_ROOT, save_dir=save_dir, flip_odd=True)
    if dataset_name in ['mitbih', 'ptb']:
        test_samples, test_labels = load_ecg_heartbeat_samples_labels(DATA_ROOT, dataset=dataset_name, train=False)
        print(f"test samples shape: {test_samples.shape}")

    print(f"Loaded data {dataset_name}"
          f"\n\ttrainset length: {len(trainset)}"
          f"\n\tsamples width, height: {width}, {height}"
          f"\n\tnumber of classes: {NUM_CLASSES}, classes split: {classes_split}\n\n")

    IMG_DIM = width * height
    samples_dim = (width, height)
    SAMPLES_PER_BATCH = NUM_CLASSES # for data generation, is not batchsize for training

    ### Fix noise for visualization
    if latent_type == 'normal':
        fix_noise = torch.randn(SAMPLES_PER_BATCH, z_dim)
    elif latent_type == 'bernoulli':
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((SAMPLES_PER_BATCH, z_dim)).view(SAMPLES_PER_BATCH, z_dim)
    else:
        raise NotImplementedError

    ### Set up models
    if gen_arch == 'DCGAN':
        netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=NUM_CLASSES,
                              samples_dim=samples_dim, torch_type=torch_type)
    elif gen_arch == 'ResNet':
        netG = GeneratorResNet(z_dim=z_dim, model_dim=model_dim, num_classes=NUM_CLASSES,
                               samples_dim=samples_dim, torch_type=torch_type)
    else:
        raise ValueError

    netGS = copy.deepcopy(netG)
    netD_list = []
    for i in range(num_discriminators):
        netD = DiscriminatorDCGAN(num_classes=NUM_CLASSES, samples_dim=samples_dim, model_dim=model_dim)
        netD_list.append(netD)

    ### Load pre-trained discriminators
    if load_dir is not None:
        for netD_id in range(num_discriminators):
            print('Load NetD ', str(netD_id))
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD = netD_list[netD_id]
            netD.load_state_dict(torch.load(network_path, map_location=map_location))

    netG = netG.to(device0)
    for netD_id, netD in enumerate(netD_list):
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD.to(device)

    ### Set up optimizers
    optimizerD_list = []
    for i in range(num_discriminators):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_dir is not None:
        assert os.path.exists(os.path.join(load_dir, 'indices.npy'))
        print('load indices from disk')
        indices_full = np.load(os.path.join(load_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('create indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    dataset_full_size = len(trainset)
    print('Size of the dataset: ', trainset_size)

    input_pipelines = []
    for i in range(num_discriminators):
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=args.batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Register hook
    global dynamic_hook_function
    for netD in netD_list:
        netD.conv1.register_backward_hook(master_hook_adder)

    ### Create generator save dir
    os.makedirs(os.path.join(save_dir, "gens"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    prev = 0
    costs = []
    accuracies = []
    for iter in range(args.iterations + 1):
        #########################
        ### Update D network
        #########################
        netD_id = np.random.randint(num_discriminators, size=1)[0]
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD = netD_list[netD_id]
        optimizerD = optimizerD_list[netD_id]
        input_data = input_pipelines[netD_id]

        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(critic_iters):
            real_data, real_y = next(input_data)
            real_data = real_data[:,:,:width, :width].reshape(-1, IMG_DIM)
            real_data = real_data.to(device)
            real_y = real_y.view(-1)
            real_y = real_y.to(device)
            real_data_v = autograd.Variable(real_data)

            ### train with real
            dynamic_hook_function = dummy_hook
            netD.zero_grad()
            D_real_score = netD(real_data_v, real_y)
            D_real = -D_real_score.mean()

            ### train with fake
            batchsize = real_data.shape[0]
            if latent_type == 'normal':
                noise = torch.randn(batchsize, z_dim).to(device0)
            elif latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
            else:
                raise NotImplementedError
            noisev = autograd.Variable(noise)
            fake = autograd.Variable(netG(noisev, real_y.to(device0)).data)
            inputv = fake.to(device)
            D_fake = netD(inputv, real_y.to(device))
            D_fake = D_fake.mean()

            ### train with gradient penalty
            gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, device)
            D_cost = D_fake + D_real + gradient_penalty

            ### train with epsilon penalty
            logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
            D_cost += logit_cost

            ### update
            D_cost.backward()
            Wasserstein_D = -D_real - D_fake
            optimizerD.step()

        del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
        torch.cuda.empty_cache()

        ############################
        # Update G network
        ###########################
        if if_dp:
            ### Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_conv_hook
        else:
            ### Only modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_conv_hook

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ### train with sanitized discriminator output
        if latent_type == 'normal':
            noise = torch.randn(batchsize, z_dim).to(device0)
        elif latent_type == 'bernoulli':
            noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
        else:
            raise NotImplementedError
        label = torch.randint(0, NUM_CLASSES, [batchsize]).to(device0)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, label)
        fake = fake.to(device)
        label = label.to(device)
        G = netD(fake, label)
        G = - G.mean()

        ### update
        G.backward()
        G_cost = G
        optimizerG.step()

        ### update the exponential moving average
        exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

        ############################
        ### Results visualization
        ############################
        samples, labels, acc = None, None, None
        if iter % args.vis_step == 0:
            generate_sample(dataset_name, iter, netGS, fix_noise, save_dir+"/samples", device0,
                            num_classes=NUM_CLASSES, img_h=height, img_w=width, experimental_flip=True)


        if iter < 5 or iter % args.print_step == 0:
            now = datetime.datetime.now().timestamp()
            seconds = now - prev
            prev = datetime.datetime.now().timestamp()
            # save all costs in list to print on termination
            G_cost_val = G_cost.cpu().data
            D_cost_val = D_cost.cpu().data
            Wasserstein_D_val = Wasserstein_D.cpu().data
            costs.append([str(iter), G_cost_val, D_cost_val, Wasserstein_D_val])

            metric_string = 'Step: {}, G_cost:{}, D_cost:{}, ' \
                            'Wasserstein:{}, Time since previous print {}s'.format(iter, G_cost.cpu().data, D_cost_val,
                                                                                   Wasserstein_D_val, round(seconds))
            print(metric_string)

            with open(os.path.join(save_dir, 'costs_intermediate.csv'), 'a+') as f:
                latest = costs[-1]
                latest = [str(x) for x in latest]
                f.write("{},{},{},{}".format(iter, G_cost.cpu().data, D_cost_val, Wasserstein_D_val))

        if iter % args.save_step == 0 or iter in save_iters:
            ### save model
            torch.save(netGS.state_dict(), os.path.join(save_dir, 'gens/netGS_%d.pth' % iter))

            samples, labels = generate_samples(netGS, dataset_full_size, classes_split=classes_split,
                                               img_h=samples_dim[0], img_w=samples_dim[1], flatten=True,
                                               experimental_flip=True)

            if dataset_name in ['mitbih', 'ptb']:
                acc = train_classifier_mitbih(samples, labels, test_samples, test_labels)
                epsilon = get_epsilon(noise_multiplier, iter, delta=1/10000, batch_size=batchsize, prob=1/num_discriminators) if iter > 0 else 0
                accuracies.append({"iter": iter, "epsilon": epsilon, **acc})
                print(f"Accuracy of classifier: {acc}, ")

            # save generated samples
            if samples is not None and labels is not None:
                np.savez(os.path.join(save_dir, f'samples/samples_{iter}.npz'), samples=samples, labels=labels)

        del label, fake, noisev, noise, G, G_cost, D_cost, acc
        torch.cuda.empty_cache()

    # write costs
    with open(os.path.join(save_dir, 'costs.csv'), 'w+') as f:
        f.write("iteration, Generator cost, Discriminator cost, Wasserstein\n")
        for values in costs:
            # This is done with format string because str.join was giving issues with tensors.
            f.write(f"{values[0]},{values[1]},{values[2]},{values[3]}\n")
    print(accuracies)

    with open(os.path.join(save_dir, 'accuracies.csv'), 'w+') as f:
        f.write("Iteration, epsilon, accuracy, AUROC, AUPRC, F1\n")
        for adict in accuracies:
            f.write(",".join([str(val) for val in adict.values()]) + "\n")

    ### save model
    torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
    torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)
