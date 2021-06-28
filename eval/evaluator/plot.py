import csv
import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd

from matplotlib import gridspec

font_size = 14
plt.rcParams.update({'font.size': font_size})

real_values = {
    "MNIST": {"Inception Score": 9.93, "Frechet Inception Distance": 4.46, "Downstream Classifier Accuracy": 0.8475},
    "Fashion-MNIST": {"Inception Score": 8.490821, "Frechet Inception Distance": 21.437124,
                      "Downstream Classifier Accuracy": 0.7456}}

gswgan_epsilon = [0, 1.9773212984297708, 2.8359660875805357, 3.505516654709685, 4.085556133864012, 4.619599772452158,
                  5.08300270614067, 5.5464056693000225, 5.961636258679441, 6.347062016815745, 6.732487620428369,
                  7.1179129496953815, 7.503338562723269, 7.838380111565679, 8.146129361470138, 8.453878185643482,
                  8.761627902183061, 9.069376973839756, 9.377125861681217, 9.684874584077484, 9.992624296723676]
gswgan_iters = np.arange(0, 20001, 1000)

dpcgan_epsilon = [0.240615351, 0.725263442, 1.020782934, 1.250130075, 1.444492501, 1.618196586, 1.774888015,
                  1.920578892, 2.057869056, 2.184734338, 2.308699737, 2.423266077, 2.53514399, 2.642830889, 2.74688979,
                  2.847625143, 2.945384001, 3.040494761, 3.133139278, 3.223502934, 3.311806327, 3.398115963,
                  3.482679968, 3.565615207, 3.646926106, 3.726784601, 3.805334139, 3.882552443, 3.958477927,
                  4.033264922, 4.106963156, 4.179625629, 4.251308898, 4.32207339, 4.391836788, 4.460690671, 4.528760124,
                  4.59611941, 4.662714804, 4.72838862, 4.79351828, 4.858004022, 4.921627805, 4.984899983, 5.047277949,
                  5.109253038, 5.170577818, 5.231373314, 5.291645376, 5.351388574, 5.410608385, 5.469437443, 5.52760547,
                  5.58567062, 5.64278733, 5.69990404, 5.756318105, 5.812383964, 5.868376946, 5.923392418, 5.978407891,
                  6.033125346, 6.087090898, 6.141056449, 6.194713628, 6.247629724, 6.300545819, 6.353379171,
                  6.405246275, 6.457113379, 6.508980482, 6.560201844, 6.611020419, 6.661838995, 6.71265757, 6.762555296,
                  6.812325807, 6.862096318, 6.911866828, 6.960759995, 7.009482904, 7.058205813, 7.106928722,
                  7.155171392, 7.202847162, 7.250522931, 7.298198701, 7.34587447, 7.392814017, 7.439443109, 7.486072201,
                  7.532701294, 7.579330386, 7.625407651, 7.670990527, 7.716573404, 7.76215628, 7.807739157, 7.853322033,
                  7.89798492, 7.942522042, 7.987059164, 8.031596286, 8.076133408, 8.120670531, 8.16447469, 8.207966519,
                  8.251458347, 8.294950176, 8.338442004, 8.381933833, 8.425425661, 8.467952833, 8.510399828,
                  8.552846823, 8.595293819, 8.637740814, 8.680187809, 8.722634804, 8.764595222, 8.805997844,
                  8.847400466, 8.888803088, 8.93020571, 8.971608332, 9.013010953, 9.054413575, 9.095570569, 9.135929277,
                  9.176287985, 9.216646693, 9.257005401, 9.297364109, 9.337722817, 9.378081525, 9.418440233,
                  9.458672842, 9.497988096, 9.537303349, 9.576618602, 9.615933855, 9.655249108, 9.694564362,
                  9.733879615, 9.773194868, 9.812510121, 9.851825374, 9.890106526, 9.928378783, 9.96665104, 9.999182458,
                  9.337722817, 9.378081525, 9.418440233, 9.458672842, 9.497988096, 9.537303349, 9.576618602,
                  9.615933855, 9.655249108, 9.694564362, 9.733879615, 9.773194868, 9.812510121, 9.851825374,
                  9.890106526, 9.928378783, 9.96665104, 9.999182458]

dpcgan_iters = np.arange(0, 150001, 1000)


def add_values_to_plot(ax, x, y, name, std=None, color=None, marker=None, line=None):
    if std is not None and len(std) > 0:
        stdev = np.array(std)
        ax.fill_between(x, y - stdev, y + stdev, color='grey' if not color else color, alpha=0.2)

    ax.plot(x, y, label=name, color=color, marker=marker, linestyle=line)
    #
    # if color and marker:
    # elif color:
    #     ax.plot(x, y, label=name, color=color)
    # elif marker:
    #     ax.plot(x, y, label=name, marker=marker)
    # else:
    #     ax.plot(x, y, label=name)
    return ax


def add_metric_to_plot(ax, result_list, metric_key, name=None, epsilon=None, calibrate=False):
    result_list = result_list[result_list[:, 0].argsort()]
    iters = np.array([int(x) for x in result_list[:, 0]])

    results = [x.get(metric_key) for x in result_list[:, 1]]

    filtered_res = []
    filtered_iters = []
    for res, itera in zip(results, iters):
        if res:
            filtered_res.append(res)
            filtered_iters.append(itera)

    results = filtered_res
    iters = filtered_iters

    # if res is dict average all values and also plot stdev
    y = []
    stdev = []
    if type(results[0]) == dict:
        for i, iteration in enumerate(results):
            accuracies = [iteration.get(x) for x in iteration.keys()]
            y.append(np.mean(accuracies))
            # stdev.append(np.std(accuracies))
    elif type(results[0]) == tuple:  # if res is tuple also plot stdev
        y = [x[0] for x in results]
        stdev = [x[1] for x in results]
    else:
        y = results

    if epsilon:
        # find epsilons corresponding to iters:
        eps, eps_iters = epsilon[1], epsilon[0]
        epsilons = [eps[x] for x in [np.argwhere(eps_iters == iteration)[0][0] for iteration in iters]]
        x = epsilons
    else:
        x = iters

    y = np.array(y) if not calibrate else np.array(y) / calibrate
    x = np.array(x)

    if metric_key == 'Downstream Classifier Accuracy':
        print('\n'.join(map(str, y)))

    add_values_to_plot(ax, x, y, std=stdev, name=name)

    print(f"{name}, {metric_key}, last value: {y[-1]}")
    return ax, iters


def plot_metric(result_list, metric_key):
    """
    Returns simple plot of the metric given as key and its values in the result list
    @param  result_list 2d numpy array containing iterations in first column and a
            dict containing the corresponding values for the given metric in key
    @param metric_key metric to plot
    @param save_loc saves plots to this file location
    """
    # If there is a generator in the original files without an iter number this gen will have a
    # very large iter in the result_list. This check fixes that.
    # if iters[-1] > 1000000:
    #     iters[-1] = iters[-2] + 1000

    fig, ax = plt.subplots()
    fig, iters = add_metric_to_plot(ax, result_list, metric_key)

    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0, right=iters[-1])
    curr_tick_locs, _ = plt.xticks()
    plt.xticks(curr_tick_locs, curr_tick_locs / 1000)
    plt.xlabel("Iterations (x 1000)")
    plt.ylabel(metric_key)
    plt.grid()
    return fig


def plot_all_metrics(result_list, save_loc=None, iteration=None):
    # TODO create nice plot showing all metrics
    for metric in result_list[0][1].keys():
        plot_metric(result_list, metric)
        fig = plt.gcf()

        if not matplotlib.get_backend == "Agg":
            try:
                fig.show()
            except:
                print("unable to show plot")

        if save_loc:
            if iteration:
                fig.savefig(f"{save_loc}/{iteration}/{metric}.png", dpi=150, format='png')
            else:
                fig.savefig(f"{save_loc}/{metric}.png", dpi=150, format='png')


def plot_2_from_dict(res_dict1, res_dict2, save_loc=None, plot_real=False):
    plot_2_datasets(res_dict1["path"], res_dict1["name"], res_dict2["path"], res_dict2["name"],
                    res_dict1["batchsize"], res_dict2["batchsize"], save_loc, plot_real)


def plot_2_datasets(res_path1, name1, res_path2, name2, batchsize1=None, batchsize2=None, save_loc=None,
                    plot_real=False, epsilon1=None, epsilon2=None, dataset=None, metrics_to_plot=None,
                    calibrate_classifier_acc=False):
    res1 = get_results_from_file(res_path1)
    res2 = get_results_from_file(res_path2)

    if batchsize1 and batchsize2:
        res1 = np.array([[x[0] / (60000 / batchsize1), x[1]] for x in res1])
        res2 = np.array([[x[0] / (60000 / batchsize2), x[1]] for x in res2])

    fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(4.5 * len(metrics_to_plot), 4.3), dpi=250)

    i = 0
    for metric in res1[0][1].keys():
        if metrics_to_plot and metric not in metrics_to_plot:
            print(f"skipping metric {metric}")
            continue

        if metric == 'Downstream Classifier Accuracy' and calibrate_classifier_acc:
            calibrate = real_values[dataset][metric]
        else:
            calibrate = False

        curr_ax = axs[i]
        i += 1

        # plot also res2 current metric
        fig, x1 = add_metric_to_plot(curr_ax, res1, metric, name1, epsilon1, calibrate=calibrate)
        fig, x2 = add_metric_to_plot(curr_ax, res2, metric, name2, epsilon2, calibrate=calibrate)

        # plot values for real lines
        # TODO fix colors
        if plot_real:
            curr_ax.axhline(y=real_values[name1][metric], linestyle='--', label=f"{name1} real")
            curr_ax.axhline(y=real_values[name2][metric], linestyle='--', label=f"{name2} real")

        right_cutoff = 10 if epsilon1 else max(x1[-1], x2[-1])

        curr_ax.set_ylim(bottom=0)
        curr_ax.set_xlim(left=0,
                         right=right_cutoff
                         )
        curr_tick_locs = curr_ax.get_xticks()

        if batchsize1 and batchsize2:
            curr_ax.set(xlabel='Epoch')
            # curr_ax.xlabel("Epoch")
        elif epsilon1:
            base = 1.2
            curr_ax.set_xscale('function', functions=(lambda x: base ** x, lambda x: np.log(x) / np.log(base)))
            curr_ax.set(xlabel='Epsilon')
            # curr_ax.xlabel("Epsilon")
        else:
            curr_ax.set(xlabel='Iterations (x 1000)')
            # curr_ax.xlabel("Iterations (x 1000)")
            curr_ax.set_xticks(curr_tick_locs)
            curr_ax.set_xticklabels(curr_tick_locs / 1000)

        if i == len(metrics_to_plot):
            plt.legend()

        title = metric if metric != "Downstream Classifier Accuracy" else "Calibrated Downstream\n Classifier Accuracy"
        curr_ax.set_title(title)
        curr_ax.set_ylabel(title)
        curr_ax.grid()
        # curr_ax.legend()
    fig = plt.gcf()
    fig.suptitle(dataset)
    fig.tight_layout()
    if save_loc:
        fig.savefig(f"{save_loc}{name1}_{name2}_combined.png", dpi=fig.dpi, format='png')
    plt.show()

def plot_final(resfiles: list, names:list, save_loc=None, dataset=None):
    resdfs = [pd.read_csv(file) for file in resfiles]

    metrics_to_plot = \
        [
            "Downstream Classifier Accuracy",
            "Frechet Inception Distance",
            "Inception Score"
        ]

    markers = ['o', 'x']
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=250)
        ylabel = None
        for i, df in enumerate(resdfs):
            x = df['epsilon'].to_numpy()
            y = df[metric].to_numpy()
            stdev = df[f"{metric} stdev"].to_numpy()

            name = names[i] if names else resfiles[i].split("/")[-1]
            if metric == 'Downstream Classifier Accuracy' and dataset:
                # calibrate
                y = y / real_values[dataset][metric]
                ylabel = 'Calibrated accuracy'
            if metric == 'Downstream Classifier Accuracy':
                ylabel = 'Accuracy'
            add_values_to_plot(ax, x, y, name, std=stdev, marker=markers[i])
        ylabel = ylabel if ylabel else metric
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
        ax.set_xlim((0, 10))
        base = 1.2
        ax.set_xscale('function', functions=(lambda x: base ** x, lambda x: np.log(x) / np.log(base)))
        ax.set(xlabel='Epsilon')
        fig.tight_layout()

        if save_loc:
            fig.savefig(f"{save_loc}_{metric}.png", dpi=fig.dpi, format='png')
        fig.show()



def classifier_acc_to_csv(acc_dict, headers=True):
    """
    Transforms dict containing accuracy for different classifiers to csv
    """
    res = ""
    if headers:
        res += ", ".join([key for key in acc_dict.keys()])
        res += "\n"

    res += ", ".join([str(acc_dict[key]) for key in acc_dict.keys()])
    return res


def get_results_from_file(path):
    with open(path, 'r') as f:
        # for line in f:
        #     split = line.split(':', 1)
        #     iter = int(split[0])
        #     values = eval(split[1])
        #     res.append([iter, values])

        text = f.read()
        text = text.replace('\n', '')
        res = eval(text)

    return np.array(res)


def results_to_csv(path, outfile=None):
    result_list = get_results_from_file(path)
    csv = [["iteration", "IS", "IS std", "FID", "FID std", "classifier accuracy"]]

    for iteration, result in result_list:
        metrics = result.keys()
        scores = [str(result.get(metric)) for metric in metrics]
        scores.insert(0, str(iteration))
        csv.append(scores)

    csv_string = "\n".join([",".join(scores) for scores in csv]).replace("(", "").replace(")", "")

    if not outfile:
        outfile = path.replace(".txt", ".csv")

    with open(outfile, 'w+') as f:
        f.write(csv_string)


def eval_results_file(path, save_loc=None):
    result_list = get_results_from_file(path)
    plot_all_metrics(result_list, save_loc)


def plot_from_logs(path):
    gens = []
    results = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if "quantitative eval results" in line:
                line = line.replace("quantitative eval results: ", "")
                results.append(eval(line))
            elif "Finished generator evaluation" in line:
                words = line.split(" ")
                filename = words[5]

                underscore_index = filename.find("_")
                iteration = int(filename[underscore_index + 1:-5]) if underscore_index else -1

                gens.append(iteration)

    resfinal = []
    # for res, iteration in zip(gens, results):
    #     resfinal.append([res, iteration])
    # plot_all_metrics(np.array(resfinal))

    skiplist = []
    for x in gens:
        if x % 5000 == 0:
            skiplist.append(x)

    print(skiplist)


def merge_files():
    dpcgan_fashion_og = "../data/evaluations/inception_score/own_model/dpcgan_fashionmnist_combined.txt"
    dpcgan_mnist_og = "../data/evaluations/inception_score/own_model/dpcgan_mnist_combined.txt"
    gswgan_fashion_og = "../data/evaluations/inception_score/own_model/gswgan_fashionmnist_combined.txt"
    gswgan_mnist_og = "../data/evaluations/inception_score/own_model/gswgan_mnist_combined.txt"

    dpcgan_fashion_acc = "../data/evaluations/classifier_acc/dpcgan_fashionmnist.txt"
    dpcgan_mnist_acc = "../data/evaluations/classifier_acc/dpcgan_mnist.txt"
    gswgan_fashion_acc = "../data/evaluations/classifier_acc/gswgan_fashionmnist.txt"
    gswgan_mnist_acc = "../data/evaluations/classifier_acc/gswgan_mnist.txt"

    acc = ["../data/evaluations/classifier_acc/dpcgan_mnist.txt"]
    og = ["../data/evaluations/results_combined/dpcgan_mnist_150k.txt"]

    for inception_file, og_file in zip(acc, og):
        inception = get_results_from_file(inception_file)
        og = get_results_from_file(og_file)

        vals_counter = 0
        for row in og:
            iteration = row[0]
            for row2 in inception:
                if row2[0] == iteration:
                    row[1].update(row2[1])
            # if ogs[i][0] == inception[vals_counter][0]:
            #     ogs[i][1].update(inception[vals_counter])

        filename = inception_file[:-4] + "_combined.txt"
        print(filename)
        with open(filename, 'w+') as f:
            string = pprint.pformat(og)[6:-21]
            f.write(string)


def create_plots_epsilon():
    dpcgan_fashion = "../data/evaluations/results_combined/dpcgan_fashionmnist_combined.txt"
    dpcgan_mnist = "../data/evaluations/results_combined/dpcgan_mnist_combined.txt"
    gswgan_fashion = "../data/evaluations/results_combined/gswgan_fashionmnist_combined.txt"
    gswgan_mnist = "../data/evaluations/results_combined/gswgan_mnist_combined.txt"
    # results_to_csv(dpcgan_fashion)

    # plot_2_datasets(dpcgan_fashion, "DP CGAN", gswgan_fashion, "GS-WGAN",
    #                 epsilon1=(dpcgan_iters, dpcgan_epsilon), epsilon2=(gswgan_iters, gswgan_epsilon),
    #                 dataset="Fashion-MNIST")
    # plot_2_datasets(dpcgan_mnist, "DP CGAN", gswgan_mnist, "GS-WGAN",
    #                 epsilon1=(dpcgan_iters, dpcgan_epsilon), epsilon2=(gswgan_iters, gswgan_epsilon),
    #                 dataset="MNIST")
    metrics_to_plot = \
        [
            "Downstream Classifier Accuracy",
            "Frechet Inception Distance",
            "Inception Score"
        ]

    print("fashion")
    plot_2_datasets(dpcgan_fashion, "DP-CGAN", gswgan_fashion, "GS-WGAN",
                    epsilon1=(dpcgan_iters, dpcgan_epsilon), epsilon2=(gswgan_iters, gswgan_epsilon),
                    metrics_to_plot=metrics_to_plot,
                    dataset="Fashion-MNIST",
                    save_loc='../data/evaluations/results_combined/plots/multi_fashion',
                    calibrate_classifier_acc=True
                    )
    print("mnist")
    plot_2_datasets(dpcgan_mnist, "DP-CGAN", gswgan_mnist, "GS-WGAN",
                    epsilon1=(dpcgan_iters, dpcgan_epsilon), epsilon2=(gswgan_iters, gswgan_epsilon),
                    metrics_to_plot=metrics_to_plot,
                    dataset="MNIST",
                    save_loc='../data/evaluations/results_combined/plots/multi_mnist',
                    calibrate_classifier_acc=True
                    )


def plot_final_images(samples, labels, nrows, saveloc=None):
    """
    Plots nrows of samples in order of labels
    @param samples numpy array containing samples
    @param labels numpy array containing labels
    @param nrows number of rows to plot
    """
    # convert samples to single dimension
    if len(labels.shape) > 1:
        labels = np.squeeze(np.array([np.argwhere(x == 1) for x in labels]))

    # get samples to plot (assume shuffled samples,labels):
    order = labels.argsort()
    samples = samples[order]
    labels = labels[order]

    num_classes = len(np.unique(labels))
    to_plot = []
    for i in range(nrows):
        row = []
        for label in np.unique(labels):
            indices = np.squeeze(np.argwhere(labels == label))
            indices = indices[np.random.randint(0, len(indices))]
            row.append(samples[indices])
        to_plot.append(np.array(row))

    to_plot = np.array(to_plot)

    # from https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    nrow = nrows
    ncol = num_classes
    plt.figure(figsize=(ncol, nrows))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1.0, bottom=0.0,
                           left=0.0, right=1.0)
    for i in range(nrow):
        for j in range(ncol):
            im = to_plot[i, j].reshape(28, 28)
            ax = plt.subplot(gs[i, j])
            ax.imshow(im, cmap='gray')
            plt.axis('off')

    if saveloc:
        plt.savefig(saveloc, dpi=150)
    plt.show()


def plot_real_images():
    # archive = np.load("../../resources/dp-cgan/dp-merf/fashionmnist/eps10.0_last.npz")
    archive = np.load("../../resources/dp-cgan/dp-merf/fashionmnist/eps10.0_last.npz")
    samples = archive['data']
    labels = archive['labels']
    plot_final_images(samples, labels, nrows=3,
                      saveloc="../../eval/data/evaluations/results_combined/images/dpcgan-fashionmnist.png")

def merge_and_average_dicts(dict_a, dict_b):
    # https://stackoverflow.com/questions/25408370/combine-dictionaries-with-average-values-for-similar-keys-python
    d = {}
    for k, v in dict_a.items():
        if k in dict_b:
            val1, val2 = (v[0] + dict_b[k][0]) / 2, (
                        v[1] + dict_b[k][1]) / 2  # average sum of ele1 and ele2 in each value list
            d[k] = [val1, val2]  # set new value to key
        else:
            d[k] = v  # else just add the k,v

    for k, v in dict_b.items():  # add rest of keys from dict_b
        if k not in d:  # if key not already in d add it
            d[k] = v
    print(d)
    return d

def dict_to_csv(reslist, file, key, index=None):
    with open(file, 'w+') as f:
        f.write(f"{key}\n")
        for iter, adict in reslist:
            if index is not None:
                f.write(f"{adict.get(key)[index]}\n")
            else:
                f.write(f"{adict.get(key)}\n")


if __name__ == '__main__':
    gs_final_fashion = "../data/evaluations/gs-wgan/final/fashionmnist/final_results.csv"
    gs_final_mnist = "../data/evaluations/gs-wgan/final/mnist/final_results.csv"
    dp_final_fashion = "../data/evaluations/dpcgan/final/fashionmnist/final_results.csv"
    dp_final_mnist = "../data/evaluations/dpcgan/final/mnist/final_results.csv"

    plot_final([gs_final_fashion, dp_final_fashion], ["GS-WGAN", "DP-CGAN"], dataset='Fashion-MNIST',
               save_loc="../data/evaluations/final_plots/fashionmnist"
               )
    plot_final([gs_final_mnist, dp_final_mnist], ["GS-WGAN", "DP-CGAN"], dataset='MNIST',
               save_loc="../data/evaluations/final_plots/mnist"
               )

    # dataset = 'mnist'
    # model = 'gs-wgan'
    # run = 3
    # inputfile = f"../data/evaluations/{model}/final/{dataset}/run{run}/downstream classifier acc.txt"
    # outfile = inputfile.replace(".txt", ".csv")
    # res = get_results_from_file(inputfile)
    # printheaders = True
    # averages = []
    # with open(outfile, 'w+') as f:
    #     for iter in res:
    #         if iter[1].get('Downstream Classifier Accuracy'):
    #             if printheaders:
    #                 f.write("iter," + ",".join(list(iter[1]['Downstream Classifier Accuracy'].keys()))+ ",average\n")
    #                 printheaders = False
    #             accs = iter[1]['Downstream Classifier Accuracy'].values()
    #             average = np.mean(list(accs))
    #             averages.append(average)
    #             endline = '\n\n' if model == 'dpcgan' else '\n'
    #             f.write(str(iter[0]) + "," + ",".join(map(str, accs)) + f",{average}{endline}")
    #         # f.write("\n")
    # print(endline.join(map(str, averages)))

    # merge_files()
    # create_plots_epsilon()
    # run = 3
    # for model in ['gs-wgan', 'dpcgan']:
    #     for dataset in ['mnist', 'fashionmnist']:
    #         reslist = get_results_from_file(f"../data/evaluations/{model}/final/{dataset}/run{run}/run{run}.txt")
    #         for metric in ['Inception Score', 'Frechet Inception Distance']:
    #             dict_to_csv(reslist, f"../data/evaluations/{model}/final/{dataset}/run{run}/{metric}.csv", metric, 0)
    # dpcgan_fashion = "../data/evaluations/dpcgan/inception_own/fashionmnist/results.txt"
    # dpcgan_mnist = "../data/evaluations/dpcgan/inception_own/mnist/results.txt"
    # gswgan_fashion = "../data/evaluations/gs-wgan/inception_own/mnist/results.txt"
    # gswgan_mnist = "../data/evaluations/gs-wgan/inception_own/fashionmnist/results.txt"

    #
    #
    # mnist_path = "../data/evaluations/ResNet_default_mnist/results.txt"
    # fashionmnist_path = "../data/evaluations/ResNet_default_fashionmnist/results.txt"
    # # eval_results_file("../data/evaluations/dpcgan-mnist/results.txt")
    # # plot_2_datasets(mnist_path, "MNIST", fashionmnist_path, "Fashion-MNIST", plot_real=False)
    #
    # gswgan_path = "../data/evaluations/ResNet_default_mnist/results.txt"
    # dpcgan_path = "../data/evaluations/dpcgan-mnist/results.txt"
    # dataset1 = {"path": gswgan_path, "name": "GS-WGAN", "batchsize": 32}
    # dataset2 = {"path": dpcgan_path, "name": "DP-CGAN", "batchsize": 600}
    # # plot_2_datasets(gswgan_path, "GS-WGAN", dpcgan_path, "DP-CGAN", plot_real=False)
    # plot_2_from_dict(dataset1, dataset2)
