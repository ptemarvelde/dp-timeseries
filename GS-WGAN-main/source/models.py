import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import functools
import numpy as np
from math import ceil, floor

from ops import SpectralNorm, one_hot_embedding, pixel_norm

# IMG_W = IMG_H = 28  # image width and height
IMG_C = 1  # image channel


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.ReLU(inplace=False),
                 upsample=functools.partial(F.interpolate, scale_factor=2)):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = SpectralNorm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, padding=0))
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = pixel_norm(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(pixel_norm(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GeneratorDCGAN(nn.Module):
    def z_in_y(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes, torch_type=self.torch_type)
        z_in = torch.cat([z, y_onehot], dim=1)
        return z_in

    def z_in_no_y(self, z, y):
        return z

    def __init__(self, z_dim=10, model_dim=64, num_classes=None, samples_dim=(28, 28), outact=nn.Sigmoid(),
                 torch_type="torch.cuda", experimental_timeseries=False):
        super(GeneratorDCGAN, self).__init__()

        self.experimental_timeseries=experimental_timeseries

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.sample_width = samples_dim[0]
        self.sample_height = samples_dim[1]

        assert self.sample_width == self.sample_height, "input must be square, add zeros if necessary"
        # # Factor to account from inputs of other shape than 28,28 (assumes square input)
        self.mult_factor = 4 * ceil(self.sample_width / 8) ** 2

        fc = nn.Linear(z_dim + num_classes, self.mult_factor * model_dim)
        deconv1 = nn.ConvTranspose2d(4 * model_dim, 2 * model_dim, 5)
        deconv2 = nn.ConvTranspose2d(2 * model_dim, model_dim, 5)

        # fit output of final layer to image dimensions by configuring kernel and padding precisely
        # we want the lowest amount of padding, so find compatible kernel
        padding = -1
        kernel = 0
        while kernel < 1:
            padding += 1
            kernel = - (14 + 2 * ceil(self.sample_width / 8) - 2 * padding - self.sample_width)

        deconv3 = nn.ConvTranspose2d(model_dim, IMG_C, kernel_size=int(kernel), stride=2, padding=int(padding))

        if num_classes is not None and num_classes > 1:
            self.z_in = self.z_in_y
        else:
            self.z_in = self.z_in_no_y

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact
        self.torch_type = torch_type

    def forward(self, z, y):
        z_in = self.z_in(z, y)
        output = self.fc(z_in)
        x = int(ceil(self.sample_width / 8))
        output = output.view(-1, 4 * self.model_dim, x, x)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        # output = output[:, :, :7, :7]
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.relu(output).contiguous()
        output = pixel_norm(output)

        output = self.deconv3(output, output_size=(-1, 1, self.sample_width, self.sample_height))
        output = self.outact(output)

        output = output.view(-1, self.sample_width * self.sample_height)
        # flip every other row to preserve temporal dynamics across pixel rows
        if self.experimental_timeseries:
            output[:, 0, 1::2, :] = output[:, 0, 1::2, ::-1]

        return output


class GeneratorResNet(nn.Module):
    def z_in_y(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes, torch_type=self.torch_type)
        z_in = torch.cat([z, y_onehot], dim=1)
        return z_in

    def z_in_no_y(self, z, y):
        return z

    def __init__(self, z_dim=10, model_dim=64, num_classes=10, samples_dim=(28, 28), outact=nn.Sigmoid(),
                 torch_type="torch.cuda", experimental_timeseries=False):
        super(GeneratorResNet, self).__init__()

        self.experimental_timeseries=experimental_timeseries
        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.sample_width = samples_dim[0]
        self.sample_height = samples_dim[1]
        assert self.sample_width == self.sample_height, "input must be square, add zeros if necessary"
        # # Factor to account from inputs of other shape than 28,28 (assumes square input)
        self.mult_factor = 4 * ceil(self.sample_width / 8) ** 2

        fc = SpectralNorm(nn.Linear(z_dim + num_classes, self.mult_factor * model_dim))
        block1 = GBlock(model_dim * 4, model_dim * 4)
        block2 = GBlock(model_dim * 4, model_dim * 4)
        block3 = GBlock(model_dim * 4, model_dim * 4)

        if num_classes is not None and num_classes > 1:
            self.z_in = self.z_in_y
        else:
            self.z_in = self.z_in_no_y

        # padding = 0 if ceil(self.sample_width / 8) ** 4 - self.sample_width < 2 else 1
        output = SpectralNorm(nn.Conv2d(model_dim * 4, IMG_C, kernel_size=3, padding=1))

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.fc = fc
        self.output = output
        self.relu = nn.ReLU()
        self.outact = outact
        self.torch_type = torch_type

    def forward(self, z, y):
        z_in = self.z_in(z, y)
        output = self.fc(z_in)
        x = int(np.ceil(self.sample_width / 8))
        output = output.view(-1, 4 * self.model_dim, x, x)
        output = self.relu(output)
        output = pixel_norm(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.outact(self.output(output))
        # TODO find better way to scale down to desired size
        cut_index = self.sample_width - output.shape[2]

        if cut_index:  # to facilitate shapes other than (28,28)
            output = output[:, :, :cut_index, :cut_index]
        output = output.view(-1, self.sample_width * self.sample_height)
        # flip every other row to preserve temporal dynamics across pixel rows
        if self.experimental_timeseries:
            output[:, 0, 1::2, :] = output[:, 0, 1::2, ::-1]

        return output


class DiscriminatorDCGAN(nn.Module):

    ####### differentiate between case with and without labels
    def y_h_sum_y(self, h, y):
        ly = self.linear_y(y)
        return torch.sum(ly * h, dim=1, keepdim=True)

    def y_h_sum_no_y(self, h, y):
        return torch.sum(h, dim=1, keepdim=True)

    ######

    def __init__(self, model_dim=64, samples_dim=(28, 28), num_classes=10, if_SN=True, experimental_timeseries=False):
        super(DiscriminatorDCGAN, self).__init__()

        self.experimental_timeseries=experimental_timeseries
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.sample_width = samples_dim[0]
        self.sample_height = samples_dim[1]

        assert self.sample_width == self.sample_height, "input must be square"
        # Factor to account from inputs of other shape than 28,28 (assumes square input)
        self.mult_factor = 4 * ceil(self.sample_width / 8) ** 2

        if self.num_classes is not None and self.num_classes > 1:
            self.y_h_sum = self.y_h_sum_y
        else:
            self.y_h_sum = self.y_h_sum_no_y

        if if_SN:
            self.conv1 = SpectralNorm(nn.Conv2d(1, model_dim, 5, stride=2, padding=2))
            self.conv2 = SpectralNorm(nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2))
            self.conv3 = SpectralNorm(nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2))
            self.linear = SpectralNorm(nn.Linear(self.mult_factor * model_dim, 1))
            self.linear_y = SpectralNorm(nn.Embedding(num_classes, self.mult_factor * model_dim))
        else:
            self.conv1 = nn.Conv2d(1, model_dim, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(model_dim, model_dim * 2, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, 5, stride=2, padding=2)
            self.linear = nn.Linear(self.mult_factor * model_dim, 1)
            self.linear_y = nn.Embedding(num_classes, self.mult_factor * model_dim)
        self.relu = nn.ReLU()

    def forward(self, input, y):
        input = input.view(-1, 1, self.sample_width, self.sample_height)

        h = self.relu(self.conv1(input))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        # TODO figure out what those 4s and model_dim needs to be for 14*14
        h = h.view(-1, self.mult_factor * self.model_dim)
        out = self.linear(h)
        out += self.y_h_sum(h, y)
        return out.view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, y, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param y:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        y = y.to(device)
        alpha = torch.rand(batchsize, 1)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, y)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty
