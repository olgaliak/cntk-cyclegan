import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

import cntk as C
from cntk import Trainer
from cntk.layers import default_options
from cntk.device import gpu, cpu
from cntk.initializer import normal
from cntk.io import (MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDef, StreamDefs,
                     INFINITELY_REPEAT)
from cntk.layers import Dense, Convolution2D, ConvolutionTranspose2D, BatchNormalization
from cntk.learners import (adam, UnitType, learning_rate_schedule,
                           momentum_as_time_constant_schedule, momentum_schedule)
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter

import cntk.io.transforms as xforms

C.device.try_set_default_device(C.device.gpu(0))
TB_LOGDIR_G = "tblogs_G"
TB_LOGDIR_D = "tblogs_D"
MAP_FILE = "data//trainingMNIST//map.txt"

isFast = True
PROGRESS_SAVE_STEP = 500

# architectural parameters
NUM_CHANNELS = 3
IMG_H, IMG_W = 28, 28
IMAGE_DIMS = (NUM_CHANNELS, IMG_H, IMG_W)
KERNEL_H, KERNEL_W = 5, 5
STRIDE_H, STRIDE_W = 2, 2

# Input / Output parameter of Generator and Discriminator
G_INPUT_DIM = 100

# training config
MINIBATCH_SIZE = 128
NUM_MINIBATCHES = 5000 if isFast else 10000
LR = 0.0002
MOMENTUM = 0.5  # equivalent to beta1

data_found = False

if os.path.isfile(MAP_FILE):
    data_found = True

if not data_found:
    raise ValueError("Can not file map file {0}".format(MAP_FILE))

print("Map file is {0}".format(MAP_FILE))

# Creates a minibatch source for training or testing
def create_mb_source(map_file, num_classes, randomize=True):
    transforms = [xforms.scale(width=IMG_H,  height = IMG_H, channels= NUM_CHANNELS, interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),
        labels=StreamDef(field='label', shape=num_classes))),
                           randomize=randomize)


np.random.seed(123)


def noise_sample(num_samples):
    return np.random.uniform(
        low=-1.0,
        high=1.0,
        size=[num_samples, G_INPUT_DIM]
    ).astype(np.float32)


# We expect the kernel shapes to be square in this tutorial and
# the strides to be of the same length along each data dimension
if KERNEL_H == KERNEL_W:
    gkernel = dkernel = KERNEL_H
else:
    raise ValueError('This tutorial needs square shaped kernel')

if STRIDE_H == STRIDE_W:
    gstride = dstride = STRIDE_H
else:
    raise ValueError('This tutorial needs same stride in all dims')


# Helper functions
def bn_with_relu(x, activation=C.relu):
    h = BatchNormalization(map_rank=1)(x)
    return C.relu(h)


# We use param-relu function to use a leak=0.2 since CNTK implementation
# of Leaky ReLU is fixed to 0.01
def bn_with_leaky_relu(x, leak=0.2):
    h = BatchNormalization(map_rank=1)(x)
    r = C.param_relu(C.constant((np.ones(h.shape) * leak).astype(np.float32)), h)
    return r


def convolutional_generator(z):
    with default_options(init=C.normal(scale=0.02)):
        print('Generator input shape: ', z.shape)

        s_h2, s_w2 = IMG_H // 2, IMG_W // 2  # Input shape (14,14)
        s_h4, s_w4 = IMG_H // 4, IMG_W // 4  # Input shape (7,7)
        gfc_dim = 1024
        gf_dim = 64

        h0 = Dense(gfc_dim, activation=None)(z)
        h0 = bn_with_relu(h0)
        print('h0 shape', h0.shape)

        h1 = Dense([gf_dim * 2, s_h4, s_w4], activation=None)(h0)
        h1 = bn_with_relu(h1)
        print('h1 shape', h1.shape)

        h2 = ConvolutionTranspose2D((gkernel,gkernel),
                                    num_filters=gf_dim * 2,
                                    strides=(gstride, gstride),
                                    pad=True,
                                    output_shape=(s_h2, s_w2),
                                    activation=None)(h1)
        h2 = bn_with_relu(h2)
        print('h2 shape', h2.shape)

        h3 = ConvolutionTranspose2D((gkernel,gkernel),
                                    num_filters=NUM_CHANNELS,
                                    strides=(gstride, gstride),
                                    pad=True,
                                    output_shape=(IMG_H, IMG_W),
                                    activation=C.sigmoid)(h2)
        print('h3 (output) shape :', h3.shape)

        # return C.reshape(h3, img_h * img_w)
        return h3


def convolutional_discriminator(x):
    with default_options(init=C.normal(scale=0.02)):
        dfc_dim = 1024
        df_dim = 64

        print('Discriminator convolution input shape', x.shape)
        #  x = C.reshape(x, (1, img_h, img_w))

        h0 = Convolution2D((dkernel, dkernel), 1, strides=(dstride, dstride))(x)
        h0 = bn_with_leaky_relu(h0, leak=0.2)
        print('h0 shape :', h0.shape)

        h1 = Convolution2D((dkernel, dkernel), df_dim, strides=(dstride, dstride))(h0)
        h1 = bn_with_leaky_relu(h1, leak=0.2)
        print('h1 shape :', h1.shape)

        h2 = Dense(dfc_dim, activation=None)(h1)
        h2 = bn_with_leaky_relu(h2, leak=0.2)
        print('h2 shape :', h2.shape)

        h3 = Dense(1, activation=C.sigmoid)(h2)
        print('h3(output) shape :', h3.shape)

        return h3

def build_graph(noise_shape, image_shape, generator, discriminator):
    input_dynamic_axes = [C.Axis.default_batch_axis()]
    Z = C.input(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input(image_shape, dynamic_axes=input_dynamic_axes)
    X_real_scaled = X_real/255

    # Create the model function for the generator and discriminator models
    X_fake = generator(Z)
    D_real = discriminator(X_real_scaled)
    D_fake = D_real.clone(
        method='share',
        substitutions={X_real_scaled.output: X_fake.output}
    )

    # Setup Tensor Board
    print_frequency_mbsize = NUM_MINIBATCHES // 25
    pp_G = [ProgressPrinter(print_frequency_mbsize)]
    pp_D = [ProgressPrinter(print_frequency_mbsize)]
    tb_G = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_G, model=X_fake)
    pp_G.append(tb_G)
    tb_D = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_D, model=D_real)
    pp_D.append(tb_D)

    # Create loss functions and configure optimazation algorithms
    G_loss = 1.0 - C.log(D_fake)
    D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))

    G_learner = adam(
        parameters=X_fake.parameters,
        lr=learning_rate_schedule(LR, UnitType.sample),
        momentum=momentum_schedule(0.5)
    )
    D_learner = adam(
        parameters=D_real.parameters,
        lr=learning_rate_schedule(LR, UnitType.sample),
        momentum=momentum_schedule(0.5)
    )

    # Instantiate the trainers
    G_trainer = Trainer(
        X_fake,
        (G_loss, None),
        G_learner,
        progress_writers=pp_G
    )
    D_trainer = Trainer(
        D_real,
        (D_loss, None),
        D_learner,
        progress_writers=pp_D
    )

    return X_real, X_fake, Z, G_trainer, D_trainer, tb_G, tb_D


def train(reader_train, generator, discriminator):
    X_real, X_fake, Z, G_trainer, D_trainer, tb_G, tb_D = build_graph(G_INPUT_DIM, IMAGE_DIMS, generator,
                                                                      discriminator)

    k = 2

    input_map = {X_real: reader_train.streams.features}
    for train_step in range(NUM_MINIBATCHES):

        # train the discriminator model for k steps
        for gen_train_step in range(k):
            Z_data = noise_sample(MINIBATCH_SIZE)
            X_data = reader_train.next_minibatch(MINIBATCH_SIZE, input_map)
            if X_data[X_real].num_samples == Z_data.shape[0]:
                batch_inputs = {X_real: X_data[X_real].data, Z: Z_data}
                D_trainer.train_minibatch(batch_inputs)

        # train the generator model for a single step
        Z_data = noise_sample(MINIBATCH_SIZE)
        batch_inputs = {Z: Z_data}

        G_trainer.train_minibatch(batch_inputs)
        G_trainer.train_minibatch(batch_inputs)

        if np.mod(train_step, PROGRESS_SAVE_STEP) == 0:
            noise = noise_sample(36)
            images = X_fake.eval(noise)
            utils.plot_images(images, subplot_shape=[6, 6],iteration=train_step)

        D_trainer.summarize_training_progress()
        G_trainer.summarize_training_progress()

        utils.logTensorBoard(G_trainer, tb_G, "G", train_step)
        utils.logTensorBoard(D_trainer, tb_D, "D", train_step)

        G_trainer_loss = G_trainer.previous_minibatch_loss_average

    return Z, X_fake, G_trainer_loss


reader_train = create_mb_source(MAP_FILE, num_classes=10)

G_input, G_output, G_trainer_loss = train(reader_train,
                                          convolutional_generator,
                                          convolutional_discriminator)

print("Training loss of the generator is: {0:.2f}".format(G_trainer_loss))

noise = noise_sample(36)
images = G_output.eval({G_input: noise})
utils.plot_images(images, subplot_shape=[6, 6], iteration="test")
