import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

import cntk as C
from cntk import Trainer
from cntk.layers import default_options
from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, LayerNormalization, Convolution, ConvolutionTranspose2D, Dense
from cntk.ops import element_times, relu, leaky_relu, reduce_mean
import cntk.device
from cntk.io import (MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDef, StreamDefs,
                     INFINITELY_REPEAT)
from cntk.learners import (adam, UnitType, learning_rate_schedule,
                           momentum_as_time_constant_schedule, momentum_schedule)
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter

import cntk.io.transforms as xforms

L1_lambda = 10

# training config
MINIBATCH_SIZE = 128
NUM_MINIBATCHES = 5000
LR = 0.0002
MOMENTUM = 0.5  # equivalent to beta1
MAP_FILE_X = "data//trainingMNIST//map.txt"
MAP_FILE_Y = "data//trainingMNIST//map.txt"

TB_LOGDIR_G_F = "tblogs_G_F"
TB_LOGDIR_G_G = "tblogs_G_G"
TB_LOGDIR_D_Y = "tblogs_D_Y"
TB_LOGDIR_D_X = "tblogs_D_X"

# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_dims, num_classes, randomize=True):
    transforms = [
        xforms.scale(width=image_dims[2], height=image_dims[1], channels=image_dims[0], interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),
        labels=StreamDef(field='label', shape=num_classes))),
                           randomize=randomize)
def conv(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    return c

def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_layernorm(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = LayerNormalization()(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init)
    return relu(r)

def conv_bn_leaky_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init)
    return leaky_relu(r)

def conv_leaky_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv(input, filter_size, num_filters, strides, init)
    return leaky_relu(r)


def conv_frac_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = ConvolutionTranspose2D(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_fract_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_frac_bn(input, filter_size, num_filters, strides, init)
    return relu(r)

def resblock_basic(input, num_filters):
    c1 = conv_layernorm(input, (3,3), num_filters)
    c2 = conv_layernorm(c1, (3, 3), num_filters)
    return relu(c2)

def resblock_basic_stack(input, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resblock_basic(l, num_filters)
    return l

def generator(h0):
    with default_options(init=C.normal(scale=0.02)):
        print('Generator input shape: ', h0.shape)

        # c7s1-32,d64,d128,R128,R128,R128, R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
        # c7s1-32
        h1 = conv_bn_relu(h0, (7,7), 32)
        print('h1 shape', h1.shape)

        # d64
        h2 = conv_bn_relu(h1, (3,3), 64, strides=(2,2))
        print('h2 shape', h2.shape)

        # d128
        h3 = conv_bn_relu(h2, (3,3), 128, strides=(2,2))
        print('h3 shape', h3.shape)

        # R128 x 9
        h4 = resblock_basic_stack(h3, 9, 128)
        print('h4 shape', h4.shape)

        # u64
        h5 = conv_fract_bn_relu(h4, (3,3), 64, (2, 2))
        print('h5 shape', h5.shape)

        # u32
        h6 = conv_fract_bn_relu(h5, (3,3), 32, (2, 2))
        print('h6 shape', h6.shape)

        # c7s1-3
        h7 = conv_bn_relu(h6, (7,7), 3)
        print('h7 shape', h7.shape)
        return h7



def discriminator(h0):
    with default_options(init=C.normal(scale=0.02)):
        print('Discriminator input shape: ', h0.shape)

        h1 = conv_leaky_relu(h0, (4,4), 64, strides=(2,2))
        print('h1 shape', h1.shape)

        h2 = conv_bn_leaky_relu(h1, (4,4), 128, strides=(2,2))
        print('h2 shape', h2.shape)

        h3 = conv_bn_leaky_relu(h2, (4,4), 256, strides=(2,2))
        print('h3 shape', h3.shape)

        h4 = conv_bn_leaky_relu(h3, (4,4), 512, strides=(2,2))
        print('h4 shape', h4.shape)

        h5 = conv(h4, (1,1), 1, strides=(1,1))
        print('h5 shape', h5.shape)

def build_graph(image_shape, generator, discriminator):
    real_X = C.input(image_shape)
    real_Y = C.input(image_shape)
    fake_X_sample = C.input(image_shape)
    fake_Y_sample = C.input(image_shape)

    # genG(X) => Y            - fake_Y
    genG = generator(real_X)
    # genF(Y) => X            - fake_X
    genF = generator(real_Y)
    # genF( genG(Y) ) => Y    - fake_X~
    genF_back = genG.clone(
        method='share',
        substitutions={real_X.output: genG.output}
    )
    # genF( genG(X)) => X     - fake_Y~
    genG_back = genF.clone(
        method='share',
        substitutions={real_Y.output: genF.output}
    )
    # DY_fake is the discriminator for Y that takes in genG(X)
    # DX_fake is the discriminator for X that takes in genF(Y)
    discY_fake = discriminator(genG)
    discX_fake = discriminator(genF)

    g_loss_G = reduce_mean((discY_fake - np.ones(discY_fake)) ** 2) \
            + L1_lambda * reduce_mean(np.abs(real_X - genF_back)) \
            + L1_lambda * reduce_mean(np.abs(real_Y - genG_back))

    g_loss_F = reduce_mean((discX_fake - np.ones(discX_fake)) ** 2) \
            + L1_lambda * reduce_mean(np.abs(real_X - genF_back)) \
            + L1_lambda * reduce_mean(np.abs(real_Y - genG_back))

    # DY is the discriminator for Y that takes in Y
    # DX is the discriminator for X that takes in X
    DY = discY_fake.clone(
        method='share',
        substitution={genG.output : real_Y.output}
    )
    DX = discX_fake.clone(
        method='share',
        substitution={genF.output : real_X.output}
    )
    DY_fake_sample = discY_fake.clone(
        method='share',
        substitution={genG.output : fake_Y_sample.output}
    )
    DX_fake_sample = discX_fake.clone(
        method='share',
        substitution={genF.output : fake_X_sample.output}
    )

    softL_c =0.05
    DY_loss_real = reduce_mean((DY - np.ones_like(DY) * np.abs(np.random.normal(1.0, softL_c))) ** 2)
    DY_loss_fake = reduce_mean((DY_fake_sample - np.zeros_like(DY_fake_sample)) ** 2)
    DY_loss = (DY_loss_real + DY_loss_fake) / 2

    DX_loss_real = reduce_mean((DX - np.ones_like(DX) * np.abs(np.random.normal(1.0, softL_c))) ** 2)
    DX_loss_fake = reduce_mean((DX_fake_sample - np.zeros_like(DX_fake_sample)) ** 2)
    DX_loss = (DX_loss_real + DX_loss_fake) / 2

    test_X = C.input(image_shape)
    test_Y = C.input(image_shape)

    testG = genG.clone(
        methond='share',
        substitution={real_X.output:test_X.output}
    )
    testF = genF.clone(
        methond='share',
        substitution={real_Y.output:test_Y.output}
    )
    testF_back = genF_back.clone(
        method='share',
        substitution={genG.output:testG.output}
    )
    testG_back = genG_back.clone(
        method='share',
        substitution={genF.output:testF.output}
    )

    DX_optim= adam(DX_loss.parameters,
        lr=learning_rate_schedule(LR, UnitType.sample),
        momentum=momentum_schedule(0.5))

    DY_optim = adam(DY_loss.parameters,
                    lr=learning_rate_schedule(LR, UnitType.sample),
                    momentum=momentum_schedule(0.5))
    G_optim = adam(g_loss_G.parameters,
                    lr=learning_rate_schedule(LR, UnitType.sample),
                    momentum=momentum_schedule(0.5))

    F_optim = adam(g_loss_F.parameters,
                    lr=learning_rate_schedule(LR, UnitType.sample),
                   momentum=momentum_schedule(0.5))

    # Setup Tensor Board
    print_frequency_mbsize = NUM_MINIBATCHES // 25
    pp_G_G = [ProgressPrinter(print_frequency_mbsize)]
    pp_G_F = [ProgressPrinter(print_frequency_mbsize)]
    pp_D_X = [ProgressPrinter(print_frequency_mbsize)]
    pp_D_Y = [ProgressPrinter(print_frequency_mbsize)]

    tb_G_G = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_G_G, model=genG)
    pp_G_G.append(tb_G_G)

    tb_G_F = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_G_F, model=genG)
    pp_G_F.append(tb_G_F)

    tb_D_X = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_D_X, model=DX)
    pp_D_X.append(tb_D_X)

    tb_D_Y = TensorBoardProgressWriter(freq=10, log_dir=TB_LOGDIR_D_Y, model=DY)
    pp_D_Y.append(tb_D_Y)

    # Instantiate the trainers
    G_G_trainer = Trainer(
        genG,
        (g_loss_G, None),
        G_optim,
        progress_writers=pp_G_G
    )

    G_F_trainer = Trainer(
        genF,
        (g_loss_F, None),
        F_optim,
        progress_writers=pp_G_F
    )

    # DY is the discriminator for Y that takes in Y
    # DX is the discriminator for X that takes in X
    D_Y_trainer = Trainer(
        DY,
        (DY_loss, None),
        DY_optim,
        progress_writers=pp_D_Y
    )

    D_X_trainer = Trainer(
        DX,
        (DX_loss, None),
        DX_optim,
        progress_writers=pp_D_X
    )

    return (real_X, real_Y, fake_X_sample, fake_Y_sample, test_X, test_Y,
            DX_optim, DY_optim, G_optim, F_optim, G_G_trainer, G_F_trainer, D_X_trainer, D_Y_trainer,
            tb_G_G, tb_G_F, tb_D_X, tb_D_Y)

if __name__=='__main__':

    num_channels, image_height, image_width = (3, 256, 256)
    h0 = C.input((num_channels, image_height, image_width))
    genG = generator(h0)
    disc = discriminator(h0)
    # DEFINE OUR MODEL AND LOSS FUNCTIONS
    # -------------------------------------------------------

def train():
    IMAGE_SHAPE = (3, 256, 256)
    reader_train_X = create_mb_source(MAP_FILE_X, num_classes=10)
    reader_train_Y = create_mb_source(MAP_FILE_Y, num_classes=10)
    real_X, real_Y, fake_X_sample, fake_Y_sample, test_X, test_Y,\
            DX_optim, DY_optim, G_optim, F_optim, \
            G_G_trainer, G_F_trainer, D_X_trainer, D_Y_trainer, \
            tb_G_G, tb_G_F, tb_D_X, tb_D_Y = build_graph(image_shape=IMAGE_SHAPE,generator=generator, discriminator=discriminator)

    input_dynamic_axes = [C.Axis.default_batch_axis()]
    real_X = C.input((3, 256, 256), dynamic_axes=input_dynamic_axes)
    real_Y = C.input((3, 256, 256), dynamic_axes=input_dynamic_axes)

    input_map_X = {real_X: reader_train_X.streams.features}
    input_map_Y = {real_Y: reader_train_Y.streams.features}
    for train_step in range(NUM_MINIBATCHES):
        X_data = reader_train_X.next_minibatch(MINIBATCH_SIZE, input_map_X)
        batch_inputs_X = {real_X: X_data[real_X].data}
        Y_data = reader_train_Y.next_minibatch(MINIBATCH_SIZE, input_map_Y)
        batch_inputs_Y = {real_Y: Y_data[real_Y].data}

        G_G_trainer.train_minibatch(batch_inputs_X)
        generated_images_G = fake_Y_sample.eval(batch_inputs_X)
        D_X_trainer.train_minibatch(batch_inputs_Y, generated_images_G)

        G_F_trainer.train_minibatch(batch_inputs_Y)
        generated_images_F = fake_X_sample.eval(batch_inputs_Y)
        D_Y_trainer.train_minibatch(batch_inputs_X, generated_images_F)

        G_G_trainer.summarize_training_progress()
        D_X_trainer.summarize_training_progress()

        G_F_trainer.summarize_training_progress()
        D_Y_trainer.summarize_training_progress()

        utils.logTensorBoard(G_G_trainer, tb_G_G, "G_G", train_step)
        utils.logTensorBoard(D_X_trainer, tb_D_X, "D_X", train_step)
        utils.logTensorBoard(G_F_trainer, tb_G_G, "G_F", train_step)
        utils.logTensorBoard(D_Y_trainer, tb_D_Y, "D_Y", train_step)

        G_G_trainer_loss = G_G_trainer.previous_minibatch_loss_average
        G_F_trainer_loss = G_F_trainer.previous_minibatch_loss_average