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

# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_dims, num_classes, randomize=True):
    transforms = [
        xforms.scale(width=image_dims[2], height=image_dims[1], channels=image_dims[0], interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),
        labels=StreamDef(field='label', shape=num_classes))),
                           randomize=randomize)


def generator():
    print("TBD")


def discriminator():
    print("TBD")

def main():
    print("TBD")

    # DEFINE OUR MODEL AND LOSS FUNCTIONS
    # -------------------------------------------------------

    real_X = None # read images
    real_Y = None # TBD

    # genG(X) => Y            - fake_B
    genG = generator(real_X, name="generatorG")
    # genF(Y) => X            - fake_A
    genF = generator(real_Y, name="generatorF")
    # genF( genG(Y) ) => Y    - fake_A_
    genF_back = generator(genG, name="generatorF", reuse=True)
    # genF( genG(X)) => X     - fake_B_
    genG_back = generator(genF, name="generatorG", reuse=True)

    # DY_fake is the discriminator for Y that takes in genG(X)
    # DX_fake is the discriminator for X that takes in genF(Y)
    discY_fake = discriminator(genG, reuse=False, name="discY")
    discX_fake = discriminator(genF, reuse=False, name="discX")

    g_loss_G = tf.reduce_mean((discY_fake - tf.ones_like(discY_fake)) ** 2) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

    g_loss_F = tf.reduce_mean((discX_fake - tf.ones_like(discX_fake)) ** 2) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_X - genF_back)) \
            + L1_lambda * tf.reduce_mean(tf.abs(real_Y - genG_back))

    fake_X_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_X_sample")
    fake_Y_sample = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_Y_sample")

    # DY is the discriminator for Y that takes in Y
    # DX is the discriminator for X that takes in X
    DY = discriminator(real_Y, reuse=True, name="discY")
    DX = discriminator(real_X, reuse=True, name="discX")
    DY_fake_sample = discriminator(fake_Y_sample, reuse=True, name="discY")
    DX_fake_sample = discriminator(fake_X_sample, reuse=True, name="discX")

    DY_loss_real = tf.reduce_mean((DY - tf.ones_like(DY) * np.abs(np.random.normal(1.0, softL_c))) ** 2)
    DY_loss_fake = tf.reduce_mean((DY_fake_sample - tf.zeros_like(DY_fake_sample)) ** 2)
    DY_loss = (DY_loss_real + DY_loss_fake) / 2

    DX_loss_real = tf.reduce_mean((DX - tf.ones_like(DX) * np.abs(np.random.normal(1.0, softL_c))) ** 2)
    DX_loss_fake = tf.reduce_mean((DX_fake_sample - tf.zeros_like(DX_fake_sample)) ** 2)
    DX_loss = (DX_loss_real + DX_loss_fake) / 2

    test_X = None # TBD read images
    test_Y = None # TBD read images

    testG = generator(test_X,  name="generatorG", reuse=True)
    testF = generator(test_Y,  name="generatorF", reuse=True)
    testF_back = generator(testG, name="generatorF", reuse=True)
    testG_back = generator(testF, name="generatorG", reuse=True)

    print('Optimizing using {}'.format(OPTIM_PARAMS))
    DX_optim, DX_lr = adam(DX_loss, DX_vars,
                           OPTIM_PARAMS['start_lr'][2], OPTIM_PARAMS['end_lr'][2], OPTIM_PARAMS['lr_decay_start'][2],
                           OPTIM_PARAMS['momentum'][2], 'D_X')

    DY_optim, DY_lr = adam(DY_loss, DY_vars,
                           OPTIM_PARAMS['start_lr'][3], OPTIM_PARAMS['end_lr'][3], OPTIM_PARAMS['lr_decay_start'][3],
                           OPTIM_PARAMS['momentum'][3], 'D_Y')

    G_optim, G_lr = adam(g_loss_G, g_vars_G,
                         OPTIM_PARAMS['start_lr'][1], OPTIM_PARAMS['end_lr'][1], OPTIM_PARAMS['lr_decay_start'][1],
                         OPTIM_PARAMS['momentum'][1], 'G')

    F_optim, F_lr = adam(g_loss_F, g_vars_F,
                         OPTIM_PARAMS['start_lr'][0], OPTIM_PARAMS['end_lr'][0], OPTIM_PARAMS['lr_decay_start'][0],
                         OPTIM_PARAMS['momentum'][0], 'F')

     for train_step in range(NUM_MINIBATCHES):
        generated_X, generated_Y = sess.run([genF, genG])
        _, _, _, _, summary_str = sess.run([G_optim, DY_optim, F_optim, DX_optim, summary_op],
                                   feed_dict={fake_Y_sample: cache_Y.fetch(generated_Y),
                                              fake_X_sample: cache_X.fetch(generated_X)})





