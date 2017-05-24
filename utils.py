import matplotlib.pyplot as plt
import os
import numpy as np
from cntk import reduce_mean
import os
from scipy.misc import imsave

def plot_images(images, subplot_shape, iteration):
    dirToSave = "testResults/"
    if not os.path.exists(dirToSave):
        os.makedirs(dirToSave)
    filePathName = dirToSave + "test_"
    path = ''.join([filePathName, "_", str(iteration).zfill(4), '.png'])
    path_txt = ''.join([filePathName, "_", str(iteration).zfill(4), '.txt'])
    plt.style.use('ggplot')
    fig, axes = plt.subplots(*subplot_shape)
    indx = 0
    for image, ax in zip(images, axes.flatten()):
        reshaped = None
        if image.shape[0] == 3:
            reshaped = np.rollaxis(image, 0,3)
        else:
            reshaped = image.reshape(28, 28)
        ax.imshow(reshaped, vmin=0, vmax=1.0, cmap='gray')
        ax.axis('off')
        indx = indx + 1
    plt.savefig(path, dpi = 100)

def logTensorBoard(trainer, tbWriter, prefix, trainStep):
    # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
    for parameter in trainer.model.parameters:
        tbWriter.write_value("{0}_{1}_{2}{3}".format(prefix, parameter.name, parameter.uid, "/mean"),
                             reduce_mean(parameter).eval(), trainStep)

def save_trained_models(objects, object_labels, ckp_label, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i in range(len(objects)):
        checkpoint_file = os.path.join(model_dir, \
                                       "{}_{}.dnn".format(object_labels[i], ckp_label))
        objects[i].save(checkpoint_file)

def save_generated_images(images, model_name, train_step, images_dir):
    model_images_dir = os.path.join(images_dir, "%s_%d" % (model_name, train_step))
    if not os.path.exists(model_images_dir):
        os.makedirs(model_images_dir)

    for i in range(len(images)):
        # img = images[i].transpose(1,2,0)
        img = images[i].transpose(2, 1, 0)
        img = img * 255
        img_file_path = os.path.join(model_images_dir, "%d.png" % i)
        imsave(img_file_path, img)

        rgb = img
        bgr = rgb[..., ::-1]
        img_file_path = os.path.join(model_images_dir, "%d_bgr.png" % i)
        imsave(img_file_path, bgr)
