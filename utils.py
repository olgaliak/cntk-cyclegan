import matplotlib.pyplot as plt
import os
import numpy as np
from cntk import reduce_mean

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
      #  if indx == 0:
      #      np.savetxt(path_txt, reshaped)
        indx = indx + 1
    plt.savefig(path, dpi = 100)

def logTensorBoard(trainer, tbWriter, prefix, trainStep):
    # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
    for parameter in trainer.model.parameters:
        tbWriter.write_value("{0}_{1}_{2}{3}".format(prefix, parameter.name, parameter.uid, "/mean"),
                             reduce_mean(parameter).eval(), trainStep)