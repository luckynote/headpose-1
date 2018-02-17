import numpy as np
import matplotlib.pyplot as plt
import utils

def plot_weights(weights, name, path):

    channels = weights.shape[2]
    grid_r, grid_c = utils.get_grid_dim(int(channels))
    fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]))

    w_min = np.min(weights)
    w_max = np.max(weights)

    for i, ax in enumerate(axes.flat):
        img = weights[:, :, i]
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(path + name, bbox_inches = 'tight')

def plot_outputs(outputs, name, path):

    channels = outputs.shape[3]
    grid_r, grid_c = utils.get_grid_dim(int(channels))
    fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]))

    w_min = np.min(outputs)
    w_max = np.max(outputs)

    for i, ax in enumerate(axes.flat):
        img = outputs[0, :, :, i]
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(path + name, bbox_inches = 'tight')

def shuffle_arrays(a, b, seed):
    assert len(a) == len(b)
    np.random.seed(seed)
    np.random.shuffle(a)
    np.random.seed(seed)
    np.random.shuffle(b)

def rerange(a, nmin, nmax):

    assert nmin < nmax

    b = a

    min = b[0]
    max = b[0]

    for i in range(1, len(b)):
        if b[i] < min:
            min = b[i]
        else:
            if b[i] > max:
                max = b[i]

    for i in range(0, len(b)):
        b[i] = ((nmax - nmin)/(max - min))*(b[i] - min) + nmin

    return b

def normalize_set(a, in_avg = None, in_std = None):

    avg = np.mean(a)
    std = np.std(a)

    if(in_avg != None and in_std != None):
        a = (a - in_avg)/in_std
    else:
        a = (a - avg)/std
        return avg, std

def compare(pred, real):

    for i in range(0, len(pred)):
        print("Tilt:")
        print("Predicho: %f Real: %f" % (pred[i, 0], real[i, 0]))
        print("Pan:")
        print("Predicho: %f Real: %f\n" % (pred[i, 1], real[i, 1]))

def mean_error(pred, real):

    err1 = np.sum(abs(pred[:, 0] - real[:, 0])) / len(pred)
    err2 = np.sum(abs(pred[:, 1] - real[:, 1])) / len(pred)

    return err1, err2

def print_history(losses, val_losses):

    print("Loss: Val_Loss:")

    for i in range(0, len(losses)):
        print(losses[i], val_losses[i])

def test_model(model, x1, y1, x2, y2, epochs, verbose = 1):

    model.fit(x1, y1, batch_size = 100, epochs = epochs, verbose = verbose)
    score = model.evaluate(x2, y2)
    pred = model.predict(x2)
    real = y2

    pred[:,0] = pred[:,0]*90
    pred[:,1] = pred[:,1]*90

    real[:,0] = real[:,0]*90
    real[:,1] = real[:,1]*90

    [err1, err2] = mean_error(pred, real)

    return err1, err2, score