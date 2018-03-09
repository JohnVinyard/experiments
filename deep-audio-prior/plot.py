from matplotlib import pyplot as plt
import os
import re


def plot_losses(losses):
    plt.figure()
    handles = []
    for k, v in losses.iteritems():
        _id, upsampling_name = k
        handle, = plt.plot(
            v, label='{_id}_{upsampling_name}'.format(**locals()))
        handles.append(handle)
    legend = plt.legend(handles=handles)
    with open('samples/losses.png', 'wb') as f:
        plt.savefig(
            f,
            bbox_extra_artists=(legend,),
            bbox_inches='tight',
            pad_inches=0,
            format='png')


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def plot_gradients(gradients):
    for experiment_name, grads in gradients.iteritems():
        _id, upsampling_name = experiment_name
        plt.figure()
        for g in grads.itervalues():
            plt.semilogy(g)
        filename = '{_id}_{upsampling_name}_gradients.png'.format(**locals())
        filename = get_valid_filename(filename)
        with open(os.path.join('samples', filename), 'wb') as f:
            plt.savefig(f, bbox_inches='tight', pad_inches=0, format='png')
