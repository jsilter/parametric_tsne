#!/usr/bin/python
from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__= """ Example usage of parametric_tSNE. 
Generate some simple data in high (14) dimension, train a model, 
and run additional generated data through the trained model"""

import sys
import datetime
import os

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)
from parametric_tSNE import Parametric_tSNE

has_sklearn = False
try:
    from sklearn.decomposition import PCA
    has_sklearn = True
except Exception as ex:
    print('Error trying to import sklearn, will not plot PCA')
    print(ex)
    pass


def _gen_test_data(num_clusters, num_samps):
    cluster_centers = 5.0*np.identity(num_clusters)
    perm_vec = np.array([x % num_clusters for x in range(1, num_clusters+1)])
    cluster_centers += cluster_centers[perm_vec, :]
    pick_rows = np.arange(0, num_samps) % num_clusters
    
    test_data = cluster_centers[pick_rows, :]
    test_data += np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
    
    return test_data, pick_rows


def _plot_scatter(output_res, pick_rows, color_palette, alpha=0.5, symbol='o'):
    num_clusters = len(set(pick_rows))
    for ci in range(num_clusters):
        cur_plot_rows = pick_rows == ci
        cur_color = color_palette[ci]
        plt.plot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], symbol, 
        color=cur_color, label=ci, alpha=alpha)
        

def _plot_kde(output_res, pick_rows, color_palette, alpha=0.5):
    num_clusters = len(set(pick_rows))
    for ci in range(num_clusters):
        cur_plot_rows = pick_rows == ci
        cur_cmap = sns.light_palette(color_palette[ci], as_cmap=True)
        sns.kdeplot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], cmap=cur_cmap, shade=True, alpha=alpha,
            shade_lowest=False)
    

if __name__ == "__main__":
    # Parametric tSNE example
    num_clusters = 14
    model_path = 'example_viz_tSNE.h5'
    override = False
    
    num_samps = 2000
    do_pretrain = True
    epochs = 20
    batches_per_epoch = 8
    batch_size = 128
    plot_pca = has_sklearn
    
    debug = False
    if debug:
        model_path = 'example_viz_tSNE_debug.h5'
        num_samps = 400
        do_pretrain = False
        epochs = 5
        plot_pca = False
    
    num_outputs = 2
    
    alpha_ = num_outputs - 1.0
    
    # Generate "training" data
    np.random.seed(12345)
    train_data, pick_rows = _gen_test_data(num_clusters, num_samps)
    # Generate "test" data
    np.random.seed(86131894)
    test_data, test_pick_rows = _gen_test_data(num_clusters, num_samps)
    
    perplexity = 30
    
    ptSNE = Parametric_tSNE(train_data.shape[1], num_outputs, perplexity,
                            alpha=alpha_, do_pretrain=do_pretrain, batch_size=batch_size,
                            seed=54321)
    
    if override or not os.path.exists(model_path):
        ptSNE.fit(train_data, epochs=epochs, verbose=1)
        print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
        ptSNE.save_model(model_path)
    else:
        print('{time}: Loading from {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
        ptSNE.restore_model(model_path)
    
    color_palette = sns.color_palette("hls", num_clusters)
    
    transformer_list = [{'tag': 'tSNE', 'transformer': ptSNE}]
    if plot_pca:
        pca_transformer = PCA(n_components=2)
        pca_transformer.fit(train_data)
        transformer_list.append({'tag': 'PCA', 'transformer': pca_transformer})
    
    for transformer_dict in transformer_list:
        transformer = transformer_dict['transformer']
        tag = transformer_dict['tag']
        
        output_res = transformer.transform(train_data)
        test_res = transformer.transform(test_data)
    
        plt.figure()
        # Create a contour plot of training data
        _plot_kde(output_res, pick_rows, color_palette, 0.5)
        
        # Scatter plot of test data
        _plot_scatter(test_res, test_pick_rows, color_palette, alpha=0.1, symbol='*')
        
        leg = plt.legend(bbox_to_anchor=(1.0, 1.0))
        # Set marker to be fully opaque in legend
        for lh in leg.legendHandles: 
            lh._legmarker.set_alpha(1.0)

        plt.title('{tag} Transform with {num_clusters:d} clusters'.format(tag=tag, num_clusters=num_clusters))

        plt.savefig('example_viz_{tag}.png'.format(tag=tag))
    plt.show()
