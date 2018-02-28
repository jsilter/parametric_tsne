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
from matplotlib.backends.backend_pdf import PdfPages

cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)
from parametric_tSNE import Parametric_tSNE
from parametric_tSNE.utils import get_multiscale_perplexities

has_sklearn = False
try:
    from sklearn.decomposition import PCA
    has_sklearn = True
except Exception as ex:
    print('Error trying to import sklearn, will not plot PCA')
    print(ex)
    pass
    
def _gen_cluster_centers(num_clusters, top_cluster_size):
    # Make two sets of points, to have local and global distances
    cluster_centers = np.zeros([num_clusters, num_clusters])
    cluster_centers[0:top_cluster_size, 0:top_cluster_size] = 1.0
    cluster_centers[top_cluster_size::, top_cluster_size::] = 1.0
    cluster_centers[np.diag_indices(num_clusters)] *= -1
    cluster_centers *= top_cluster_size
    
    return cluster_centers
    
def _gen_hollow_spheres(num_clusters, num_samps, num_rand_points=0):
    top_cluster_size = min([5, num_samps])
    cluster_centers = _gen_cluster_centers(num_clusters, top_cluster_size)
    cluster_assignments = np.arange(0, num_samps) % num_clusters
    
    per_samp_centers = cluster_centers[cluster_assignments, :]
    
    radii = 0.5*np.ones([num_clusters])
    # Make two sets, have second set be larger spheres
    radii[top_cluster_size::] = 1.5
    
    cluster_radii = radii[cluster_assignments]
    # Add a little noise to the radius
    cluster_radii += np.random.normal(loc=0.0, scale=0.05, size=num_samps)
    
    # Add high variance to a subset of points, to simulate noise
    for xx in range(num_rand_points):
        rand_ind = np.random.randint(len(cluster_radii))
        cluster_radii[rand_ind] = np.random.uniform(low=0.05, high=10.0)
        per_samp_centers[rand_ind, :] += np.random.normal(loc=0.0, scale=10.0, size=cluster_centers.shape[1])
        
    #Apparently normally distributed points will be uniform
    #across the surface of a sphere
    init_points = np.random.normal(loc=0.0, scale=1.0, size=[num_samps, num_clusters])
    # Regenerate any points too close to the origin
    min_rad = 1e-3
    init_radii = np.linalg.norm(init_points, axis=1)
    bad_points = np.where(init_radii < min_rad)[0]
    num_bad_points = len(bad_points)
    while num_bad_points >= 1:
        init_points[bad_points, :] = np.random.normal(loc=0.0, scale=1.0, 
            size=[num_bad_points, num_clusters])
        init_radii = np.linalg.norm(init_points, axis=1)
        bad_points = np.where(init_radii < min_rad)[0]
        num_bad_points = len(bad_points)
    
    init_points = init_points / init_radii[:, np.newaxis]
    
    final_points = init_points * cluster_radii[:, np.newaxis]
    #final_radii = np.linalg.norm(final_points, axis=1)
    # Center spheres on different points
    final_points += per_samp_centers
    
    return final_points, cluster_assignments
        
    
def _gen_dense_spheres(num_clusters, num_samps, num_rand_points=0):
    """ Generate `num_clusters` sets of dense spheres of points, in
    `num_clusters` - dimensonal space. Total number of points = `num_samps`"""
    # Make two sets of points, to have local and global distances
    top_cluster_size = min([5, num_samps])
    cluster_centers = _gen_cluster_centers(num_clusters, top_cluster_size)
    
    pick_rows = np.arange(0, num_samps) % num_clusters
    scales = 1.0 + 2*(np.array(pick_rows, dtype=float) / num_clusters)
    
    test_data = cluster_centers[pick_rows, :]
    
    # Add high variance to a subset of points, to simulate points
    # not belonging to any cluster 
    for xx in range(num_rand_points):
        rand_ind = np.random.randint(len(scales))
        scales[rand_ind] = 10.0
    
    # Loop through so as to provide a difference variance for each cluster
    for xx in range(num_samps):
        test_data[xx, :] += np.random.normal(loc=0.0, scale=scales[xx], size=num_clusters)
    
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
        centroid = output_res[cur_plot_rows, :].mean(axis=0)
        plt.annotate('%s' % ci, xy=centroid, xycoords='data', alpha=0.5,
                     horizontalalignment='center', verticalalignment='center')
    

if __name__ == "__main__":
    # Parametric tSNE example
    num_clusters = 14
    model_path_template = 'example_viz_{model_tag}_{test_data_tag}.h5'
    figure_template = 'example_viz_tSNE_{test_data_tag}.pdf'
    override = True
    
    num_samps = 1000
    do_pretrain = True
    epochs = 20
    batches_per_epoch = 8
    batch_size = 128
    plot_pca = has_sklearn
    color_palette = sns.color_palette("hls", num_clusters)
    test_data_tag = 'hollow'
    #test_data_tag = 'dense'
    
    debug = False
    if debug:
        model_path_template = 'example_viz_debug_{model_tag}_{test_data_tag}.h5'
        figure_template = 'example_viz_debug_{test_data_tag}.pdf'
        num_samps = 400
        do_pretrain = False
        epochs = 5
        plot_pca = False
        override = True
    num_rand_points = int(num_samps / num_clusters)
    
    num_outputs = 2
    
    alpha_ = num_outputs - 1.0
    
    if test_data_tag == 'dense':
        _gen_test_data = _gen_dense_spheres
    elif test_data_tag == 'hollow':
        _gen_test_data = _gen_hollow_spheres
    else:
        raise ValueError('Unknown test data tag {test_data_tag}'.format(test_data_tag=test_data_tag))
    
    # Generate "training" data
    np.random.seed(12345)
    train_data, pick_rows = _gen_test_data(num_clusters, num_samps, num_rand_points)
    # Generate "test" data
    np.random.seed(86131894)
    test_data, test_pick_rows = _gen_test_data(num_clusters, num_samps, num_rand_points)

    transformer_list = [{'label': 'Multiscale tSNE', 'tag': 'tSNE_multiscale', 'perplexity': None, 'transformer': None},
                        {'label': 'tSNE (Perplexity=10)', 'tag': 'tSNE_perp10', 'perplexity': 10, 'transformer': None},
                        {'label': 'tSNE (Perplexity=100)', 'tag': 'tSNE_perp100', 'perplexity': 100, 'transformer': None},
                        {'label': 'tSNE (Perplexity=500)', 'tag': 'tSNE_perp500', 'perplexity': 500, 'transformer': None}]
    
    for tlist in transformer_list:
        perplexity = tlist['perplexity']
        if perplexity is None:
            perplexity = get_multiscale_perplexities(2*num_samps)
            print('Using multiple perplexities: %s' % (','.join(map(str, perplexity))))
            
        ptSNE = Parametric_tSNE(train_data.shape[1], num_outputs, perplexity,
                            alpha=alpha_, do_pretrain=do_pretrain, batch_size=batch_size,
                            seed=54321)

        model_path = model_path_template.format(model_tag=tlist['tag'], test_data_tag=test_data_tag)
    
        if override or not os.path.exists(model_path):
            ptSNE.fit(train_data, epochs=epochs, verbose=1)
            print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
            ptSNE.save_model(model_path)
        else:
            print('{time}: Loading from {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
            ptSNE.restore_model(model_path)

        tlist['transformer'] = ptSNE
    

    if plot_pca:
        pca_transformer = PCA(n_components=2)
        pca_transformer.fit(train_data)
        transformer_list.append({'label': 'PCA', 'tag': 'PCA', 'transformer': pca_transformer})
        
    pdf_obj = PdfPages(figure_template.format(test_data_tag=test_data_tag))
    
    for transformer_dict in transformer_list:
        transformer = transformer_dict['transformer']
        tag = transformer_dict['tag']
        label = transformer_dict['label']
        
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

        plt.title('{label:s} Transform with {num_clusters:d} clusters\n{test_data_tag:s} Data'.format(label=label, num_clusters=num_clusters, test_data_tag=test_data_tag.capitalize()))
        
        if pdf_obj:
            plt.savefig(pdf_obj, format='pdf')
            
    if pdf_obj:
        pdf_obj.close()
    else:
        plt.show()
