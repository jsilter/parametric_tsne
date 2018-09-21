# Overview
This is a python package implementing parametric t-SNE. We train a neural-network to learn a mapping by minimizing the Kullback-Leibler divergence between the Gaussian distance metric in the high-dimensional space and the Students-t distributed distance metric in the low-dimensional space. By default we use similar archictecture<sup>1</sup> as van der Maaten 2009, which is a dense neural network with layers:
[input dimension], 500, 500, 2000, [output dimension]

Simple example usage may be:

```python
train_data = load_training_data_somehow()
high_dims = train_data.shape[1]
num_outputs = 2
perplexity = 30
ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity)
ptSNE.fit(train_data)
output_res = ptSNE.transform(train_data)
```

`output_res` will be `N x num_outputs`, the transformation of each point.
At this point, `ptSNE` will be a trained model, so we can quickly transform other data:

```python
test_data = load_test_data_somehow()
test_res = ptSNE.transform(test_data)
```

See [the example script](./example/example_viz_parametric_tSNE.py) (and its corresponding [readme](./example/README.md)) for more information.

If one wants to use a different network architecture, one must specify the layers.
The neural network is implemented using Keras and Tensorflow, so layers should be specified using Keras:

```python
from tensorflow.contrib.keras import layers
all_layers = [layers.Dense(10, input_shape=(high_dims,), activation='sigmoid', kernel_initializer='glorot_uniform'),
layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_uniform'),
layers.Dense(num_outputs, activation='relu', kernel_initializer='glorot_uniform')]
ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity, all_layers=all_layers)
```

The "perplexity" parameter can also be a list (e.g. [10,20,30,50,100,200]), in which case the total loss function is a sum of the loss function calculated from each perplexity. This is an ad-hoc method inspired by Verleysen et al 2014. Initialization and training step computation time will be linear in the number of perplexity values used, though it shouldn't affect the speed of the final trained model.

If the dimensionality is large (>100), it is recommended that one use a simple dimensionality reduction method first. See van der Maaten's FAQ on tSNE.

# Footnotes

1. van der Maaten 2009 used a ReLu as the output layer. The default here is a linear output layer. ReLu would occasionally produce poor results in the form of all zeroes in one dimension.

# References

van der Maaten, L. (2009). Learning a parametric embedding by preserving local structure. RBM, 500(500), 26.

L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008

John A. Lee, , Diego H. Peluffo-Ordonez, and Michel Verleysen. Multiscale stochastic neighbor embedding: Towards parameter-free dimensionality reduction. ESANN 2014 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium), 23-25 April 2014, ISBN 978-287419095-7. https://pdfs.semanticscholar.org/1e4b/21aca0590d4572a99fa3df3edd453f2d8a5a.pdf

MATLAB Parametric tSNE implementation: https://lvdmaaten.github.io/tsne/code/ptsne.tar.gz
Available at https://lvdmaaten.github.io/tsne/

Mirrored at https://github.com/jsilter/lvdmaaten.github.io/tree/master/tsne/code

Also see the tSNE FAQ: https://lvdmaaten.github.io/tsne/#faq
Archive: http://archive.is/lc4lr


