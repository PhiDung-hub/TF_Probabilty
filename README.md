# TensorFlow Probability introduction & exercises

## Step-by-step guides to explore basic features of Tensorflow Probability: 

1. Install TensorFlow Probability: `pip install tensorflow-probability`.

2. Import it: `import tensorflow_probability as tfp`.

3. Distributions: TensorFlow Probability provides a wide range of distributions that you can use for different purposes (Uni/Multi-Variate, Discrete, Continous, Complex [mixture models, hidden Markov], Special [Exponential, Beta, Chi-Squared, Gamma], and Probabilistic Layers [for Variational Autoencoders, Bayes Network, ..]) 

4. Working with Probabilistic Layers: TensorFlow Probability provides a set of probabilistic layers that you can use to build probabilistic models. For example, you can use a `DenseVariational` layer to build a variational autoencoder.

5. Sampling from Distributions: For example, `tfp.distributions.Normal.sample` method to sample from a normal distribution.

6. Visualizing Distributions: For example, `tfp.distributions.Normal.prob` method to visualize the probability density function of a normal distribution.

7. Training Techniques: TensorFlow Probability provides a variety of techniques for training probabilistic models, including maximum likelihood, variational inference, and Monte Carlo methods.

## Details in the walkthrough notebooks (Imperial College of London): 

### Week 1: Naive Bayes & Logistic regression
Construct a Naive Bayes Classifier using Logits loss for the Iris Dataset. Explore different plotting techniques

### Week 2: Bayes Convolutional Neural Networks
Construct a Bayes CNN for the MNIST and MNIST corrupted (with added noise) dataset.

**Part 1**: turn a CNN into a probabilistic CNN (with added aletoric uncertainty) -> output a distribution.

**Part 2**: Create a Bayes CNN, with added uncertainty for Dense and Convo2D layer -> DenseVariational and Cono2DReparameterization.

### Week 3: RealNVP model
An implementation of [Real-valued non-volume preserving](https://arxiv.org/abs/1605.08803) for the LSUN bedroom dataset, with normalising flow architecture from scratch, including the affine coupling layers and combining into a multiscale architecture.

Important concepts:

1. **Bijectors**: Implement forward and inverse transformations, as well as log Jacobian determinants, to help with probability density transformation between different spaces

2. **Affine coupling**: Stacking bijectors to create a revisible map between different probabilistic spaces (i.e. normalizing [probabilistic] flow)

### Week 4: Variational Auto-encoder
Create a Variational Auto-encoder architecture to generate/alter images of celebrities in the CelebA dataset. Model utilize gaussian mixture distribution.

