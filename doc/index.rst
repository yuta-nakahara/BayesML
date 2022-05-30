.. BayesML documentation master file, created by
   sphinx-quickstart on Mon Feb 21 21:21:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BayesML's Documentation
=======================

Purpose
-------

BayesML is a library designed for promoting research, education, and application of machine learning based on Bayesian statistics and Bayesian decision theory. Through these activities, BayesML aims to contribute to society.

Characteristics
---------------

BayesML has the following characteristics.

* The structure of the library reflects the philosophy of Bayesian statistics and Bayesian decision theory: updating the posterior distribution learned from the data and outputting the optimal estimate based on the Bayes criterion.
* Many of our learning algorithms are much faster than general-purpose Bayesian learning algorithms such as MCMC methods because they effectively use the conjugate property of a probabilistic data generative model and a prior distribution. Moreover, they are suitable for online learning.
* All packages have methods to visualize the probabilistic data generative model, generated data from that model, and the posterior distribution learned from the data in 2~3 dimensional space. Thus, you can effectively understand the characteristics of probabilistic data generative models and algorithms through the generation of synthetic data and learning from them.

Installation
------------

Please use the following command to install BayesML.

.. code-block::

   pip install bayesml

The following are required.

* Python (>= 3.7)
* NumPy (>= 1.20)
* SciPy (>= 1.7)
* MatplotLib (>= 3.5)

Example
-------

We show an example of generating data drawn according to the Bernoulli distribution and learning from them.

First, we create an instance of a probabilistic data generative model. Here, the parameter $\theta$, which represents an occurrence probability of 1, is set to 0.7.

.. code-block::

   from bayesml import bernoulli
   gen_model = bernoulli.GenModel(theta=0.7)

You can visualize the characteristics of the created model by the following method.

.. code-block::

   gen_model.visualize_model()

Outputs:

   | theta:0.7  
   | x0:[1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0]  
   | x1:[1 0 1 0 1 0 0 0 0 1 1 1 1 1 1 1 1 0 1 1]  
   | x2:[1 0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 1 0 1 1]  
   | x3:[0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0]  
   | x4:[1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0]  
   .. image:: ./images/README_ex_img1.png

After confirming that the frequency of occurrence of 1 is around $\theta$=0.7, we generate a sample and store it to variable x.

.. code-block::

   x = gen_model.gen_sample(sample_size=20)


Next, we create an instance of a model for learning posterior distribution.

.. code-block::

   learn_model = bernoulli.LearnModel()


A method to visualize the posterior distribution also exists (the prior distribution is shown here because learning from data has not been performed yet).

.. code-block::

   learn_model.visualize_posterior()

Outputs:

   .. image:: ./images/README_ex_img2.png

After learning from the data, we can see that the density of the posterior distribution is concentrated around the true parameter $\theta$=0.7.

.. code-block::

   learn_model.update_posterior(x)
   learn_model.visualize_posterior()

Outputs:

   .. image:: ./images/README_ex_img3.png

In Bayesian decision theory, the optimal estimator under the Bayes criterion is derived as follows. First, we set a loss function, e.g., a squared-error loss, absolute-error loss, and 0-1 loss. Then, the Bayes risk function is defined by taking the expectation of the loss function with respect to the distribution of data and parameters. By minimizing the Bayes risk function, we obtain the optimal estimator under the Bayes criterion. For example, if we set a squared-error loss, the optimal estimator under the Bayes criterion of the parameter `theta` is the mean of the posterior distribution.

In BayesML, the above calclulation is performed by the following methods.

.. code-block::

   print(learn_model.estimate_params(loss='squared'))
   print(learn_model.estimate_params(loss='abs'))
   print(learn_model.estimate_params(loss='0-1'))

Outputs:

   | 0.6428571428571429  
   | 0.6474720009710451  
   | 0.6578947368421053  

Different settings of a loss function yield different optimal estimates.

Package list
------------

The following packages are currently available. In this library, a probabilistic data generative model, prior distribution, posterior distribution (or approximate posterior distribution), and predictive distribution (or approximate predictive distribution) are collectively called a model.

* :doc:`bayesml`
* :doc:`bayesml.autoregressive`
* :doc:`bayesml.bernoulli`
* :doc:`bayesml.exponential`
* :doc:`bayesml.linearregression`
* :doc:`bayesml.multivariate_normal`
* :doc:`bayesml.normal`
* :doc:`bayesml.poisson`

Citation
--------

When you use BayesML for your academic work, please provide the following bibliographic reference.

Plain text

.. code-block::

   Y. Nakahara, N. Ichijo, K. Shimada, 
   K. Tajima, K. Horinouchi, L. Ruan, 
   N. Namegaya, R. Maniwa, T. Ishiwatari, 
   W. Yu, Y. Iikubo, S. Saito, 
   K. Kazama, T. Matsushima, ``BayesML,'' 
   [Online] https://github.com/yuta-nakahara/BayesML


BibTeX

.. code-block::
   
   @misc{bayesml,
   author = {Nakahara Yuta and Ichijo Naoki and Shimada Koshi and
               Tajima Keito and Horinouchi Kohei and Ruan Luyu and
               Namegaya Noboru and Maniwa Ryota and Ishiwatari Taisuke and
               Yu Wenbin and Iikubo Yuji and Saito Shota and Kazama Koki and
               Matsushima Toshiyasu}
   title = {BayesML},
   howpublished = {\url{https://github.com/yuta-nakahara/BayesML}},
   year = {2022}
   }

Contents
--------

.. toctree::
   :maxdepth: 1
   
   bayesml
   developers

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
