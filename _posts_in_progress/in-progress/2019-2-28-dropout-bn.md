---
layout: post
title: Digest of Implementing BN and Dropout
katex: True
category: tech-blogs
labels: ml, nn
---

According to Ng's course, the bn layer is usually put right after the FC layer and before the 
activation layer. Its main effect is to reduce the amount of covarience shift of its input,
He uses below explainations:
1) we have  a bn layer after $z^{2}$ and we want to learn the parameters for this bn layer,
if we cover the left part of the bn layer, we learn to map the input to the bn layer to the output. 

2) if we uncover the the left part before the BN layer, we find we are also learning parameters in the left part
networks, i.e. the weights of the networks before the BN are constantly changing. Which will affect the
learning in the part 1). What BN does is to reduce such effect. The input to the BN layer may change much but 
they have same mean and variance, i.e. the BN parameters we are learning $\gamma$ and $\beta$. 
"it weakens the coupling between what the early layers parameters has to do and what the later layers parameters have to do."
in Ng's words, and thus speed up the learning process.

3) the BN also introduces little regularization, if we do mini-batch training. Bcause the mini-batch are
drawn randomly, and each mini-batch is scaled  by the mean and variance computed on that mini-batch, their mean and variance are also noisy for the learning process, 
thus cause regularization to the learning process. 

### Inference

during the inference, since we usually test a single example one time, the batch mean and average computed is meaningless.
Instead, we use the statistics in the training data. Specifically, we keep track of the mean and variance of those training
mini-batches and use exponential moving average to regess the current mean and variance as the test mean and variance.

### References

* (https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
* (https://wiseodd.github.io/techblog/2016/07/04/batchnorm/)
* (https://kevinzakka.github.io/2016/09/14/batch_normalization/)
* (http://cthorey.github.io./backpropagation/)