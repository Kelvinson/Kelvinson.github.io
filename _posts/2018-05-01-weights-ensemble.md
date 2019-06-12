---
title: 'Recent Weights Ensembel Techniques in Deep Learning'
date: 2018-05-10
permalink: /posts/2018/05/weights-ensemble/
tags:
  - machine learning
---


Disclaimer: To write this blog I read several blogs from Medium authors [Max Pechyonkin](https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a)
and [Vitaly Bushaev](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b). Thanks to them and the relevant arxiv papers. Also I referto [this](https://mlwave.com/kaggle-ensembling-guide/) beautiful blog.

### Summary 

Ensemble is common way for kagglers to get the best performance by commbining the
performances of different models. In practice, serveral kagglers share their 
own best-performance models to ensemble them to a final best one. Xgboost and 
other boost methods are thus becoming the secret weapons for the final round 
kagglers. However, it is difficult to train so many models in research to boost 
the performance by serveral points or so for the GPU computing is quite expensive.
We still want to take the advantages of ensemble. So the first weights ensemble comes.
Before that, let's look at some relevant techniques in training.

### Before start
*Geometric Views of weights*

At any point of training. The network with the weights in it, or the solution 
we have so far can be viewd as a vector while the inputs can be viewd as a plane.
Our goal is to find such "good" vectors that can seperate the input planes, or find 
vectors which multiplies with the input vectors can lead to positive sign of the plane.
The solution space is concave, thus two such good solutions can combine to make a 
new good solution. This is a base knowledge that we can emsemble our weights to get 
a better one. 

<div class="figure" style="margin-bottom:30px;">
<img src="{{ site.base_url }}/images/weights-ensemble/weights_space.png" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/weights-ensemble/feasible_solutions.png" style="width:340px"/>
<div class="caption" markdown="span" style="margin-top:10px">
**Left:** In weight space, every input is a plane and every set of weights(solution) is a vector(point) in the space. 
**Right:** sum of two good solutions is also a good solution due tot concave properties.
</div>
</div>

*Wide-Sharpe-Minimum*

As in the paper [ON LARGE-BATCH TRAINING FOR DEEP LEARNING:GENERALIZATION GAP AND SHARP MINIMA](https://arxiv.org/pdf/1609.04836.pdf) (I should have read it earlier!). The ability of generation from training set to test set is viewed as the problem whether the minimum is sharpe or wide. Just as the 
figure below, the good(well generalized) solution should be the wider ones.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/weights-ensemble/wide-sharp-minimum.png" style="width:500px"/>
<div class="caption" markdown="span">
The dotted line is the test error function while the real line is the train error function. The error function is 
somewhat divergent with the train error function. Intuitively, the wide local minimum of training loss can also achive
good result in test error function while there is big gap between the training and test loss in the sharp local minimum.
Which shows that the wide local minimum can generalize better and is the solution we want. 
</div>
</div>

*Learning rate schedule*

**You can refer to [this blog](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b) for more details about this part**. We usually use constant learning rates during training. But the problem is that such constant lr can lead to traing stuck at the saddle point. In Leslie N. Smith's ["triangular" learning rate paper](https://arxiv.org/pdf/1702.04283.pdf), a triangle shape learning  rate is proposed to alleviate the problem. Inspired by this, cosine lr up and down also [comes out](arXiv preprint arXiv:1608.03983) and proves to be more effective.

<div class="figure" style="margin-bottom:30px;">
<img src="{{ site.base_url }}/images/weights-ensemble/tri1.png" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/weights-ensemble/tri2.png" style="width:340px"/>
<div class="caption" markdown="span" style="margin-top:10px">
**Left:** In a cycle(a fixed number of iterations), lr goes from low to high and returns to the original low point. 
**Right:** "triangle" learing rate v2, the high points reduces to its half as cycle number goes on.
</div>
</div>

<div class="figure" style="margin-bottom:30px;">
<img src="{{ site.base_url }}/images/weights-ensemble/cos1.png" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/weights-ensemble/cos2.png" style="width:340px"/>
<div class="caption" markdown="span" style="margin-top:10px">
**Left:** In a cycle(a fixed number of iterations), lr goes high to low in a cosine rate and cycles again. 
**Right:** Cycles like left one except the time period of the cycle doubles.
</div>
</div>

### Snapshot Ensembling 
*Ensemble weights of different local minimums*

Unlike other model ensemble methods. The author comes up with the creative idea of increasing the learning rate to escape the current local
minimum instead of training from start over to get another minum. Thus the training cost is cut down sharply. I followed the [Keras code implementation](https://github.com/titu1994/Snapshot-Ensembles) to get a 71.04, 71.78 and 72.24 accuracy on the CIFAR100 dataset using single best model, non-weighted ensemble model, weighted ensemble model respectively.

<div class="figure" style="margin-bottom:30px;">
<img src="{{ site.base_url }}/images/weights-ensemble/sgd_path.png" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/weights-ensemble/sgd_restart_path.png" style="width:340px"/>
<div class="caption" markdown="span" style="margin-top:10px">
**Left:** Single best model falls into a local minimum. 
**Right:** Snapshot ensemble restarts from the minimum to find more minimum by increasing the lr when stuck.
</div>
</div>

### Fast Geometric Ensembling
*Ensemble minimas on the the path*

The authors found that there exists path between the local minimas, on this path the loss stays low. Therefore it can take smaller steps and shorter cycles to find different enough minimas to ensemble with, and produce better results than the Snapshot Ensembling.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/weights-ensemble/path.png" style="width:700px"/>
<div class="caption" markdown="span">
**Left:** By intuition, we need to find a path to get to another minima which crosses a high loss region.
**Middle,Right:** However the author finds that there is a path directly connects these local minimas. On this path the loss stays low and it takes smaller steps and shorter cycles. 
</div>
</div>

### Stochastic Weight Averaging (SWA)
*Only two set of weights to ensemble*

The Author observes that after every cycle, the solution stops at the boarder of the "real global minima", so intuitively just average these solution weights. In practice, the paper gives a formula to let two set of weights to update the averaging. And at last one final weight is used to inference.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/weights-ensemble/swa.png" style="width:700px"/>
<div class="caption" markdown="span">
**Left:** W1, W2 and W3 represent 3 independently trained networks, Wswa is the average of them. **Middle:**  Wswa provides superior performance on the test set as compared to SGD. 
**Right:** Note that even though Wswa shows worse loss during training, it generalizes better.
</div>
</div>

\\[
 \frac{w_{SWA} + n_{models} + w}{n_{models}+1} \to w_{SWA}.
\\]

This is the formula to update the average after get a new minima

**Update on Feb,23: the same group put a new paper "SWAG"**
"A Simple Baseline for Bayesian Uncertainty in Deep Learning", [pdf link](https://arxiv.org/abs/1902.02476)

to be continued....
