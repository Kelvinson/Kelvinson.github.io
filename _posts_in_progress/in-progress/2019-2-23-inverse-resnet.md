---
layout: post
title: Invertible Residual Networks
katex: True
---

Paper is here: [Invertible Residual Networks](https://arxiv.org/pdf/1811.00995.pdf)


<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/invertible-resnet/generative-models.png" style="width:500px"/>
<div class="caption" markdown="span">
</div>
</div>


I am always fascinated by the recent advance in deep generative models. From NICE of 2014 to recent Flow, deep generative models achieves
great results in image generation and other unsupervised tasks. Lili has a great [blog](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
particularly for flow based deep generative models. The methods in the following chart are all flow based generative models except the last one.
These flow based methods have to define a sequence of invertible transformation functions.
 
<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/invertible-resnet/flow-methods.png" style="width:500px"/>
<div class="caption" markdown="span">
</div>
</div>

This paper introduces to change the normalization scheme of standard ResNets, which does not affect the classification
performance much but brings the possibilty of invertable transformation and thus the possbility of using ResNet as 
generative modelling tools.



<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/invertible-resnet/i-resnet.png" style="width:500px"/>
<div class="caption" markdown="span">
</div>
</div>

to be continued...