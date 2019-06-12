---
title: 'Intrinsically Motivated Goal Exploration'
date: 2019-02-24
permalink: /posts/2019/02/goal-space-intrinsic/
tags:
  - rl
---

Unsupervised Learning of goal spaces for intrinsically motivated goal exploration, [paper link](https://arxiv.org/abs/1803.00781)

### Intrisically Motivated Goal Exploration Process.


First let's get familiar with some elements of IMGEP. 
- A context c, element of a Context Space C. This context represents the initial 
experimental factors that are not under the robotic agent control. In most cases, the context is considered fully observable (e.g. state of the world as measured by sensors).
-  A parameterization $\theta$, element of a Parameterization Space $\Theta$. This 
parameterization 
rep- resents the experimental factors that can be controlled by the robotic agent (e.g. parameters of a policy).
- An outcome $o$, element of an Outcome Space $O$. The outcome contains information 
qualifying properties of the phenomenon during the execution of the experiment (e.g. measures characterizing the trajectory of raw sensor observations during the experiment).
- A phenomenon dynamics $D : C, \Theta \rightarrow O$, which in most interesting cases is unknown. 

In Developmental Robotics(simply speaking, view robotics from view of childern's 
cognitive development), we have
- forward model: $\tilde{D}: C  \times \Theta \rightarrow O$
- inverse model: $I: C \times O \rightarrow \Theta$ 
with tuples of ${(c, \theta, o)}$,we can learn a model. Random  Parametrization 
Exploration just samples $\theta \tilde \mu(\theta)$. In order to sample more efficiently, 
following elements are introduced:
- A Goal Space $\Tau$ whose elements $\tau$ represent parameterized goals that can be 
targeted by
 the autonomous agent. In the context of this article, and of the IMGEP-UGL architecture, we consider the simple but important case where the Goal Space is equated with the Outcome space. Thus, goals are simply vectors in the outcome space that describe target properties of the phenomenon that the learner tries to achieve through actions.
- A Goal Policy $\gamma(\tau)$, which is a probability distribution over the Goal Space 
used 
for 
sam-pling goals (see Algorithmic Architecture 2). It can be stationary, but in most cases, it will be updated over time following an intrinsic motivation strategy. Note that in some cases, this Goal Policy can be conditioned on the context γ(τ|c).
- A set of Goal-parameterized Cost Functions $C_\tau : O \rightarrow R$ defined over all
 O, which 
maps
 every outcome with a real number representing the goodness-of-fit of the outcome o regarding the goal τ . As these cost functions are defined over O, this enables to compute the cost of a policy for a given goal even if the goal is imagined after the policy roll-out. Thus, as IMGEPs typically memorize the population of all executed policies and their outcomes, this enables reuse of experimentations across multiple goals.
- A Meta-Policy $\Pi : T , C \rightarrow \Theta$ which is a mechanism to approximately 
solve the mini-mization problem $\Pi(\tau, c) = argmin_\theta C_\tau (\tilde{D}(θ, c)
)$, 
where $\tilde{D}$̃ 
is a 
running forward model (approximating D), trained on-line during exploration.

Q: what is the difference of cost function $C_\tau$ and Exploration Performance 
Measuremnt **Kullback-Leibler Coverage (KLC)**? \\
**Kullback-Leibler Coverage (KLC)** 

## Disentangled Goal Space
Curiosity Driven Exploration of Learned Disentangled Goal Spaces, [paper link](https://arxiv.org/abs/1807.01521)

The second paper further investigate the problem when in more complex environments there are multiple objects and 
distractor, the structure of the goal space has to reflect that of the environment.

Practically, the paper applies $\beta-$VAE to represent the disentangled goal space for
 its good disentangled properties(from the paper, actually I don't quite understand). 
 **disentangled: 1. free (something or someone) from an entanglement; 2.remove knots or
 tangles from (wool, rope, or hair)** 
 To evaluate the such representation by $\beta-$VAE: 1) learn a representation 
 algorithm of $\tilde{R}: x \rightarrow o$ which converts passive observations to outcome. 2) 
 use the representation (latent variable in the algorithm A) to interact with the world,
 by sampling goals that provide high learning progress, and where goals are target values
 of one or several latent variables to be reached through action.
