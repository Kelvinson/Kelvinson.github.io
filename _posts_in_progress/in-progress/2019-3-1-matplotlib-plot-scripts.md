---
layout: post
title: Matplotlib Plot Scripts I frequently Use
katex: True
category: tech-blogs
labels: ml, nn
---

### From my classmate:
Last semester, I happen to team with a ML PhD student for course project. He is an awesome guy and has super high work
effeciency. He has many ways to implement algorithms and makes it run over night. I accidently got some plot scripts 
from him. Here is a script to read data from AWS S3 bucket and plot the result with LR 
```python
from urllib2 import urlopen
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

output_file = open('output_log.txt', 'w')
print >> output_file, "This is the output of Logestic Regression on BERT"
print >> output_file, "======================================================="

data_file = urlopen("https://s3.amazonaws.com/bert-1/outfile.out")
X = pd.read_csv(data_file, sep=" ")
X = X.values

label_file = urlopen("https://s3.amazonaws.com/bert-1/labels.out")
y = pd.read_csv(label_file, sep=" ")
y = y.values

# Just splitting the data once for all the different models so we have a common
# benchmark
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                          random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data

accuracy_scores = np.zeros(
  (2, 10))  # store the accuracy scores for different max_iters
values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125]

print >> output_file, "L2 regularization"
print >> output_file, \
"---------------------------------------------------------"

for epoch in range(len(values)):
  clf = SGDClassifier(loss="log", penalty="l2", max_iter=values[epoch], \
            warm_start=True)  # no dim red
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
  accuracy = metrics.accuracy_score(y_test, y_pred)
  f1_score = 2 * ((precision * recall) / (precision + recall))
  accuracy_scores[0, epoch] = accuracy
  print >> output_file, "***This is L2 reg, LR, BERT for max_iter=", epoch, \
  " ***"
  print >> output_file, "Accuracy: ", accuracy
  print >> output_file, "Precision: ", precision
  print >> output_file, "Recall: ", recall
  print >> output_file, "F1 score: ", f1_score
  print >> output_file, "Values of the coefficients is:\n", clf.coef_
  print >> output_file, "Values of the bias (or offset) is:\n", clf.intercept_
  print >> output_file, "\n\n"

print >> output_file, \
"---------------------------------------------------------"
print >> output_file, "L1 regularization"
print >> output_file, \
"---------------------------------------------------------"

for epoch in range(len(values)):
  clf = SGDClassifier(loss="log", penalty="l1", max_iter=values[epoch],
            early_stopping=True, warm_start=True)  # no dim red
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
  accuracy = metrics.accuracy_score(y_test, y_pred)
  f1_score = 2 * ((precision * recall) / (precision + recall))
  accuracy_scores[1, epoch] = accuracy
  print >> output_file, "***This is L1 reg, LR, BERT for max_iter=", epoch, \
  " ***"
  print >> output_file, "Accuracy: ", accuracy
  print >> output_file, "Precision: ", precision
  print >> output_file, "Recall: ", recall
  print >> output_file, "F1 score: ", f1_score
  print >> output_file, "Values of the coefficients is:\n", clf.coef_
  print >> output_file, "Values of the bias (or offset) is:\n", clf.intercept_
  print >> output_file, "\n\n"

print >> output_file, "--------------------------------------------------------"
print >> output_file, "LR, BERT, values:", values
print >> output_file, "LR, BERT, L2 Reg:", accuracy_scores[0, :]
print >> output_file, "LR, BERT, L1 Reg:", accuracy_scores[1, :]

output_file.close()

yp = None
xi = np.linspace(values[0], values[-1], 100)

yi = np.interp(xi, values, accuracy_scores[0, :], yp)
f = plt.figure(1)
plt.plot(values, accuracy_scores[0, :], 'o', xi, yi, '.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('BERT LR L2')

f.savefig('bert_LR_l2.png', bbox_inches='tight')

yi = np.interp(xi, values, accuracy_scores[1, :], yp)
g = plt.figure(2)
plt.plot(values, accuracy_scores[1, :], 'o', xi, yi, '.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('BERT LR L1')

g.savefig('bert_LR_l1.png', bbox_inches='tight')
``` 

And this one for printing out the summary of the output file

```python
# This finds the highest F1 scores for each configuration
import numpy as np
import re

values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125]
bert_svm_f1 = np.zeros((32))
bert_lr_f1 = np.zeros((32))

counter = 0
with open('download/bert/output_svm.txt', 'r') as f:
  for line in f:
    if re.match("F1 score(.*)", line):
      bert_svm_f1[counter] = line.split()[2]
      counter += 1

bert_svm_f1 = np.nan_to_num(bert_svm_f1.reshape((2, 16)))

counter = 0
with open('download/bert/output_log.txt', 'r') as f:
  for line in f:
    if re.match("F1 score(.*)", line):
      bert_lr_f1[counter] = line.split()[2]
      counter += 1

bert_lr_f1 = np.nan_to_num(bert_lr_f1.reshape((2, 16)))

with open('download/f1_scores/summary.txt', 'a') as f:
  print("Summary of best F1 Scores", file=f)
  print("===============================================", file=f)
  print("BERT SVM L1: ", np.amax(bert_svm_f1[1, :]), " at n_iter = ",
      values[np.argmax(bert_svm_f1[1, :])], file=f)
  print("BERT SVM L2: ", np.amax(bert_svm_f1[0, :]), " at n_iter = ",
      values[np.argmax(bert_svm_f1[0, :])], file=f)
  print("BERT LR L1: ", np.amax(bert_lr_f1[1, :]), " at n_iter = ",
      values[np.argmax(bert_lr_f1[1, :])], file=f)
  print("BERT LR L2: ", np.amax(bert_lr_f1[0, :]), " at n_iter = ",
      values[np.argmax(bert_lr_f1[0, :])], file=f)

```

and this grep the score values and plot & save.
```python
# This creates plots for F1 scores for each configuration
import matplotlib.pyplot as plt
import numpy as np
import re

values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 125]
g50_drmm_f1 = np.zeros((32))
g100_drmm_f1 = np.zeros((32))

counter = 0
with open('output_DRMM50.txt', 'r') as f:
  for line in f:
    if re.match("F1 score(.*)", line):
      g50_drmm_f1[counter] = line.split()[2]
      counter += 1

g50_drmm_f1 = np.nan_to_num(g50_drmm_f1.reshape((2, 16)))

counter = 0
with open('output_DRMM100.txt', 'r') as f:
  for line in f:
    if re.match("F1 score(.*)", line):
      g100_drmm_f1[counter] = line.split()[2]
      counter += 1

g100_drmm_f1 = np.nan_to_num(g100_drmm_f1.reshape((2, 16)))

yp = None
xi = np.linspace(values[0], values[-1], 100)

yi = np.interp(xi, values, g50_drmm_f1[0, :], yp)
f = plt.figure(1)
plt.plot(values, g50_drmm_f1[0, :], 'o', xi, yi, '.')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('GLoVe - 50 DRMM')

f.savefig('glove50_DRMM.png', bbox_inches='tight')

yi = np.interp(xi, values, g100_drmm_f1[0, :], yp)
f = plt.figure(1)
plt.plot(values, g100_drmm_f1[0, :], 'o', xi, yi, '.')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('GLoVe - 100 DRMM')

f.savefig('glove100_DRMM.png', bbox_inches='tight')

print(np.amax(g50_drmm_f1))
print(np.amax(g100_drmm_f1))
```
### My own routines:
It turns out that numpy read data fomats other than npz,pickle much more slowly
 than these two formats. So first I will 
convert the data to npz format or pickle format.
 
```python
# extract train data from txt file to npz
# extract train data from txt file to npz
train_data = np.loadtxt("train.txt", delimiter=",", usecols=range(1569))
trainX = train_data[:, :-1]
y_labels = train_data[:, -1].astype(int)

# extract validate data
validate_data = np.loadtxt("val.txt", delimiter=",", usecols=range(1569))
validX = validate_data[:, :-1]
valid_y_labels = validate_data[:, -1].astype(int)

# first save it to npz file
# np.savez('data.npz', train=train_data, valid=validate_data)
# data = np.load('data.npz')
# print data['train']
# print data['valid']

# next save it to pickle binary file
pkl_file = open("serialized_data.pkl", 'wb')
pickle.dump({'train': train_data, 'valid': validate_data}, pkl_file)
# load pickel file
# data = pickle.load( open( "save.p", "rb" ) )

import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *

# extract train data
train_data = np.loadtxt("train.txt", delimiter=",", usecols=range(1569))
trainX = train_data[:, :-1]
y_labels = train_data[:, -1].astype(int)

# extract validate data
validate_data = np.loadtxt("val.txt", delimiter=",", usecols=range(1569))
validX = validate_data[:, :-1]
valid_y_labels = validate_data[:, -1].astype(int)

print("")
# make one-hot encode of the current y data in shape (N, 1)
input_size = 1568
output_size = 19
lr_decay_iter = 20
# y = np.zeros((y_labels.shape[0], output_size))
# y[np.arange(y_labels.shape[0]), y_labels] = 1
y = makeOneHotVector(y_labels, output_size)
valid_y = makeOneHotVector(valid_y_labels, output_size)

ids_to_show = [0, 100, 120]

# parameters in a dictionary
params = {}


def run_once(learning_rate, momentum, hidden, max_iters=300, batch_size=256):
  initial_lr = learning_rate
  # np.random.seed(*2**17 + 31)
  initialize_weights(input_size, hidden, params, 'layer1')
  initialize_weights(hidden, output_size, params, 'output')
  assert (params['Wlayer1'].shape == (input_size, hidden))
  assert (params['blayer1'].shape == (hidden,))

  layers_name = ["layer1", "output"]
  for layer_name in layers_name:
    params['m_W' + layer_name] = np.zeros_like(params['W' + layer_name])
    params['m_b' + layer_name] = np.zeros_like(params['b' + layer_name])

  # estimate a rough of the weights sum
  print("{}, {:.2f}".format(params['blayer1'].sum(),
                params['Wlayer1'].std() ** 2))
  print("{}, {:.2f}".format(params['boutput'].sum(),
                params['Woutput'].std() ** 2))

  # test rightness of sigmoid function implementation
  test = sigmoid(np.array([-1000, 1000]))
  print('test sigmoid implementatin, results should be zero and one\t', \
      test.min(), test.max())
  # implement forward
  h1 = forward(trainX, params, 'layer1')
  print(h1.shape)

  # implement softmax
  probs = forward(h1, params, 'output', softmax)
  # make sure you understand these values!
  # positive, ~1, ~1, (40,4)
  print('probs min, min of sum of probs along axis 1, max of sum of probs \
  along axis 1, probs shape')
  print(probs.min(), min(probs.sum(1)), max(probs.sum(1)), probs.shape)

  # implement compute_loss_and_acc
  print("the shape of y and probs is: ", y.shape, probs.shape)
  # note here loss is the total cross entropy of the N examples
  loss, acc = compute_loss_and_acc(y, probs)

  # should be around -np.log(0.25)*40 [~55] and 0.25
  # if it is not, check softmax!
  print("{}, {:.2f}".format(loss, acc))

  # here we cheat for you
  # the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
  delta1 = probs
  delta1[np.arange(probs.shape[0]), y_labels] -= 1

  # we already did derivative through softmax
  # so we pass in a linear_deriv, which is just a vector of ones
  # to make this a no-op
  delta2 = backwards(delta1, params, 'output', linear_deriv)
  # Implement backwards!
  backwards(delta2, params, 'layer1', sigmoid_deriv)

  # check the size of parameters: W and b should match their gradients sizes
  for k, v in sorted(list(params.items())):
    if 'grad' in k:
      name = k.split('_')[1]
      print(name, v.shape, params[name].shape)

  # get random batches
  batches = get_random_batches(trainX, y, batch_size)  # 5 is batch size, \
  # every batch has 5 pairs of (x, y) tuple
  print("............")
  print("max_iters and learning rate are: ", max_iters, learning_rate)
  print("............")
  average_train_losses = np.zeros(max_iters)  # record the average loss over\
  # training epoches
  average_validation_losses = np.zeros(max_iters)
  train_accs = np.zeros(max_iters)
  validate_accs = np.zeros(max_iters)
  # with default settings, you should get loss < 35 and accuracy > 75%
  for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    total_acc = 0
    for xb, yb in batches:
      # forward
      h1 = forward(xb, params, 'layer1')
      # print(h1.shape)
      # Q 2.2.2
      # implement softmax
      probs = forward(h1, params, 'output', softmax)
      # loss
      loss, acc = compute_loss_and_acc(yb, probs)

      # be sure to add loss and accuracy to epoch totals
      total_loss += loss
      total_acc += acc
      #
      # print(ga)
      label_idx = np.argmax(yb, axis=1)
      delta1 = probs
      delta1[np.arange(probs.shape[0]), label_idx] -= 1

      # we already did derivative through softmax
      # so we pass in a linear_deriv, which is just a vector of ones
      # to make this a no-op
      delta2 = backwards(delta1, params, 'output', linear_deriv)

      backwards(delta2, params, name='layer1',
            activation_deriv=sigmoid_deriv)

      # apply gradient
      for layer_name in layers_name:
        params['m_W' + layer_name] = momentum * params['m_W' +
                                 layer_name] - \
                       learning_rate * \
                                       params[
                                         'grad_W' + layer_name]
        params['m_b' + layer_name] = momentum * params['m_b' +
                                 layer_name] - \
                       learning_rate * \
                                       params[
                                         'grad_b' + layer_name]
        params['W' + layer_name] += params['m_W' + layer_name]
        params['b' + layer_name] += params['m_b' + layer_name]

    # now run a forward pass on the validation set to calculate the loss
    validh = forward(validX, params, 'layer1')
    valid_probs = forward(validh, params, 'output', softmax)
    valid_total_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

    average_train_losses[itr] = total_loss / trainX.shape[0]
    average_validation_losses[itr] = valid_total_loss / validX.shape[0]
    train_acc = total_acc / (len(batches))
    train_accs[itr] = train_acc
    validate_accs[itr] = valid_acc
    if itr % 20 == 0:
      print("itr: {:02d} \t train loss: {:.6f} \t train acc : {:.4f}" \
          .format(itr, average_train_losses[itr], train_acc))
      print("itr: {:02d} \t valid loss: {:.6f} \t valid acc : {:.4f}". \
          format(itr, average_validation_losses[itr], valid_acc))
    if itr % lr_decay_iter == lr_decay_iter - 1:
      learning_rate *= 0.9
  seed = "{}_{}_{}_{}_{}".format(initial_lr, momentum, hidden, max_iters, \
                   batch_size)
  np.savez("stats_results/e_celoss_{}".format(seed), \
       train_loss=average_train_losses,
       validation_loss=average_validation_losses)
  np.savez("stats_results/e_error_{}".format(seed), train_err=train_accs, \
       validation_err=validate_accs)

```

```python
#plot the loss and acc result from npz file
lr = 0.01
momentums = 0.5
hiddens = [20, 100, 200, 500]
for hidden_num in hiddens:
  run_once(lr, momentums, hidden_num)

# plot the loss and acc result from npz file
import numpy as np
import matplotlib.pyplot as plt

train_losses = np.zeros([1, 300])
validation_losses = np.zeros([1, 300])

hiddens = [20, 100, 200, 500]

Xs = []
# Ys = []
losses = {}
errors = {}

for hidden_num in hiddens:
  seed = "{}_{}_{}_{}_{}".format(0.01, 0.5, hidden_num, 300, 256)
  this_loss = np.load("stats_results/e_celoss_{}.npz".format(seed))
  this_error = np.load("stats_results/e_error_{}.npz".format(seed))
  # average_train_losses = data.train_loss
  losses["val_" + str(hidden_num)] = this_loss["validation_loss"]
  losses["train_" + str(hidden_num)] = this_loss["train_loss"]
  errors["val_" + str(hidden_num)] = this_error["validation_err"]
  errors["train_" + str(hidden_num)] = this_error["train_err"]

# for seed in seeds:
#   train_losses[0] = 1 - np.load("train_acc_{}.npy".format(seed))

x = np.arange(1, 301, 1)
plt.figure(1)
ax = plt.subplot(121)
ax.set_xlabel('iterations', fontsize=8)
ax.set_ylabel('avg loss', fontsize=8)
for key in losses.keys():
  plt.plot(x, losses[key], label=key)

ax.legend(loc=0, fontsize="x-small")

ax = plt.subplot(122)
ax.set_xlabel('iterations', fontsize=8)
ax.set_ylabel('error rate', fontsize=8)
for key in errors.keys():
  plt.plot(x, errors[key], label=key)

ax.legend(loc=0, fontsize="x-small")
plt.show()
```

TIPS: legend can be also customized by setting loc, fontsize etc.

***note: I should do some clean instead of pasting the whole code but I promise I will update it soon*** 
