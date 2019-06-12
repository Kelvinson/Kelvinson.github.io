---
title: 'Configure Remote Server for Colab'
date: 2018-02-03
permalink: /posts/2018/02/configure-colab/
tags:
  - misc
  - tools
  - tips
---

I set up the server and remote access to it  as the example [here](https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415) and use the crontab reboot command as given. Moreover, I refer [this](https://blog.kovalevskyi.com/gce-deeplearning-images-as-a-backend-for-google-colaboratory-bc4903d24947) to use the server as the backend via my macbook, so now the network mapping becomes Colab-> Chrome on my mac-> localhost:8880(on my mac) -> server_ip: 8888 on my home.

In order to do this. Extra work has to be done, [this](https://blog.kovalevskyi.com/gce-deeplearning-images-as-a-backend-for-google-colaboratory-bc4903d24947) gives an example to dothis using the remote GCE as server backend.  Colab also gives an official document on [how to do this](https://research.google.com/colaboratory/local-runtimes.html). Combining the two. Now I revise the confrontab command to this: 
```shell
@reboot /home/kelvinson/anaconda3/bin/jupyter notebook
 --no-browser 
--port=8888 
--NotebookApp.allow_origin='https://colab.research.google.com' 
--NotebookApp.token=''  --notebook-dir ~/DL/ &
```
***NotebookApp.token is an option about "token used for authenticating first-time connections to the server"***. Also some other useful commands are below:

```shell
 # view users Crontab job
crontab -u userName -l
crontab -u vivek -l
```


