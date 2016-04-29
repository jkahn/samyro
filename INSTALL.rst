Installation hints
==================

On Ubuntu Linux, which is where I (jkahn) write code.

Installing the nvidia cuda dnn libraries
---------------------------------------------

https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#optional-install-cuda-gpus-on-linux

and further. This is pretty painful, and my laptop doesn't have a
GPU. Not bothering with it for now.

Vanilla install (python 3.4)
----------------------------

Install a Conda environment::
  conda create --name samyro-dev-py3 python=3.4
  source activate samyro-dev-py3

  pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl

  pip install samyro

Installing for development
--------------------------
Get from git::
  git clone https://github.com/jkahn/samyro.git
  cd samyro
  git checkout master
  git checkout -b my-feature-name
  pip install -e .[dev,doc,test]
  
  

  
  
