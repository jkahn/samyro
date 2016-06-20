Installation hints
==================

On Ubuntu Linux, which is where I (jkahn) write code.

Installing in general
---------------------

Probably easiest to do in a conda environment::

  conda create --name samyro-dev-py2 python=2.7
  source activate samyro-dev-py2

or::
  conda create --name samyro-dev-py3 python=3.4
  source activate samyro-dev-py3

Installing Tensorflow
---------------------
Easiest to do by following the instructions here:

https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html


Installing the nvidia cuda dnn libraries
---------------------------------------------

https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#optional-install-cuda-gpus-on-linux

and further. This is pretty painful, and my laptop doesn't have a
GPU. Not bothering with it for now.

Installing Samyro for development
--------------------------
Get from git::
  git clone https://github.com/jkahn/samyro.git
  cd samyro
  git checkout master
  git checkout -b my-feature-name
  pip install -e .[dev,doc,test]
  
  

  
  
