#!/bin/bash

export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')