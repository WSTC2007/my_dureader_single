#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nohup /home/taok/anaconda3/bin/python3.6 -u run.py 1>my_dureader.log 2>&1 &