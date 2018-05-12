#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
nohup /home/taok/anaconda3/bin/python3.6 -u run.py 1>my_dureader_single_zhidao.log 2>&1 &