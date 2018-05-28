#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
nohup /home/taok/anaconda3/bin/python3.6 -u run.py 1>test_search.log 2>&1 &