#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=8 train.py /home/ImageNet --model poolformer_s24 -b 128 --lr 2e-3 --drop-path 0.1 --apex-amp

