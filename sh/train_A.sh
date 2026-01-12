#!/bin/zsh
#
# train_A.sh - Training run with larger model configuration
#

sh/train.sh \
    --input txt_local/war_and_peace_sentences.txt \
    --mode sentence \
    --n_layer 12 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 256 \
    --max_iters 50000 \
    --warmup_iters 1000
