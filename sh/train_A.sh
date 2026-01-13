#!/bin/zsh
#
# train_A.sh - Training run with larger model configuration
#

sh/train.sh \
    --input txt_local/gibbon_war_and_peace_sentences_cleaned_1a.txt \
    --mode sentence \
    --n_layer 12 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 256 \
    --max_iters 50000 \
    --warmup_iters 500 \
    --min_sentence_tokens 17 \
    --tokenizer char

