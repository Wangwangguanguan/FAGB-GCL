#!/bin/sh

python GB_train.py \
    --dataset=cora  \
    --SEED=0 \
    --K=100 \
    --lr_p=0.001 \
    --wd_p=0.00001 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=1.2 \
    --l1=0.1 \
    --metric='jaccard' \
    --optimizer='adam' \
    --drop_edge_rate=0.1 \
    --drop_kg_edge_rate=0.2

python GB_train.py \
    --dataset=citeseer \
    --SEED=0 \
    --K=1 \
    --lr_p=0.0001 \
    --wd_p=0.0001 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=1.5 \
    --l1=0.01 \
    --metric='jaccard'
