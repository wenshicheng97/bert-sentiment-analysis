#!/bin/bash

echo "python main.py --train --eval"
python main.py --train --eval

echo "python main.py --train --eval --dataset amazon"
python main.py --train --eval --dataset amazon

echo "python main.py --train --eval --dataset rotten"
python main.py --train --eval --dataset rotten

echo "python main.py --train --eval --dataset yelp"
python main.py --train --eval --dataset yelp

echo "python main.py --train --eval --dataset tweet"
python main.py --train --eval --dataset tweet

echo "python main.py --train_augmented --eval_transformed"
python main.py --train_augmented --eval_transformed

echo "python main.py --train_augmented --eval_transformed --dataset amazon"
python main.py --train_augmented --eval_transformed --dataset amazon

echo "python main.py --train_augmented --eval_transformed --dataset rotten"
python main.py --train_augmented --eval_transformed --dataset rotten

echo "python main.py --train_augmented --eval_transformed --dataset yelp"
python main.py --train_augmented --eval_transformed --dataset yelp

echo "python main.py --train_augmented --eval_transformed --dataset tweet"
python main.py --train_augmented --eval_transformed --dataset tweet
