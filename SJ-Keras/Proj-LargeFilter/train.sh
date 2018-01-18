#!/bin/bash

for k in 3 5 7 9 11 13
do
    python training.py --kernel_size=$k
done
