#!/bin/bash


for (( i = 0; i < 10; i++ )); do
  echo $i
  cd $i
  for j in *; do
    fold=seq${j}
    mkdir $fold
    ../gen2.py $j $i $fold
    rm $j
  done
  cd ..
done
