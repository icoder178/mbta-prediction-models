#!/bin/bash
# script for running just model.py for all models; run in the directory it is in
echo "Starting model training; this will occupy significant computational resources and take 5-30 minutes"
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "Poisson")
for model in "${models[@]}"
do
  python -W ignore models.py $model > ../../output/data_appendix_output/$model.out &
done
wait
echo "model training done, outputting final results to output/results/"
