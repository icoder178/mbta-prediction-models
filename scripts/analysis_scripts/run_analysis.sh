#!/bin/bash
# script for running just analysis scripts; run in the directory it is in. 
# note that this does not ignore errors like with the master script to make debugging a little easier
echo "Starting model training; this will occupy significant computational resources and take 5-30 minutes"
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "Poisson")
for model in "${models[@]}"
do
  python models.py $model > ../../output/data_appendix_output/$model.out &
done
wait
echo "model training done, outputting final results to output/results/"
python performance_display.py > ../../output/results/performance_summary.txt
echo "output done, analysis script done!"
