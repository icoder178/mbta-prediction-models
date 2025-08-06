#!/bin/bash
# script for running just analysis scripts; run in the directory it is in.
echo "Starting model training; this will occupy significant computational resources and take 1-10 minutes"
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "Poisson")
for model in "${models[@]}"
do
  python -W ignore models.py $model 5 0.8 NO_BOOTSTRAP &
done
wait
echo "model training done, outputting final results to output/results/"
python performance_display.py > ../../output/results/performance_summary.txt
echo "output done, selecting best model and placing in output/data_appendix_output"
python select_best_model.py delay
python select_best_model.py gse
echo "selection done, testing best model, graphing residuals and feature importance, and placing in output/results"
python test_model.py 5 0.8 > ../../output/results/predictor_summary.txt
if [ -z "$1" ]; then
  echo "By default, skipping bootstrap computation and proceeding with pre-computed values."
  echo "Specify COMPUTE_BOOTSTRAP to compute bootstraps from scratch."
  echo "For example, ./master_script.sh NO_ENV COMPUTE_BOOTSTRAP"
  echo "Or ./analysis_script COMPUTE_BOOTSTRAP"
  echo "Be warned this takes very long (often hours) on a standard computer."
else
  echo $1
  python bootstrapping.py
fi
echo "bootstrap computation done, building graphs with confidence intervals"
python bootstrap_display.py > ../../output/results/bootstrap_summary.txt
echo "testing done, analysis script done!"
