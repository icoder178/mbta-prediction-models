#!/bin/bash
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "LSTM" "Poisson")
for model in "${models[@]}"
do
  screen -dmS $model bash -c "python models.py $model > ../results/$model.out"
done