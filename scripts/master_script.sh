#!/bin/bash
# master script; running it goes straight from raw data to final results. Run in the directory it is in, or else it will not work.
cd ..
conda deactivate
conda create -n 'mbta-prediction-models' --file environment.txt -y
conda activate 'mbta-prediction-models'
pip install meteostat
cd scripts/processing
python meteostat.py
python data_wrangling.py no_debug no_cheatsheet
cd ../analysis_scripts
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "LSTM" "Poisson")
for model in "${models[@]}"
do
  python models.py $model > ../../output/data_appendix_output/$model.out &
done
wait
python performance_display.py > ../../output/results/performance_summary.txt
