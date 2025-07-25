#!/bin/bash
# master script; running it goes straight from raw data to final results. Run in the directory it is in, or else it will not work.
cd ..
if [ -z "$1" ]; then
  echo "No anaconda path provided. Try:"
  echo "/opt/anaconda3/bin/conda if you have installed Anaconda system-wide on Mac."
  echo "NO_ENV if the environment for this analysis is already up and running."
  exit 1
fi
if [ "$1" != "NO_ENV" ]; then
    echo "building conda environment"
    eval "$($1 shell.bash hook)" > /dev/null
    conda deactivate > /dev/null
    conda create -n mbta-prediction-models --file environment.txt -y > /dev/null
    conda activate mbta-prediction-models > /dev/null
    pip install --force-reinstall meteostat > /dev/null
fi
echo "conda environment done, starting data loading"
cd scripts/processing_scripts
python meteostat_import.py
echo "data loading done, starting data processing; this may take a minute"
python data_wrangling.py no_debug no_cheatsheet
cd ../analysis_scripts
echo "data wrangling done, starting model training; this will occupy significant computational resources and take 5-30 minutes"
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "LSTM" "Poisson")
for model in "${models[@]}"
do
  python models.py $model > ../../output/data_appendix_output/$model.out &
done
wait
echo "model training done, outputting final results to output/results/"
python performance_display.py > ../../output/results/performance_summary.txt
echo "output done, master script done!"
