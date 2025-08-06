#!/bin/bash
# master script; running it goes straight from raw data to final results. Run in the directory it is in, or else it will not work.
if [ -z "$1" ]; then
  echo "No anaconda path provided. Try:"
  echo "/opt/anaconda3/bin/conda if you have installed Anaconda system-wide on Mac."
  echo "NO_ENV if the environment for this analysis is already up and running."
  exit 1
fi
cd ..
if [ "$1" != "NO_ENV" ]; then
    echo "building conda environment"
    __conda_setup="$('/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    eval "$__conda_setup"
    conda deactivate > /dev/null
    conda create -n mbta-prediction-models --file environment.txt -y > /dev/null
    conda activate mbta-prediction-models > /dev/null
    pip install --no-deps meteostat > /dev/null
fi
echo "conda environment done, starting data loading"
cd scripts/processing_scripts
python meteostat_import.py
cd ../../data/input_data
unzip -o GSE_by_year.zip > /dev/null
unzip -o MBTA_Service_Alerts.csv.zip > /dev/null
cd ../../scripts/processing_scripts
echo "data loading done, starting data processing; this may take a minute"
python point_process_data_wrangling.py
python data_wrangling.py no_debug no_cheatsheet
cd ../analysis_scripts
echo "data processing done, starting analysis script"
if [ -z "$2" ]; then
  ./run_analysis.sh
else
  ./run_analysis.sh COMPUTE_BOOTSTRAP
fi
echo "analysis script done, master script done!"
