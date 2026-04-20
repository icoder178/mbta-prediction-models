#!/bin/bash
set -e
# Runs only SupportVector hyperparameter tuning. Run from any directory.

lookback="${1:-5}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

required_files=(
  "../../data/analysis_data/GSE_inner_train_inputs.csv"
  "../../data/analysis_data/GSE_validation_inputs.csv"
  "../../data/analysis_data/delay_inner_train_inputs.csv"
  "../../data/analysis_data/delay_validation_inputs.csv"
)

for file in "${required_files[@]}"
do
  if [ ! -f "$file" ]; then
    echo "Missing required tuning data file: $file"
    echo "Run data_wrangling.py first to generate inner-train and validation data."
    exit 1
  fi
done

start_time="$(date +%s)"
echo "Starting SupportVector tuning only, lookback=${lookback}."
python -u -W ignore hyperparameter_tuning.py SupportVector "$lookback"
end_time="$(date +%s)"

echo "SupportVector tuning done in $((end_time-start_time)) seconds."
echo "Outputs:"
echo "../../data/intermediate_data/hyperparameter_tuning_results_SupportVector.csv"
echo "../../data/intermediate_data/selected_hyperparameters_SupportVector.csv"
