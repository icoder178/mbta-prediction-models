#!/bin/bash
set -e
# Runs only Linear Support Vector Regression hyperparameter tuning. Run from any directory.

lookback="${1:-5}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
pid=""

cleanup() {
  if [ -n "$pid" ]; then
    kill "$pid" 2> /dev/null || true
    wait "$pid" 2> /dev/null || true
  fi
  exit 130
}

trap cleanup INT TERM

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
echo "Starting Linear Support Vector Regression tuning only, lookback=${lookback}."
python -u -W ignore hyperparameter_tuning.py SupportVector "$lookback" &
pid=$!
wait "$pid"
pid=""
end_time="$(date +%s)"

echo "Linear Support Vector Regression tuning done in $((end_time-start_time)) seconds."
echo "Outputs:"
echo "The output files keep the internal SupportVector prefix for compatibility:"
echo "../../data/intermediate_data/hyperparameter_tuning_results_SupportVector.csv"
echo "../../data/intermediate_data/selected_hyperparameters_SupportVector.csv"
