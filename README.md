# mbta-prediction-models
Model comparison for prediction of two different metrics, total daily delay count in the Boston MBTA, and total gated station entries to the Boston train system.
## Prerequisites
The following packages are required:
```bash
pip
python
conda
```
A conda environment named "mbta-prediction-models" will be created when running the master script with all other necessary dependencies.
## Documentation
Beside this README file, other documentation includes:
data/input_data/metadata/codebook.md: Explanation of raw input data.
data/input_data/metadata/data_sources.md: Sources of raw input data.
data/analysis_data/data_appendix.md: Explanation of cleaned input data.
## Reproduction
Clone and run in a single command (assuming Mac and system-wide install of Anaconda) with:
```bash
git clone https://github.com/icoder178/mbta-prediction-models.git && cd mbta-prediction-models/scripts && ./master_script.sh /opt/anaconda3/bin/conda 
```
Results will appear in output/results; two bar graphs of model performance on the two tasks as well as a table of performance rankings will appear.