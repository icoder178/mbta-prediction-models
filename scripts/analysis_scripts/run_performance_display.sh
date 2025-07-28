#!/bin/bash
# script for running just performance_display.py and feeding results to the right location; run in the directory it is in
echo "outputting performance summary:"
python performance_display.py > ../../output/results/performance_summary.txt
echo "output done!"
