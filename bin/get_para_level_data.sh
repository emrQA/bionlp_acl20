#!/bin/bash

conda activate emrqa

echo 'preparing the data....'
python ../scripts/get_emrqa_para_level_data.py

