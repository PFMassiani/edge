#!/bin/bash

cd ..
source /home/pifou/.virtualenvs/edge/bin/activate
process=$1
shift
experiment_name=$1
shift
python_script_name=$1
python_script_path="./experiments/$experiment_name/$python_script_name"
shift
python $python_script_path --rid $random_id $*