#!/bin/bash

# directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

scp iris:/cscratch/abbatej/final_data_full_batch_0.pkl $DIR/data
