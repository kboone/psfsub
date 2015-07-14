#!/bin/bash

num_tiles_x=8
num_tiles_y=8

binary=$HOME/optimal_subtraction/psfsub/test_sub_parallel.py
stitch_binary=$HOME/optimal_subtraction/psfsub/stitch.py

for ((i=0; i<num_tiles_x; i++)); do
    for ((j=0; j<num_tiles_x; j++)); do
        echo "export PATH=$PATH; cd $PWD; python $binary $i $num_tiles_x $j \
            $num_tiles_y" | qsub -wd $PWD -S /bin/bash -pe orte 3 -sync y &
    done
done

wait

# Stitch images
python $stitch_binary $num_tiles_x $num_tiles_y
