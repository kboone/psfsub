#!/bin/bash

num_tiles_x=1
num_tiles_y=1

binary=$HOME/optimal_subtraction/psfsub/test_sub.py
#stitch_binary=$HOME/optimal_subtraction/psfsub/stitch.py

#for ((i=0; i<num_tiles_x; i++)); do
    #for ((j=0; j<num_tiles_x; j++)); do
        #echo "export PATH=$PATH; cd $PWD; python $binary $i $num_tiles_x $j \
            #$num_tiles_y" | qsub -wd $PWD -S /bin/bash -pe orte 3 -sync y \
            #-p -100 &
    #done
#done
echo "export PATH=$PATH; cd $PWD; python $binary" | qsub -wd $PWD -S /bin/bash -pe orte 24 -sync y -p -100 &

wait

# Stitch images
#python $stitch_binary $num_tiles_x $num_tiles_y
