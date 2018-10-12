#!/bin/tcsh
set script_dir = /specific/netapp5_2/gamir/carmonda/research/vision/new_embed/control
set all_machines = ( pc-gamir-g01 pc-wolf-g01 pc-wolf-g02 rack-gamir-g01 rack-gamir-g02 )
rm ~/workers.txt # delete old file
foreach m ( $all_machines )
  ssh carmonda@$m python $script_dir/kill_gpus.py
end
