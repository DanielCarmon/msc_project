#!/bin/tcsh
set script_dir = /specific/netapp5_2/gamir/carmonda/research/vision/new_embed/control
set all_machines = ( pc-wolf-g01 rack-nachum-g01 pc-gamir-g01 rack-gamir-g01 savant )
#set all_machines = ( pc-gamir-g01 pc-wolf-g01 pc-wolf-g02 )
#set all_machines = ( 'savant' )
rm ~/workers.txt # delete old file
foreach m ( $all_machines )
  echo 'start sshing: '$m
  ssh carmonda@$m python $script_dir/get_gpus.py
  echo 'finished sshing: 'm
end
