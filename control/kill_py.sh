#!/bin/tcsh
set script_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/new_embed/control'
set all_machines = ( pc-gamir-g01 pc-wolf-g01 pc-wolf-g02 rack-gamir-g01 rack-gamir-g02 savant rack-nachum-g01 )
foreach m ( $all_machines )
  echo 'killing python processes on '$m
  ssh carmonda@$m $script_dir/local_kill_py.sh 
end
