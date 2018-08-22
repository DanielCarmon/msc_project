#!/bin/tcsh
set project_dir = '/specific/netapp5_2/gamir/carmonda/research/vision/msc_project'
set all_machines = ( pc-gamir-g01 pc-wolf-g01 pc-wolf-g02 rack-gamir-g01 rack-gamir-g02 )
foreach m ( $all_machines )
  echo 'killing python processes on '$m
  ssh carmonda@$m $project_dir/local_kill_py.sh 
end
