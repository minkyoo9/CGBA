#!/bin/bash

# Define the list of cluster numbers
c_list=(33 136 48 281 170 319 217 5 8 10 93 68 9 124 69 74 125 53 13 473 27 112 100 159 24 128 318 39 73 77 330 221 414 138 156 411 25 450 94 856 0 14 459 97 536 42 45 15 75 214 167 180 23 131)

# Iterate through each cluster number
for cluster in "${c_list[@]}"; do
   # Run the python script with nohup
   nohup python COVID_Make_poisonedBERT.py -g 0 -c "$cluster" -m 0.2 -a 0.1 -ag 10 > "TrainLogs/PoisonedBERT_c${cluster}.log" 2>&1
done

echo "All processes completed."