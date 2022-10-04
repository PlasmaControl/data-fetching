#!/bin/bash
# A simple job submission script called simple.sbatch
# It submits a job called simple.x
# submit this to the Slurm scheduler this using the command:
## sbatch simple.sbatch

# NO bash commands can appear before the #SBATCH directives

# For comments, use '##' or '# ' but not '#stuff', as what follows
# and '#SBATCH' is interpreted by the scheduler.

# NOTE that normal Bash script commands WILL NOT WORK behind the
# '#SBATCH' directives !!


################################################

## Specify the queue (a.k.a. the "p"artition)
#SBATCH -p long 

## Specify the number of cores
#SBATCH -n 1

## further restrict your code to a specific number of nodes (computers)
## Only really necessary if you need shared memory, e.g. for OpenMP
#SBATCH -N 1

####
## With time and memory, the less you request, the higher its priority 
## in the queue, but if you exceed your allotment, the job will be killed.
####

## Specify the amount of time for the job (in minutes)
#SBATCH -t 7-00:00:00
## or as -t days-HH:MM:SS, e.g. -t 7-00:00:00 for the long queue

## Specify the amount of memory required (M|G|T) per cpu:
#SBATCH --mem-per-cpu=100G

## Specify an output file -- Bash environment variables WILL NOT WORK HERE
## only a limited number or replacers are available, such as 
## %j = jobID 
## %u = username 
## %N = (first) node name 
#SBATCH -o slurm-vloop-%j.out

## Optionally export your environment variables to the job,
## generally a good idea
# #SBATCH --export=NONE
#SBATCH --export=ALL

## Now launch the actual job
echo "modules loaded:"
module list
echo "beginning job ..."
python new_database_maker.py configs/vloop.yaml
echo " done."
