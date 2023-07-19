# Installation

Use the same Python environment as in [training/bing_bert](../../training/bing_bert/INSTALL.md).

```
# run in an allocation with 1 GPU per task (-n <NUMPER_OF_GPUS>)
# set --maxsize to a number to saturate the bandwidth limits
srun python all_reduce.py --scan --maxsize 27 --bw-unit GBps

# for intra-node bandwith make use to use only one node
# e.g. 
srun -N 1 -n <number of gpus per node> python all_reduce.py --scan --maxsize 27 --bw-unit GBps
```
