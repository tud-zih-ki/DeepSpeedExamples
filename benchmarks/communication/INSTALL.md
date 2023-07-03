# Installation

```
# use the same python enviroment as in DeepSpeedExamples/blob/gpu_bench/training/bing_bert/INSTALL.md

# run it in an allocation with 1 gpu per task (-n <NUMPER_OF_GPUS>)
# set --maxsize to a number to saturate the bandwidth limits
srun python all_reduce.py --scan --maxsize 27 --bw-unit GBps
```
