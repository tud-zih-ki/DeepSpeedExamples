# Installation

```
#module load release/24.04  GCC/13.2.0 OpenMPI/4.1.6 CUDA/12.6.0 Python/3.11.5

python -m venv ./venv
source ./venv/bin/activate

git clone --branch=gpu_bench_deneb https://github.com/tud-zih-ki/DeepSpeedExamples.git

pip install torch\>=2.6
pip install deepspeed\>=v0.16.5

pip install tensorboardX boto3 requests h5py mpi4py
pip install tbparse # required in export_from_tensorboard.py

cd DeepSpeedExamples/training/bing_bert

# download & extract dataset from https://cloudstore.zih.tu-dresden.de/index.php/s/RyeDoaogesGJFCR
# change dataset path in ds_train_bert_nvidia_data_bsz64k_seq128_tud.sh
# change train_micro_batch_size_per_gpu in deepspeed_bsz64k_lamb_config_seq128_tud.json to gpu memory

# run it in an allocation with 1 gpu per task (-n <NUMPER_OF_GPUs>)
export TQDM_DISABLE=1 # disable progressbar in non interactive jobs
srun ./ds_train_bert_nvidia_data_bsz64k_seq128_tud.sh
```
