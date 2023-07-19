# Installation

```
module load modenv/hiera  GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0 Python/3.10.4
python -m venv venv
source venv/bin/activate

git clone --branch=v0.9.5 https://github.com/microsoft/DeepSpeed.git
git clone --branch=gpu_bench https://github.com/tud-zih-ki/DeepSpeedExamples.git

cd DeepSpeed
pip install torch\>=2
pip install .

pip install tensorboardX boto3 requests h5py mpi4py
pip install tbparse # required in export_from_tensorboard.py

cd ../DeepSpeedExamples/training/bing_bert

# download & extract dataset from https://cloudstore.zih.tu-dresden.de/index.php/s/RyeDoaogesGJFCR
# change dataset path in ds_train_bert_nvidia_data_bsz64k_seq128_tud.sh
# change train_micro_batch_size_per_gpu in deepspeed_bsz64k_lamb_config_seq128_tud.json to gpu memory

# run it in an allocation with 1 gpu per task (-n <NUMPER_OF_GPUs>)
srun ./ds_train_bert_nvidia_data_bsz64k_seq128_tud.sh
```
