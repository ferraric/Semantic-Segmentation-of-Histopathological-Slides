module load python_gpu/3.7.1
module load eth_proxy
pip3 install -r requirements.txt
bsub python3 mains/transfer_learning_unet_main.py -c configs/transfer_learning_unet_config.json -R "rusage[ngpus_excl_p=1]" -W 10:00
