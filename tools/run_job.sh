bsub -n 1 -R "rusage[mem=6000, ngpus_excl_p=1]" -W 10:00 python3 mains/transfer_learning_unet_main.py -c configs/transfer_learning_unet_config.json
