
# Baselines

ipython -i  main_restricted_idefics.py --  --method clip --model_id ViT-SO400M-14-SigLIP-384 --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_siglip_9classes.jsonl --including_label False --batch_size 32
