
# Baselines

ipython -i  main_restricted.py --  --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_clipvitl336_10classes.jsonl --including_label False --batch_size 32
