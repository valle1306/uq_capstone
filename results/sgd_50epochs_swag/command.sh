src/train_swag_proper.py --dataset chest_xray --batch_size 32 --num_workers 8 --arch resnet18 --pretrained --epochs 50 --lr_init 0.01 --momentum 0.9 --wd 1e-4 --swa_start 27 --swa_lr 0.005 --swa_c_epochs 1 --max_num_models 20 --output_dir runs/classification/swag_proper --device cuda --seed 42
er --device cuda --seed 1
