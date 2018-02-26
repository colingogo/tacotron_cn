export CUDA_VISIBLE_DEVICES=0,1
npy_data=/home/user-1/mzq/wavnet_dataset/
npy_txt=/home/user-1/mzq/wavnet_dataset/train.txt
#python3 preprocess.py --num_workers 56 --name ljspeech --in_dir /home/user-1/mzq/med/  --out_dir $npy_data --hparams sample_rate=22050
python3 train.py --num_gpus 2 --batch_size 1 --train_txt $npy_txt --hparams gc_enable=False,global_channel=0,global_cardinality=0,NPY_DATAROOT=$npy_data,sample_rate=22050 --logdir_root log_ljspeech
