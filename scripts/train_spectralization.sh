set -ex
cd ..
python train.py --dataroot dataset/202201数据集 --name spectralization --model spectralization --crop_size 512 --preprocess centercrop --no_flip --n_epochs_decay 900