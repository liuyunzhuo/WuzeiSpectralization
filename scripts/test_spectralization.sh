set -ex
cd ..
python test.py --dataroot dataset/202201数据集 --name spectralization --model spectralization --crop_size 512 --preprocess centercrop --export_mat --num_test 5