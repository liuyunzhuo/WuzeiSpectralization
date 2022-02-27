# 乌贼视觉高光谱图像重建

## 神经网络复原

### GAN网络(Spectralization)
#### 训练
Tips: 

1.将其命名为了Spectralization(光谱化)模型

2.可直接运行scripts/train_ectralization.sh

```shell
python train.py --dataroot dataset/202201数据集 --name spectralization --model spectralization --crop_size 512 --preprocess centercrop --no_flip --n_epochs_decay 900 --export_mat
```

#### 测试
Tips:
1.可直接运行scripts/test_spectralization.sh
```shell
python test.py --dataroot dataset/202201数据集 --name spectralization --model spectralization --crop_size 512 --preprocess centercrop --export_mat --num_test 5
```
## 注意
本代码大范围借鉴于https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
## Todo

1.添加UNet 单向网络

2.添加基于PSF先验的网络
