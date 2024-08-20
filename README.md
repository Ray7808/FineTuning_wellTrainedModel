# Fine tune the well-trained model

Fine tune the last layer of the well-trained model.
Here we use different activation function (sigmoid and tanh), the value of regularization (L2 loss), gradient clipping, and learning rate.

## The position is at D:/240722_YuHsinModel

Dataset(input/FILENAME) for fed into the well-trained model:

1. test_Data_beads/data62to68_from280830_BF_LS_6um.tif
2. test_Data_Lung_deconv/RL_10_from230911.tif
3. LungTissue_from230901/data1.tif、data2.tif、data3.tif、data4.tif、data4hilo.tif、data5.tif、data5hilo.tif
4. deconvoluted_6um_0.5um_10iter_from230803/Final Display of RL.tif、Final Display of RL_adjustContrast.tif

Note: "、" is used to divide images with different filename

## Pyenv and pip comments

```bash
pyenv versions // 查看裝了哪些python版本

pyenv local 3.8.18 // 切換當前目錄的版本為3.8.18


python -m venv .venv // 創建虛擬環境

source .venv/bin/activate // 激活虛擬環境 (windows是. .venv\Scripts\Activate)

deactivate // 退出虛擬環境

pip install <package1> <package2>  // 安装所需的套件

pip freeze > requirements.txt      // 生成 requirements.txt 文件

pip cache purge // 刪除快取
```

## Markdown shortcut preview

cmd + shift + v
