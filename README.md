# Fine tune the well-trained model

## 具體位置我放 D 槽的 240722_YuHsinModel

使用的 Dataset(input/FILENAME)分別有:

1. test_Data_beads/data62to68_from280830_BF_LS_6um.tif
2. test_Data_Lung_deconv/RL_10_from230911.tif
3. LungTissue_from230901/data1.tif、data2.tif、data3.tif、data4.tif、data4hilo.tif、data5.tif、data5hilo.tif
4. deconvoluted_6um_0.5um_10iter_from230803/Final Display of RL.tif、Final Display of RL_adjustContrast.tif

Note: 有用"、"隔開的代表有多張不同名稱的單一影像

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

## Markdown 快捷鍵預覽

cmd + shift + v
