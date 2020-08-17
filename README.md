# AR-Net: Adaptive Resolution Network for Efficient Video Understanding
[[Project Page]](https://mengyuest.github.io/AR-Net) [[arXiv]](https://arxiv.org/pdf/2007.15796.pdf)


![! an image](https://mengyuest.github.io/AR-Net/resources/2_network.png)


[Yue Meng](https://mengyuest.github.io/)<sup>1,3</sup>, [Chung-Ching Lin](https://scholar.google.com/citations?&user=legkbM0AAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [Rameswar Panda](https://rpand002.github.io/)<sup>1</sup>, [Prasanna Sattigeri](https://pronics2004.github.io/)<sup>1</sup>, [Leonid Karlinsky](https://researcher.watson.ibm.com/researcher/view.php?person=il-LEONIDKA)<sup>1</sup>, [Aude Oliva](http://olivalab.mit.edu/audeoliva.html)<sup>1,3</sup>, [Kate Saenko](http://ai.bu.edu/ksaenko.html)<sup>1,2</sup>, [Rogerio Feris](http://rogerioferis.com/)<sup>1</sup>

<sup>1</sup> MIT-IBM Watson AI Lab, IBM Research

<sup>2</sup> Boston University

<sup>3</sup> Massachusetts Institute of Technology

In European Conference on Computer Vision (ECCV), 2020


## Reference
If you find our code or project useful for your research, please cite:
```latex
@article{meng2020ar,
  title={AR-Net: Adaptive Frame Resolution for Efficient Action Recognition},
  author={Meng, Yue and Lin, Chung-Ching and Panda, Rameswar and Sattigeri, Prasanna and Karlinsky, Leonid and Oliva, Aude and Saenko, Kate and Feris, Rogerio},
  journal={arXiv preprint arXiv:2007.15796},
  year={2020}
}
```

## Requirements
Our experiments are conducted on 4 Tesla V100 (32GB):
```bash
conda create -n arnet python=3.7.6
conda activate arnet
conda install pytorch torchvision tqdm
pip install tensorboardX thop 
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .
```

## Dataset preparation
1. Get the ActivityNet-v1.3 train/test splits (and classes file) from [[Google Drive]](https://drive.google.com/drive/folders/1j7XF86Wq9sNbBwHCIA0lQn1w015CVF4g?usp=sharing) and put them in `/foo/bar/activity-net-v1.3`. Here `/foo/bar` is your directory to save the datasets.
2. Download ActivityNet-v1.3 videos from [here](http://activity-net.org/download.html) (contact [them](http://activity-net.org/people.html) if there is any missing video) and save to `/foo/bar/activity-net-v1.3/videos`
3. Extract frames using the script from the repository:
``` shell
cd ./ops
python video_jpg.py /foo/bar/activity-net-v1.3/videos /foo/bar/activity-net-v1.3/frames  --parallel
```
The frames will be saved to `/foo/bar/activity-net-v1.3/frames`.

Using the same procedures you can also get FCVID and mini-Kinetics. For more details please check [ops/dataset_config.py](ops/dataset_config.py)

##  Pretrained Models
Download all our models from [[Google Drive]](https://drive.google.com/drive/folders/1YlPxgFm0bI6BH8D8VqSKbH6ykZX2lhif?usp=sharing) and save them to `/foo1/bar1/model_path`

##  Evaluation
To test all the models on ActivityNet-v1.3, run:
```bash
sh full_test.sh /foo/bar/activity-net-v1.3 /foo1/bar1/model_path
```
The first parameter is for data path, and the second parameter is for model path. Make sure you have 4xV100(32G) to reproduce exactly the same results for the adaptive approaches
(In the last two exps of the script, we also reported improved performances by using an updated training logic. See the "Training" section for more details)

## Training
To train to get the AR-Net(ResNet) as shown in Table 1  (mAP-73.8), follow the training script [here](scripts_train.sh). It might take around 1-2 days.

We also improved our model's performance by simply changing some parameters like the learning rate. The new mAP for AR-Net(ResNet) now is 76.8 (while the updated baseline is 75.6) and the training script can be found [here](scripts_train_new.sh).


Our code is based on [TSM](https://github.com/mit-han-lab/temporal-shift-module)