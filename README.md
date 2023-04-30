# CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP

![Overview of the method](./assets/method.png)

# Installation
**Step 1.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**Step 2.** Install SPVCNN and MinkowskiEngine  (https://github.com/NVIDIA/MinkowskiEngine).
```shell
# MinkowskiEngine
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
pip install ninja
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# SPVCNN
```

**Step 3.** Install CLIP, MaskCLIP, Pytorch_lightning, Nuscenes devkit.
```shell
# Install CLIP (https://github.com/openai/CLIP)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
# Install MaskCLIP (https://github.com/chongzhou96/MaskCLIP)
pip install -U openmim
mim install mmcv-full
git clone https://github.com/chongzhou96/MaskCLIP.git
cd MaskCLIP
pip install -v -e .
# Install Pytorch_lightning 
pip install pytorch_lightning==1.4.0
# Install Nuscenes devkit 
pip install torchmetrics==0.4.0
pip install nuscenes-devkit==1.1.9
```

# Data Preparation
In this paper, we conduct experiments on [ScanNet](http://www.scan-net.org), [Nuscenes](https://www.nuscenes.org/nuscenes#overview), and [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview).

**Step 0.** Download the ScanNet, NuScenes and SemanticKITTI dataset. 
```shell
# Pre-processing the scannet dataset
python utils/preprocess_scannet.py
# Obtain nuScenes's sweeps information in (https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md), and save as "nuscenes_infos_dict_10sweeps_train.pkl"
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainva

```

**Step 1.** Download and convert the CLIP models,
```shell
python utils/convert_clip_weights.py --model ViT16 --backbone
```

**Step 2.** Prepare the CLIP's text embeddings of the scannet and nuscenes datasets,
```shell
python utils/prompt_engineering.py --model ViT16 --class-set nuscenes
python utils/prompt_engineering.py --model ViT16 --class-set scannet
```

# Pre-training

**ScanNet.** 
```shell
python pretrain.py --cfg_file config/clip2scene_scannet.yaml
# The pre-trained model will be saved in 
```
**NuScenes.** 
```shell
python pretrain.py --cfg_file config/clip2scene_nuscenes.yaml
# The pre-trained model will be saved in 
```

# Annotation-free

**ScanNet.** 
```shell
python pretrain.py --cfg_file config/clip2scene_scannet.yaml
# The pre-trained model will be saved in 
```
**NuScenes.** 
```shell
python pretrain.py --cfg_file config/clip2scene_nuscenes.yaml
# The pre-trained model will be saved in 
```


# Fine-tuning on labeled data

**ScanNet.** 
```shell
python pretrain.py --cfg_file config/clip2scene_scannet.yaml
# The pre-trained model will be saved in 
```
**NuScenes.** 
```shell
python pretrain.py --cfg_file config/clip2scene_nuscenes.yaml
# The pre-trained model will be saved in 
```


# Citation
If you use CLIP2Scene or this code base in your work, please cite
```

```

Ackkownled

# Contact
For questions about our paper or code, please contact [Runnan Chen](rnchen2@cs.hku.hk).
