# Matroid-Centroid-Reid

## Introduction
This is a modified implementation of the paper "On the Unreasonable Effectiveness of Centroids in Image Retrieval" to train and evaluate the model on other new datasets namely - LaST, SYSU-30k and the old academic dataset - Market1501

## Environment
```bash
conda env create -f environment.yml
conda activate ctl
```

## Config file
The main parameters and arguments for training/evaluation are specified in config.yaml file such as 
- Base model architecture
- Backbones
- input/output sizes
- Batch sizes
- Reranking
- Caching
- Parallelized evaluation

Checkout ./config/defaults.py for a comprehensive list of parameters set by config files. \
Some important ones are: 
- Model
```bash
MODEL:
  NAME: 'resnet50'
  BACKBONE_EMB_SIZE: 2048
  PRETRAIN_PATH: '<CHECKPOINT_DIR>/checkpoints/market1501_resnet50_256_128_epoch_120.ckpt'
  KEEP_CAMID_CENTROIDS: False
  CUSTOM_FINAL_CLASSIFICATION:
    ENABLED: True
    ORIGINAL_NUM_CLASSES: 751
```

**Note**: \
The option CUSTOM_FINAL_CLASSIFICATION is used when the checkpoint is trained with different number of classes that the model architecture we are loading the checkpoint into. \
This was an easier fix compared to dropping final classification layers which are not used anyway for similarity search.

- Dataset
```bash
DATASETS:
  NAMES: 'last'
  ROOT_DIR: '<DATASET_DIR>'
  VERBOSE: True
```

- Input
```bash
INPUT:
  SIZE_TRAIN: [256,128]
  SIZE_TEST: [256,128]
```

- Test options
```bash
TEST:
  IMS_PER_BATCH: 128
  ONLY_TEST: True
  USE_EVAL_SET: False
  CACHE:
    ENABLED: True
    CACHE_DIR: './logs/last/256_resnet50/cache'
    FEAT_REUSE:
      ENABLED: True
      PREFIX: 'CTL_cached_inference'
    PARALLEL:
      ENABLED: True
      NUM_WORKERS: 14
```

## Evaluation
Evaluation can be performed in two ways: \
Using Market1501 as an example. 
- Single stage evaluation 
```bash
python train_ctl_model.py --config_file="configs/256_resnet50_test.yml"
```
Single stage is possible only for Market1501 on dev machine due to CPU memory constraints.

- Two stage evaluation
    - Inference
    ```bash
    python3 inference/save_test_inference_embeddings.py --config 'configs/256_resnet50_test_cache_batched_single.yml'
    ```

    - Evaluation without centroids
    ```bash
    python3 inference/run_eval_on_saved_features.py --config 'configs/256_resnet50_test_cache_eval_parallel.yml'
    ```

    - Evaluation with centroids
    ```bash
    python3 inference/run_eval_on_saved_features.py --config 'configs/256_resnet50_test_cache_eval_parallel.yml' --use_centroids
    ```

## Final Evaluation scripts
- **LaST**
    - Eval set
        - Inference
        ```bash
        python3 inference/save_test_inference_embeddings.py --config 'configs/last/256_resnet_eval_cache_inference_single.yml'
        ```
        - Evaluation
            - Without centroids
            ```bash
            python3 inference/run_eval_on_saved_features.py --config 'configs/last/256_resnet_eval_cache_eval_single.yml'
            ```

            - With centroids (mAP not good metric here)
            ```bash
            python3 inference/run_eval_on_saved_features.py --config 'configs/last/256_resnet_eval_cache_eval_single.yml' --use_centroids
            ```

    - Test set
        - Inference
        ```bash
        python3 inference/save_test_inference_embeddings.py --config 'configs/last/256_resnet_test_cache_inference_single.yml'
        ```

        - Evaluation
            - Without centroids 
            ```bash
            python3 inference/run_eval_on_saved_features.py --config 'configs/last/256_resnet_test_cache_eval_single.yml'
            ```

            - With centroids (mAP not good metric here)
            ```bash
            python3 inference/run_eval_on_saved_features.py --config 'configs/last/256_resnet_test_cache_eval_single.yml' --use_centroids
            ```

- **SYSU-30k**
    - Inference
    ```bash
    python3 inference/save_test_inference_embeddings.py --config 'configs/sysu30k/256_resnet_cache_inference_single.yml'
    ```

    - Evaluation
        - Without centroids
        ```bash
        python3 inference/run_eval_on_saved_features.py --config 'configs/sysu30k/256_resnet_cache_eval_single.yml'
        ```

        - With centroids (mAP not good metric here) \
        **Caution**: \
        Here we don’t set respect_camids flag while creating centroid embeddings. See "256_resnet_cache_inference_only_centroids_single.yml" \
        This is so as that logic was hand tailored for Market1501. 
        ```bash
        python3 inference/run_eval_on_saved_features.py --config 'configs/sysu30k/256_resnet_cache_eval_single.yml' --use_centroids
        ```


## Training
Script to train on LaST dataset.
```bash
python train_ctl_model.py --config_file="configs/last/256_resnet_train.yml"
```

# On the Unreasonable Effectiveness of Centroids in Image Retrieval (Original Paper Documentation)

Official code repository for paper "On the Unreasonable Effectiveness of Centroids in Image Retrieval". \
Paper accepted to ICONIP 2021 conference.  
## Get Started

The whole model is implemented in PyTorch-Lightning framework.

1. `cd` to a directory where you want to clone this repo
2. Run `git clone https://github.com/mikwieczorek/centroids-reid.git`
3. Install required packages `pip install -r requirements.txt`
4. Download pre-trained weights into `models/` directory for:
    - Resnet50 from here: [[link]](https://download.pytorch.org/models/resnet50-19c8e357.pth)
    - Resnet50-IBN-A from here: [[link]](https://drive.google.com/open?id=1_r4wp14hEMkABVow58Xr4mPg7gvgOMto)

5. Prepare datasets:

    Market1501

    * Extract dataset and rename to `market1501` inside `/data/`
    * The data structure should be following:

    ```bash
    /data
        market1501
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    DukeMTMC-reID

    * Extract dataset to `/data/` directory
    * The data structure should be following:

    ```bash
    /data
        DukeMTMC-reID
           	bounding_box_test/
           	bounding_box_train/
           	......
    ```

    Street2Shop & Deep Fashion (Consumer-to-shop)

    1. These fashion datasets require the annotation data in COCO-format with additional fields in `annotations`
        ```
        JSON:{
            'images' : [...],
            'annotations': [
                        {...,
                        'pair_id': 100,         # an int type
                        'source': 'user'        # 'user' or 'shop'
                        },
                        ...
                    ]
        }
        ```
    2. The product images should be pre-cropped to the given input format (either 256x128 or 320x320) using original images and provided bounding boxes to allow faster training.

    ### Path to the data root and JSON files (only for Street2shop and Deep Fashion) can be adjusted by passing the paths as parameters to train scripts
    ### You can familiarize yourself with the detailed configuration and its meaning in `config.defaults.py`, which includes all parameters available to the user.

## Train
Each Dataset and Model has its own train script.  
All train scripts are in `train_scirpts` folder with corresponding dataset name.

Example run command to train CTL-Model on DukeMTMC-reID
```bash
CUDA_VISIBLE_DEVICES=3 ./train_scripts/dukemtmc/train_ctl_model_s_r50_dukemtmc.sh
```
`CUDA_VISIBLE_DEVICES` controls which GPUs are visible to the scripts.

By default all train scripts will launch 3 experiments.

## Test
To test the trained model you can use provided scripts in `train_scripts`, just two parameters need to be added:  
    
    TEST.ONLY_TEST True \  
    MODEL.PRETRAIN_PATH "path/to/pretrained/model/checkpoint.pth"
    
Example train script for testing trained CTL-Model on Market1501
```bash
python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet50/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "logs/market1501/256_resnet50/train_ctl_model/version_0/checkpoints/epoch=119.ckpt"
```

## Trained model weights

Weights of trained CTL-Model on Market1501 and DuketMTMC-Reid are avaialable here [[link]](https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK)

## Inference

To run inference on an image dataset with a trained model you can use provided scripts in [`inference`](https://github.com/mikwieczorek/centroids-reid/tree/main/inference) folder.
More info can be found in `inference/README.md`.


## Fashion dataset to COCO-ReID format preparation

Fashion datasets require transforming to COCO-Reid format. In [`scripts`](https://github.com/mikwieczorek/centroids-reid/tree/main/scripts) folder there is code necessary to prepare raw Street2Shop and DeepFashion annotation and images to correct format.
More info can be found in `scripts/README.md`.

## **Citation**


```
Wieczorek M., Rychalska B., Dąbrowski J. 
(2021) On the Unreasonable Effectiveness of Centroids in Image Retrieval.
In: Mantoro T., Lee M., Ayu M.A., Wong K.W., Hidayanto A.N. (eds) 
Neural Information Processing. ICONIP 2021.
Lecture Notes in Computer Science, vol 13111. Springer, Cham. https://doi.org/10.1007/978-3-030-92273-3_18

```

```
@article{Wieczorek2021OnTU,
  title={On the Unreasonable Effectiveness of Centroids in Image Retrieval},
  author={Mikolaj Wieczorek and Barbara Rychalska and Jacek Dabrowski},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.13643}
}
```