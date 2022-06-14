# encoding: utf-8
"""
Partially based on work by:
@author:  sherlock
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek

Further Adapted and extended by:
@author: Nilay Pande
@contact: nilay017@gmail.com
"""
import glob
import os.path as osp
import numpy as np
from collections import defaultdict

import pytorch_lightning as pl
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              SequentialSampler)

from .bases import (BaseDatasetLabelled, BaseDatasetLabelledPerPid,
                    ReidBaseDataModule, collate_fn_alternative, pil_loader)
from .samplers import get_sampler
from .transforms import ReidTransforms
from torchvision.datasets import ImageFolder

class sysu30k(ReidBaseDataModule):
    """SYSU-30k.
    Reference:
        SYSU-30k: Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark

    URL: `<https://github.com/wanggrun/SYSU-30k>`

    Dataset statistics:
        SYSU-30k contains 30k categories of persons, which is about 20 times larger than CUHK03 (1.3k categories) 
        and Market1501 (1.5k categories), and 30 times larger than ImageNet (1k categories). SYSU-30k contains 
        29,606,918 images. Moreover, SYSU-30k provides not only a large platform for the weakly supervised ReID problem 
        but also a more challenging test set that is consistent with the realistic setting for standard evaluation. 


        Comparision with existing datasets
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Dataset	    CUHK03	    Market-1501	    Duke	    MSMT17	        CUHK01	    PRID	    VIPeR	    CAVIAR	    SYSU-30k        |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Categories	1,467	    1,501	        1,812	    4,101	        971	        934	        632	        72	        30,508          |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Scene	    Indoor	    Outdoor	        Outdoor	    Indoor,Outdoor	Indoor	    Outdoor	    Outdoor	    Indoor	    Indoor,Outdoor  |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Annotation	Strong	    Strong	        Strong	    Strong	        Strong	    Strong	    Strong	    Strong	    Weak            |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Cameras	    2	        6	            8	        15	            10	        2	        2	        2	        Countless       |
        -----------------------------------------------------------------------------------------------------------------------------------------
        |   Images	    28,192	    32,668	        36,411	    126,441	        3,884	    1,134	    1,264	    610	        29,606,918      |
        -----------------------------------------------------------------------------------------------------------------------------------------


        Comparision with ImageNet-1k
        --------------------------------------------
        | Dataset	   | ImageNet-1k  |  SYSU-30k  |
        --------------------------------------------
        | Categories   | 1,000	      |  30,508    |
        --------------------------------------------
        | Images	   | 1,280,000	  |  29,606,918|
        --------------------------------------------
        | Annotation   | Strong	      |  Weak      |
        --------------------------------------------
    """

    dataset_dir = ''
    dataset_name = "sysu-30k-release"

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_name)
        self.use_eval_set = cfg.TEST.USE_EVAL_SET
        self.verbose = cfg.DATASETS.VERBOSE
        self.train_dir = osp.join(self.dataset_dir, 'sysu_train_set_all')
        self.query_dir = osp.join(self.dataset_dir, 'sysu_test_set_all', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'sysu_test_set_all', 'gallery')


    def setup(self):
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)
        transforms_base = ReidTransforms(self.cfg)

        train, train_dict = self.process_dir(self.train_dir, is_train=True)
        self.train_dict = train_dict
        self.train_list = train
        self.train = BaseDatasetLabelledPerPid(train_dict, transforms_base.build_transforms(is_train=True), self.num_instances, self.cfg.DATALOADER.USE_RESAMPLING)
        
        query, _ = self.process_dir(self.query_dir)
        gallery, _ = self.process_dir(self.gallery_dir)
        self.query_list = query
        self.gallery_list = gallery
        self.val = BaseDatasetLabelled(query+gallery, transforms_base.build_transforms(is_train=False))

        if self.verbose:
            print("=> SYSU-30k loaded")
            print("=> Train mode: ", self.use_eval_set)
            self.print_dataset_statistics_movie(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)

        self.num_query = len(query)
        
        self.num_classes = self.num_gallery_pids


    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


    def process_dir(self, dir_path, is_train=False):
        img_path = ImageFolder(dir_path).imgs
        dataset_dict = defaultdict(list)
        dataset = []
        for ii, (path, v) in enumerate(img_path):
            filename = path.split('/')[-1]
            if not is_train:
                label = filename.split('_')[0]
                camera = filename.split('c')[1]
                if label[0:10]=='0000others':
                    label_id = -1000
                else:
                    label_id = int(label)
                camera_id = int(camera[0])
            else:
                # Dummy label and camera ids for train images as this is a weakly supervised dataset
                label_id = -4e7
                camera_id = -4e7
            dataset.append((path, label_id, camera_id, ii))
            dataset_dict[label_id].append((img_path, label_id, camera_id, ii))
        return dataset, dataset_dict


    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


    def print_dataset_statistics_movie(self, train, query, gallery):
        num_train_pids, num_train_imgs, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, _ = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  --------------------------------------")
        print("  subset         | # ids     | # images")
        print("  --------------------------------------")
        print("  train          | {:5d}     | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query          | {:5d}     | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery        | {:5d}     | {:8d}".format(num_gallery_pids, num_gallery_imgs))