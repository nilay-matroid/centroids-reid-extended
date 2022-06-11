# encoding: utf-8
"""
@author: mikwieczorek
"""

import copy
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict, rank_zero_only
from torch import embedding, tensor
from tqdm import tqdm

from config import cfg
from losses.center_loss import CenterLoss
from losses.triplet_loss import CrossEntropyLabelSmooth, TripletLoss
from modelling.baseline import Baseline
from solver import build_optimizer, build_scheduler
from utils.reid_metric import R1_mAP


import shutil
from npy_append_array import NpyAppendArray
from time import time

import faiss
from tqdm import tqdm
import pickle as pkl

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def eval_market1501(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    # num_q, num_g = distmat.shape
    num_q = q_pids.shape[0]
    num_g = g_pids.shape[0]

    print("Num queries: ", num_q)
    print("Num gallery_images: ", num_g)

    import pdb
    pdb.set_trace()

    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    startx = time()
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("Time to take get matches: ", time() - startx)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    print(f"mAP: {mAP}")
    print(f"mINP: {mINP}")
    for r in [1, 5, 10, 20, 50]:
        print(f"Rank-{r}: {all_cmc[r-1]}")

    return


def eval_market1501_parallel(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat, num_workers=1):
    """Parallel Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    # num_q, num_g = distmat.shape
    num_q = q_pids.shape[0]
    num_g = g_pids.shape[0]

    print("Num queries: ", num_q)
    print("Num gallery_images: ", num_g)

    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    startx = time()
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("Time to take get matches: ", time() - startx)

    from multiprocessing import Process
    import math
    threads = []
    num_queries_per_thread = math.ceil(num_q / num_workers)
    results = {}

    for worker_idx in range(num_workers):
        start = worker_idx * num_queries_per_thread
        end = min((worker_idx + 1) * num_queries_per_thread, num_q)
        # No need of locks/mutexes as threads aren't modifying common variables.
        # But need to avoid creating new variables for each argument
        # threads.append(threading.Thread(target=compute_stats, args=(start, end, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results)))
        threads.append(Process(target=compute_stats, args=(start, end, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results)))

    # Start all threads
    for worker_idx in range(num_workers):
        print("Starting thread worker ", worker_idx)
        threads[worker_idx].start()

    # Wait for all threads to finish
    for worker_idx in range(num_workers):
        threads[worker_idx].join()

    # Aggregate results
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    temp_dir = "./temp_parallel_results"
    assert os.path.isdir(temp_dir)

    for worker_idx in range(num_workers):
        f = open(os.path.join(temp_dir, f"results_{worker_idx}.pkl"), "rb")
        results = pkl.load(f)
        f.close()
        all_cmc += results[worker_idx]["all_cmc"]
        all_AP += results[worker_idx]["all_AP"]
        all_INP += results[worker_idx]["all_INP"]
        num_valid_q += results[worker_idx]["num_valid_q"]

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    # Delete the temp files and folders
    print("Deleting all temporary folders and files")
    shutil.rmtree(temp_dir, ignore_errors=True)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    print(f"mAP: {mAP}")
    print(f"mINP: {mINP}")
    for r in [1, 5, 10, 20, 50]:
        print(f"Rank-{r}: {all_cmc[r-1]}")

    return


def compute_stats(start, stop, q_pids, q_camids, g_pids, g_camids, indices, matches, max_rank, worker_idx, results):
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(start, stop)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    results[worker_idx] = {}
    results[worker_idx]["all_cmc"] = all_cmc
    results[worker_idx]["all_AP"] = all_AP
    results[worker_idx]["all_INP"] = all_INP
    results[worker_idx]["num_valid_q"] = num_valid_q

    temp_dir = "./temp_parallel_results"
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

    result_save_file = os.path.join(temp_dir, f"results_{worker_idx}.pkl")
    f = open(result_save_file, 'wb')
    pkl.dump(results, f)
    f.close() 
    return


class ModelBase(pl.LightningModule):
    def __init__(self, cfg=None, test_dataloader=None, **kwargs):
        super().__init__()

        if cfg is None:
            hparams = {**kwargs}
        elif isinstance(cfg, dict):
            hparams = {**cfg, **kwargs}
            if cfg.TEST.ONLY_TEST:
                # To make sure that loaded hparams are overwritten by cfg we may have chnaged
                hparams = {**kwargs, **cfg}
        self.hparams = AttributeDict(hparams)
        self.save_hyperparameters(self.hparams)

        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

        # Create backbone model
        self.backbone = Baseline(self.hparams)

        self.contrastive_loss = TripletLoss(
            self.hparams.SOLVER.MARGIN, self.hparams.SOLVER.DISTANCE_FUNC
        )

        d_model = self.hparams.MODEL.BACKBONE_EMB_SIZE
        self.xent = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes)
        self.center_loss = CenterLoss(
            num_classes=self.hparams.num_classes, feat_dim=d_model
        )
        self.center_loss_weight = self.hparams.SOLVER.CENTER_LOSS_WEIGHT

        self.bn = torch.nn.BatchNorm1d(d_model)
        self.bn.bias.requires_grad_(False)

        self.fc_query = torch.nn.Linear(d_model, self.hparams.num_classes, bias=False)
        self.fc_query.apply(weights_init_classifier)

        self.losses_names = ["query_xent", "query_triplet", "query_center"]
        self.losses_dict = {n: [] for n in self.losses_names}

        if self.hparams.TEST.CACHE.ENABLED:
            self.cache_dir = self.hparams.TEST.CACHE.CACHE_DIR
            assert self.cache_dir is not None, 'Oops, no cache dir specified'
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)            

    def set_test_loader(self, test_loader):
        assert test_loader is not None
        self.test_dataloader = test_loader

    def reset_cache(self):
        print("Removing previously saved cache files ...")
        self.delete_saved_file(self.embedding_file)
        self.delete_saved_file(self.pid_file)
        self.delete_saved_file(self.camid_file)

    def delete_saved_file(self, file):
        if os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def _calculate_centroids(vecs, dim=1):
        length = vecs.shape[dim]
        return torch.sum(vecs, dim) / length

    def configure_optimizers(self):
        optimizers_list = build_optimizer(self.named_parameters(), self.hparams)
        self.lr_scheduler = build_scheduler(optimizers_list[0], self.hparams)
        return optimizers_list, self.lr_scheduler

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
        **kwargs,
    ):

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0, float(epoch + 1) / float(self.hparams.SOLVER.WARMUP_EPOCHS)
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR

        super().optimizer_step(
            epoch=epoch,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
            **kwargs,
        )

    def training_step(self, batch, batch_idx, opt_idx=None):
        raise NotImplementedError(
            "A used model should have its own training_step method implemented"
        )

    def training_epoch_end(self, outputs):
        if hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch + 1)

        lr = self.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])
        l2_mean_norm = np.mean([x["other"].pop("l2_mean_centroid") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
            "l2_mean_centroid": l2_mean_norm,
        }

        if hasattr(self, "losses_dict"):
            for name, loss_val in self.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                self.losses_dict[name] = []  ## Zeroing values after a completed epoch

        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)
        self.trainer.accelerator_backend.barrier()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        self.backbone.eval()
        self.bn.eval()
        x, class_labels, camid, idx = batch
        with torch.no_grad():
            _, emb = self.backbone(x)
            emb = self.bn(emb)
        return {"emb": emb, "labels": class_labels, "camid": camid, "idx": idx}


    def batched_cached_inference(self):
        assert self.hparams.TEST.CACHE.ENABLED
        assert self.hparams.TEST.CACHE.CACHE_DIR is not None

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.set_feat_files("CTL_cached_inference")        

        # Check if these files already exist and if yes, delete them
        self.reset_cache()

        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            batch = [x.cuda() for x in batch]
            outputs = self.test_step(batch, batch_idx)
            embeddings = outputs['emb'].detach().cpu()
            labels = outputs["labels"].detach().cpu().numpy()
            camids = outputs["camid"].cpu().detach().numpy()
            del outputs

            NpyAppendArray(self.pid_file).append(labels)
            NpyAppendArray(self.camid_file).append(camids)
            NpyAppendArray(self.embedding_file).append(embeddings.numpy())


    def batched_validation_create_centroids(self):
        assert self.hparams.TEST.CACHE.ENABLED
        assert self.hparams.TEST.CACHE.CACHE_DIR is not None
        assert os.path.isdir(self.cache_dir), "Oops no cache directory found"
        assert os.path.isfile(self.embedding_file), "No embeddings found"
        assert os.path.isfile(self.pid_file), "No person id file found"
        assert os.path.isfile(self.camid_file), "No camera id file found"

        embeddings = torch.tensor(np.load(self.embedding_file, mmap_mode='r+'))
        labels = np.load(self.pid_file, mmap_mode='r+')
        camids = np.load(self.camid_file, mmap_mode='r+')

        embeddings, labels, camids = self.validation_create_centroids(
            embeddings,
            labels,
            camids,
            respect_camids=self.hparams.MODEL.KEEP_CAMID_CENTROIDS,
        )

        embeddings = embeddings.numpy()
        np.save(self.embedding_file, embeddings)

        # TODO(nilay.pande): Check below lines can be removed while retaining correctness
        np.save(self.pid_file, labels)
        np.save(self.camid_file, camids)


    def parallelized_cached_eval(self):
        assert self.hparams.TEST.CACHE.FEAT_REUSE.ENABLED
        assert self.hparams.TEST.CACHE.FEAT_REUSE.PREFIX is not None, 'Oops, no prefix specified for the features to be used'

        do_parallel = self.hparams.TEST.CACHE.PARALLEL.ENABLED
        if do_parallel:
            num_workers = self.hparams.TEST.CACHE.PARALLEL.NUM_WORKERS
            assert num_workers > 1, "Oops, need to specify more than one worker for parallelization"

        assert os.path.isdir(self.cache_dir), "Oops no cache directory found"

        self.set_feat_files(self.hparams.TEST.CACHE.FEAT_REUSE.PREFIX)
        assert os.path.isfile(self.embedding_file), "No embeddings found"
        assert os.path.isfile(self.pid_file), "No person id file found"
        assert os.path.isfile(self.camid_file), "No camera id file found"

        features = np.load(self.embedding_file, allow_pickle=True).astype(np.float32)
        pids = np.load(self.pid_file, allow_pickle=True).astype(np.float32)
        camids =  np.load(self.camid_file, allow_pickle=True)
        num_query = self.hparams.num_query

        q_feats = features[:num_query]
        q_pids = pids[:num_query]
        q_camids = camids[:num_query]

        g_feats = features[num_query:]
        g_pids = pids[num_query:]
        g_camids = camids[num_query:]

        # TODO(nilay.pande): Add support for re-ranking in future
        # Using only faiss for now
        distmat = None
        use_distmat = False
        max_rank = 50

        if do_parallel:
            eval_market1501_parallel(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat, num_workers)
        else:
            eval_market1501(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat)
        return

    @rank_zero_only
    def validation_create_centroids(
        self, embeddings, labels, camids, respect_camids=False
    ):
        num_query = self.hparams.num_query
        # Keep query data samples seperated
        embeddings_query = embeddings[:num_query].cpu()
        labels_query = labels[:num_query]

        # Process gallery samples further
        embeddings_gallery = embeddings[num_query:]
        labels_gallery = labels[num_query:]

        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[label].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[label].append(idx)

        unique_labels = sorted(np.unique(list(labels2idx.keys())))

        centroids_embeddings = []
        centroids_labels = []

        if respect_camids:
            centroids_camids = []
            query_camid = camids[:num_query]

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if respect_camids:
                selected_camids_g = camids[inds]

                selected_camids_q = camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = sorted(
                        np.unique(
                            [cid for cid in selected_camids_g if cid != current_camid]
                        )
                    )
                    if tuple(used_camids) not in cmaids_combinations:
                        cmaids_combinations.add(tuple(used_camids))
                        centroids_emb = embeddings_gallery[inds][camid_inds]
                        centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                        centroids_embeddings.append(centroids_emb.detach().cpu())
                        centroids_camids.append(used_camids)
                        centroids_labels.append(label)

            else:
                centroids_labels.append(label)
                centroids_emb = embeddings_gallery[inds]
                centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                centroids_embeddings.append(centroids_emb.detach().cpu())

        # Make a single tensor from query and gallery data
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        centroids_embeddings = torch.cat(
            (embeddings_query, centroids_embeddings), dim=0
        )
        centroids_labels = np.hstack((labels_query, np.array(centroids_labels)))

        if respect_camids:
            query_camid = [[item] for item in query_camid]
            centroids_camids = query_camid + centroids_camids

        if not respect_camids:
            # Create dummy camids for query na gallery features
            # it is used in eval_reid script
            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack((camids_query, np.array(camids_gallery)))

        return centroids_embeddings.cpu(), centroids_labels, centroids_camids

    @rank_zero_only
    def get_val_metrics(self, embeddings, labels, camids):
        self.r1_map_func = R1_mAP(
            pl_module=self,
            num_query=self.hparams.num_query,
            feat_norm=self.hparams.TEST.FEAT_NORM,
        )
        respect_camids = (
            True
            if (
                self.hparams.MODEL.KEEP_CAMID_CENTROIDS
                and self.hparams.MODEL.USE_CENTROIDS
            )
            else False
        )
        cmc, mAP, all_topk = self.r1_map_func.compute(
            feats=embeddings.float(),
            pids=labels,
            camids=camids,
            respect_camids=respect_camids,
        )

        topks = {}
        for top_k, kk in zip(all_topk, [1, 5, 10, 20, 50]):
            print("top-k, Rank-{:<3}:{:.1%}".format(kk, top_k))
            topks[f"Top-{kk}"] = top_k
        print(f"mAP: {mAP}")

        log_data = {"mAP": mAP}

        # TODO This line below is hacky, but it works when grad_monitoring is active
        self.trainer.logger_connector.callback_metrics.update(log_data)
        log_data = {**log_data, **topks}
        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)

    
    def set_feat_files(self, prefix):
        use_centroids = "centroids" if self.hparams.MODEL.USE_CENTROIDS else "no_centroids" 
        prefix = f"{prefix}_{self.hparams.MODEL.NAME}_{self.hparams.DATASETS.NAMES}_{use_centroids}"
        self.embedding_file = os.path.join(self.cache_dir, f"{prefix}_feat.npy")
        self.pid_file = os.path.join(self.cache_dir, f"{prefix}_pid.npy")
        self.camid_file = os.path.join(self.cache_dir, f"{prefix}_camid.npy")

    def save_output(self, embeddings, labels, camids):
        self.set_feat_files("CTL")

        # Deleting any previously existing npy files for sake of clarity
        self.reset_cache()

        # Save features
        np.save(self.embedding_file, embeddings)
        np.save(self.pid_file, labels)
        np.save(self.camid_file, camids)

    def validation_epoch_end(self, outputs):
        if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
            embeddings = torch.cat([x.pop("emb") for x in outputs]).detach().cpu()
            labels = (
                torch.cat([x.pop("labels") for x in outputs]).detach().cpu().numpy()
            )
            camids = torch.cat([x.pop("camid") for x in outputs]).cpu().detach().numpy()
            del outputs
            if self.hparams.MODEL.USE_CENTROIDS:
                print("Evaluation is done using centroids")
                embeddings, labels, camids = self.validation_create_centroids(
                    embeddings,
                    labels,
                    camids,
                    respect_camids=self.hparams.MODEL.KEEP_CAMID_CENTROIDS,
                )
            if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
                if self.hparams.TEST.CACHE.ENABLED:
                    self.save_output(embeddings, labels, camids)
                self.get_val_metrics(embeddings, labels, camids)
            del embeddings, labels, camids
        self.trainer.accelerator_backend.barrier()

    @rank_zero_only
    def eval_on_train(self):
        if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
            outputs = []
            device = list(self.backbone.parameters())[0].device
            for batch_idx, batch in enumerate(self.test_dataloader):
                x, class_labels, camid, idx = batch
                with torch.no_grad():
                    emb = self.backbone(x.to(device))
                outputs.append(
                    {"emb": emb, "labels": class_labels, "camid": camid, "idx": idx}
                )

            embeddings = torch.cat([x["emb"] for x in outputs]).detach()
            labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
            camids = torch.cat([x["camid"] for x in outputs]).cpu().detach().numpy()
            inds = torch.cat([x["idx"] for x in outputs]).cpu().detach().numpy()

            embeddings, labels, camids = self.validation_create_centroids(
                embeddings, labels, camids
            )

            self.r1_map_func = R1_mAP(self.hparams.num_query)
            cmc, mAP, all_topk = self.r1_map_func.compute(
                feats=embeddings, pids=labels, camids=camids
            )

            topks = {}
            for top_k, kk in zip(all_topk, [1, 5, 10, 20, 50]):
                print("Train top-k, Rank-{:<3}:{:.1%}".format(kk, top_k))
                topks[f"Train Top-{kk}"] = top_k
            print(f"Train mAP: {mAP}")

            log_data = {"Train mAP": mAP}
            log_data = {**log_data, **topks}
            for key, val in log_data.items():
                tensorboard = self.logger.experiment
                tensorboard.add_scalar(key, val, self.current_epoch)

    @staticmethod
    def create_masks_train(class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)
        labels_list = [v for k, v in labels_dict.items()]
        labels_list_copy = copy.deepcopy(labels_list)
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        max_gal_num = max(
            [len(item) for item in labels_dict.values()]
        )  ## TODO Should allow usage of all permuations

        masks = torch.ones((max_gal_num, len(class_labels)), dtype=bool)
        for _ in range(max_gal_num):
            for i, inner_list in enumerate(labels_list):
                if len(inner_list) > 0:
                    masks[_, inner_list.pop(0)] = 0
                else:
                    start_ind = lens_list_cs[i - 1]
                    end_ind = start_ind + lens_list[i]
                    masks[_, start_ind:end_ind] = 0

        return masks, labels_list_copy

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        ret = self.validation_step(batch, batch_idx)
        return ret

    @rank_zero_only
    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)