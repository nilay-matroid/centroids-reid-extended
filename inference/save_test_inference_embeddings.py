from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pathlib import Path

import argparse
import sys
sys.path.insert(0, '../centroids-reid-extended')
sys.path.insert(0, '../')
print(sys.path)
from datasets import init_dataset
from train_ctl_model import CTLModel
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLT Model Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    logger_save_dir = f"{Path(__file__).stem}"
    logger = TensorBoardLogger(cfg.LOG_DIR, name=logger_save_dir)
    mlflow_logger = MLFlowLogger(experiment_name="default")
    loggers = [logger, mlflow_logger]

    dm = init_dataset(
        cfg.DATASETS.NAMES, cfg=cfg, num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    dm.setup()
    val_dataloader = dm.val_dataloader()

    method = CTLModel
    if cfg.TEST.ONLY_TEST:
        method = method.load_from_checkpoint(
            cfg.MODEL.PRETRAIN_PATH,
            cfg=cfg,
            num_query=dm.num_query,
            num_classes=dm.num_classes,
            use_multiple_loggers=True if len(loggers) > 1 else False,
        )

        method.set_test_loader(val_dataloader)
        method = method.cuda()
        method.hparams.MODEL.USE_CENTROIDS = True
        method.batched_cached_inference()
        if method.hparams.MODEL.USE_CENTROIDS:
            method.batched_validation_create_centroids()