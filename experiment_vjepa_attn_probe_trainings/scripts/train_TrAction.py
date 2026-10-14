from pathlib import Path
import os
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply
from torch.utils.data import ConcatDataset
import yaml
import copy
import math
import argparse
import wandb

import src.dataset.transforms as T
from src.dataset.build import build_dataset
from src.models.build import build_model
from src.optim.build import build_optimizer
from src.optim.build import build_scheduler
from src.core.engine import train_epoch, val_epoch
from src.utils.misc import set_rng_seeds, nest_config, str2bool, dict_to_namespace, flatten_yaml, is_better, default_best, primary_metric, seed_worker
from src.utils.checkpoint import save_checkpoint, save_encoder_checkpoint, load_checkpoint, init_weights_from_checkpoints

# Vision backbone registry — single source of truth for backbone names.
from src.models.backbones import MODELS
# Setup registry
from src.models.build import SETUPS
# Tasks
from src.tasks import TASKS


def parse_args():

    # 1) Initial parser (config only)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default='/mnt/vast-nhr/projects/nib00021/repos_jan/action_from_motion/models/configs/traj_enc.yaml')
    cfg_args, remaining_argv = config_parser.parse_known_args()

    # 2) Load YAML defaults
    yaml_defaults = {}
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            # yaml_defaults = yaml.safe_load(f) or {}
            yaml_defaults = flatten_yaml(yaml.safe_load(f) or {})

    # 3) Full parser
    parser = argparse.ArgumentParser(parents=[config_parser])

    # task args
    parser.add_argument("--TASK", type=str, default='action_recognition',
                        choices=list(TASKS.keys()))
    parser.add_argument("--TRAIN", type=str2bool, default=True)
    parser.add_argument("--TEST", type=str2bool, default=False)
    parser.add_argument("--VAL", type=str2bool, default=True)
    parser.add_argument("--NUM_SEEDS", type=int, default=1)
    parser.add_argument("--FOLD", type=int, default=1)

    # output args
    parser.add_argument("--OUTPUT_DIR", type=str, default="outputs")
    parser.add_argument("--SAVE_PREDS", type=str2bool, default=False)
    parser.add_argument("--RUN_NAME", type=str, default="ssv2_vjepa_fused")

    # train args
    parser.add_argument("--MAX_EPOCHS", type=int, default=100)
    parser.add_argument("--DEVICE", type=str, default='cuda')
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--NUM_WORKERS", type=int, default=0)
    parser.add_argument("--RNG_SEED", type=int, default=42)

    # test args
    parser.add_argument("--TTA_VIEWS", type=int, default=1)

    # optimizer args
    parser.add_argument("--OPTIM.NAME", type=str, default='sgd')
    parser.add_argument("--OPTIM.LR", type=float, default=0.0025)
    parser.add_argument("--OPTIM.MOMENTUM", type=float, default=0.9)
    parser.add_argument("--OPTIM.WEIGHT_DECAY", type=float, default=1e-4)
    parser.add_argument("--OPTIM.ADAM_BETA1", type=float, default=0.9)
    parser.add_argument("--OPTIM.ADAM_BETA2", type=float, default=0.999)

    # lr scheduler args
    parser.add_argument("--SCHED.NAME", type=str, default='steplr')
    parser.add_argument("--SCHED.STEP_SIZE", type=int, default=25)
    parser.add_argument("--SCHED.GAMMA", type=float, default=0.1)
    parser.add_argument("--SCHED.WARMUP_RATIO", type=float, default=0.1)

    # dataset args
    parser.add_argument("--DATASET.NAME", type=str, default='ssv2')
    parser.add_argument("--DATASET.DATA_ROOT", type=str, default='/mnt/vast-nhr/projects/nib00021/data/public')
    parser.add_argument("--DATASET.FEAT_ROOT", type=str, default='data')
    parser.add_argument("--DATASET.CACHE", type=str2bool, default=True)
    parser.add_argument("--DATASET.IMGS", type=str2bool, default=False)
    parser.add_argument("--DATASET.CHIRALITY_TRAIN_DIR", type=str, default='chirality-in-action/chirality_ssv2/train')
    parser.add_argument("--DATASET.CHIRALITY_VAL_DIR", type=str, default='chirality-in-action/chirality_ssv2/val')
    parser.add_argument("--DATASET.TRAIN_LABEL_FILE", type=str, default=None)
    parser.add_argument("--DATASET.VAL_LABEL_FILE", type=str, default=None)

    # model args
    parser.add_argument("--MODEL.NAME", type=str, default='traj_supervised', choices=SETUPS)
    parser.add_argument("--MODEL.DIM_EMBED", type=int, default=288)
    parser.add_argument("--MODEL.NUM_LAYERS", type=int, default=6)
    parser.add_argument("--MODEL.NUM_HEADS", type=int, default=8)
    parser.add_argument("--MODEL.TRAJ_LEN", type=int, default=32)
    parser.add_argument("--MODEL.DIM_SINE", type=int, default=36)
    parser.add_argument("--MODEL.NUM_TRAJ", type=int, default=512)

    # probe args (is used on top of already existing encoders, e.g. DINOv2, V-JEPA)
    parser.add_argument("--VIS_EXTRACTOR.NAME", type=str, default=None, choices=MODELS)
    parser.add_argument("--VIS_EXTRACTOR.SIZE", type=str, default='base')
    parser.add_argument("--VIS_EXTRACTOR.SUFFIX", type=str, default='')
    parser.add_argument("--VIS_EXTRACTOR.NUM_FRAMES", type=int, default=16)

    # probe args (is used on top of already existing encoders, e.g. DINOv2, V-JEPA)
    parser.add_argument("--TRAJ_EXTRACTOR.NAME", type=str, default='cotracker')
    parser.add_argument("--TRAJ_EXTRACTOR.SIZE", type=str, default='large')
    parser.add_argument("--TRAJ_EXTRACTOR.SUFFIX", type=str, default='')

    # probe args (is used on top of already existing encoders, e.g. DINOv2, V-JEPA)
    parser.add_argument("--DEPTH_EXTRACTOR.NAME", type=str, default='vda')
    parser.add_argument("--DEPTH_EXTRACTOR.SIZE", type=str, default='large')
    parser.add_argument("--DEPTH_EXTRACTOR.SUFFIX", type=str, default='')

    # ── MAE (only used when TASK=mae_pretrain) ──
    parser.add_argument("--MAE.MASK_RATIO", type=float, default=0.9)
    parser.add_argument("--MAE.DECODER_DIM", type=int, default=144)
    parser.add_argument("--MAE.DECODER_NUM_LAYERS", type=int, default=2)
    parser.add_argument("--MAE.DECODER_NUM_HEADS", type=int, default=4)

    # ── JEPA (only used when TASK=jepa_pretrain) ──
    parser.add_argument("--JEPA.MASK_RATIO", type=float, default=0.75)
    parser.add_argument("--JEPA.PREDICTOR_DIM", type=int, default=144)
    parser.add_argument("--JEPA.PREDICTOR_NUM_LAYERS", type=int, default=2)
    parser.add_argument("--JEPA.PREDICTOR_NUM_HEADS", type=int, default=4)
    parser.add_argument("--JEPA.BLOCK_MASKING", type=str2bool, default=True)
    parser.add_argument("--JEPA.NUM_MASK_BLOCKS", type=int, default=4)
    parser.add_argument("--JEPA.EMA_BASE", type=float, default=0.996)

    # checkpoint args
    parser.add_argument("--CHECKPOINT.SAVE", type=str2bool, default=True)
    parser.add_argument("--CHECKPOINT.SAVE_DIR", type=str, default="checkpoints")
    parser.add_argument("--CHECKPOINT.RESUME", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--CHECKPOINT.LOAD_OPTIM", type=str2bool, default=False, help="Whether to restore optimizer and lr_scheduler state from checkpoint")
    parser.add_argument("--CHECKPOINT.INIT", type=str, default=None, #'.:/mnt/vast-nhr/projects/nib00021/repos_jan/action_from_motion/checkpoints/ssv2_mae_pretrain/best_encoder.pt'
                    help="Comma-separated key:path pairs to initialize submodules, e.g. 'model_a:/path/a.pt,model_b:/path/b.pt'. Use '.' for the full model.")

    # 4) Inject YAML as defaults
    if yaml_defaults:
        parser.set_defaults(**yaml_defaults)

    # 5) Final parse
    args = parser.parse_args(remaining_argv)

    # 6) Nest config
    cfg = nest_config(args)
    cfg = dict_to_namespace(cfg)
    return cfg


def build_task(cfg):
    task_name = cfg.TASK
    if task_name in TASKS:
        return TASKS[task_name]()
    else:
        raise ValueError(f"Unknown task: {task_name}")


def build_transforms(cfg):
    if cfg.TASK in ('mae_pretrain', 'jepa_pretrain'):
        train_tf = Compose([
            T.Normalize(),
            RandomApply([T.RandomGlobalTranslate()], p=0.3),
            T.RandomSampling(cfg.MODEL.NUM_TRAJ)
        ])
    elif cfg.DATASET.NAME == 'cia_ssv2':
        train_tf = Compose([T.Normalize(), T.RandomSampling(2500)])
    else:
        train_tf = Compose([
            T.Normalize(),
            RandomApply([T.RandomGlobalTranslate()], p=0.5),
            RandomApply([T.RandomGlobalRotate()], p=0.3),
            RandomApply([T.RandomTemporalJitter()], p=0.5),
            RandomApply([T.RandomTrajectoryJitter()], p=0.3),
            RandomApply([T.RandomVelocityNoise()], p=0.3),
            RandomApply([T.RandomSpatialCrop()], p=0.5),
            RandomApply([T.RandomSpeedPerturbation()], p=0.4),
            T.RandomSampling(cfg.MODEL.NUM_TRAJ)
        ])
         
    val_tf = Compose([T.Normalize(), T.RandomSampling(2500)])
   
    train_tf = Compose([
            T.Normalize3D(),
            RandomApply([T.RandomGlobalTranslate3D()], p=0.5),
            RandomApply([T.RandomGlobalRotate3D()], p=0.3),
            RandomApply([T.RandomTemporalJitter()], p=0.5),
            RandomApply([T.RandomTrajectoryJitter3D()], p=0.3),
            RandomApply([T.RandomVelocityNoise3D()], p=0.3),
            RandomApply([T.RandomSpatialCrop3D()], p=0.5),
            RandomApply([T.RandomSpeedPerturbation()], p=0.4),
            T.RandomSampling(cfg.MODEL.NUM_TRAJ)
        ])
    
    val_tf = Compose([T.Normalize3D(), T.RandomSampling(2500)])
    
    return train_tf, val_tf


def train(cfg):
    DATASET2 = False

    # load datasets
    if cfg.TRAIN:
        data_train = build_dataset(cfg, 'train')
        
    data_val = build_dataset(cfg, 'val')

    if DATASET2:
        cfg.DATASET.NAME = 'k400'
        cfg.MODEL.STRIDE = 2
        data2_train = build_dataset(cfg, 'train')
        data2_val = build_dataset(cfg, 'val')
        data_train = ConcatDataset([data_train, data2_train])
        data_val = ConcatDataset([data_val, data2_val])

    # create dataloader
    if cfg.TRAIN:
        g = torch.Generator()
        g.manual_seed(cfg.RNG_SEED)
        dl_train = DataLoader(data_train, 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            drop_last=len(data_train) > cfg.BATCH_SIZE,
            worker_init_fn=seed_worker, generator=g)

    dl_val = DataLoader(data_val, 
                        batch_size=cfg.BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=cfg.NUM_WORKERS)
    
    if cfg.TEST:
        data_test = build_dataset(cfg, 'test')
        dl_test = DataLoader(data_test, 
                        batch_size=cfg.BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=cfg.NUM_WORKERS)
        
    # build model
    model = build_model(cfg)
    task = build_task(cfg)

    # build scheduler
    num_training_steps = (cfg.MAX_EPOCHS * len(dl_train)) if cfg.TRAIN else 1
    
    if 'cia' in cfg.DATASET.NAME:
        if num_training_steps < 100:
            cfg.MAX_EPOCHS = cfg.MAX_EPOCHS * math.ceil(100/num_training_steps)
            num_training_steps = cfg.MAX_EPOCHS * len(dl_train)

    optim = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optim, num_training_steps)

    # transforms
    train_tf, val_tf = build_transforms(cfg)

    # checkpoint setup
    save_dir = Path(cfg.CHECKPOINT.SAVE_DIR) / cfg.RUN_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best = default_best(cfg.TASK)
    best_val_loss = float('inf')

    if cfg.CHECKPOINT.INIT is not None:
        init_weights_from_checkpoints(cfg.CHECKPOINT.INIT, model)

    if cfg.CHECKPOINT.RESUME is not None:
        start_epoch, saved_best = load_checkpoint(
            cfg.CHECKPOINT.RESUME, model, optim, lr_scheduler, cfg.CHECKPOINT.LOAD_OPTIM)
        if saved_best is not None:
            best = saved_best
        
    for epoch in range(start_epoch, cfg.MAX_EPOCHS):
        if cfg.TRAIN:
            print(f'---- Training Epoch: {epoch} ----')
            train_epoch(model, optim, lr_scheduler, dl_train, train_tf, task, epoch, num_training_steps=num_training_steps)
        
        if cfg.VAL:
            print(f'---- Validation Epoch: {epoch} ----')
            val_loss, val_metrics = val_epoch(model, dl_val, val_tf, task, epoch)

        score = primary_metric(cfg.TASK, val_loss, val_metrics)
        if is_better(cfg.TASK, score, best):
            best = score
            wandb.run.summary["best_metric"] = best
            wandb.run.summary["best_epoch"] = epoch
            if cfg.CHECKPOINT.SAVE:
                if cfg.TASK == 'mae_pretrain':
                    save_encoder_checkpoint(save_dir / "best_encoder.pt", model)
                else:
                    save_checkpoint(save_dir / "best_acc.pt", epoch, model, optim, lr_scheduler, best)
        
        if cfg.VAL and val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch
            if cfg.CHECKPOINT.SAVE:
                save_checkpoint(save_dir / "best_loss.pt", epoch, model, optim, lr_scheduler, best_val_loss)

        if (epoch + 1) % 50 == 0  and cfg.CHECKPOINT.SAVE:
            save_encoder_checkpoint(save_dir / f"ep{epoch}_encoder.pt", model)

        if (epoch + 1) == cfg.MAX_EPOCHS and cfg.CHECKPOINT.SAVE:
            save_checkpoint(save_dir / "last_acc.pt", epoch, model, optim, lr_scheduler, best)

    if cfg.TEST:
        load_checkpoint(save_dir / "best_acc.pt", model, optim, lr_scheduler, load_optim=False)
        print(f'---- Testing Epoch: {epoch} ----')
        test_loss, test_metrics = val_epoch(model, dl_test, val_tf, task, epoch, test=True)
        print(test_metrics)

    return score


def train_chirality(cfg):
    """Wrapper that runs train() for each chirality group and reports aggregate results."""
    train_dir = os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.CHIRALITY_TRAIN_DIR)
    val_dir = os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.CHIRALITY_VAL_DIR)

    # Pair train/val files by matching filename
    train_files = {f: os.path.join(train_dir, f) for f in sorted(os.listdir(train_dir)) if f.endswith('.txt')}
    val_files = {f: os.path.join(val_dir, f) for f in sorted(os.listdir(val_dir)) if f.endswith('.txt')}

    groups = []
    for fname in train_files:
        if fname in val_files:
            groups.append((fname.replace('.txt', ''), train_files[fname], val_files[fname]))
        else:
            print(f"WARNING: no val file for {fname}, skipping")

    print(f"Chirality: {len(groups)} groups found")

    base_run_name = cfg.RUN_NAME

    cfg.CHECKPOINT.SAVE = False
    
    results = {}
    for group_name, train_path, val_path in groups:
        print(f"\n{'='*30}")
        print(f"Group: {group_name}")
        print(f"{'='*30}")

        cfg.MAX_EPOCHS = 10

        # Override label files and run name for this group
        cfg.DATASET.TRAIN_LABEL_FILE = train_path.replace('.txt', '')
        cfg.DATASET.VAL_LABEL_FILE = val_path.replace('.txt', '')
        cfg.RUN_NAME = f"{base_run_name}/{group_name}"
        cfg.DATASET.NUM_CLASSES = 2  # chirality is always binary

        best = train(cfg)
        results[group_name] = best
        print(best)
        wandb.log({f"chirality/{group_name}/best": best})

    # Aggregate
    values = list(results.values())
    mean_acc = sum(values) / len(values)

    print(f"\n{'='*60}")
    print(f"Chirality Results ({len(results)} groups)")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
    print(f"  Mean: {mean_acc:.4f}")

    wandb.run.summary["chirality/mean_best"] = mean_acc
    wandb.run.summary["chirality/num_groups"] = len(results)

    return mean_acc


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    cfg = parse_args()

    base_seed = cfg.RNG_SEED
    base_run_name = cfg.RUN_NAME
    scores = []

    for i in range(cfg.NUM_SEEDS):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.RNG_SEED = base_seed + i
        run_cfg.RUN_NAME = f"{base_run_name}_seed{run_cfg.RNG_SEED}"

        if run_cfg.CHECKPOINT.INIT is not None:
            run_cfg.CHECKPOINT.INIT = run_cfg.CHECKPOINT.INIT.replace(f'seed42', f'seed{run_cfg.RNG_SEED}')

        run = wandb.init(
            project="action-from-motion",
            name=run_cfg.RUN_NAME,
            group=base_run_name,          # <-- ties the runs together
            job_type="train",
            config=run_cfg,
            reinit=True,                  # allow multiple init calls in one process
        )

        set_rng_seeds(run_cfg.RNG_SEED)
        if 'cia' in run_cfg.DATASET.NAME:
            score = train_chirality(run_cfg)
        else:
            score = train(run_cfg)

        scores.append(score)

        run.finish()


