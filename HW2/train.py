import argparse
import os
from pathlib import Path
import sys


# Thin training launcher that adapts the generated Co-DETR config at runtime.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch MMDetection/Co-DETR training through Hugging Face Accelerate."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the generated MMDetection config file.")
    parser.add_argument("--work-dir", type=str, default=None, help="Optional override for cfg.work_dir.")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Optional path to a cloned repo containing projects/CO-DETR (e.g., external/mmdetection).",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default=None,
        help="Override the launcher. By default this script uses 'pytorch' when Accelerate starts multiple processes.",
    )
    parser.add_argument(
        "--amp",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Whether to keep AMP from config. Default 'off' avoids MMCV NMS half/float mismatch.",
    )
    parser.add_argument(
        "--find-unused-parameters",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Enable DDP unused-parameter detection. 'auto' enables it for multi-GPU launches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # `expandable_segments` can trigger allocator internal asserts on some
    # PyTorch/CUDA builds, especially in distributed Co-DETR runs.
    # Keep a conservative allocator setting here instead.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    os.environ.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")

    if args.repo_root is not None:
        repo_root = str(Path(args.repo_root).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise ImportError(
            "MMDetection/MMEngine is not installed. Install the Co-DETR stack first; see requirements-codetr.txt."
        ) from exc

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    print(f"[train.py] script={Path(__file__).resolve()}")
    print(f"[train.py] config={Path(args.config).resolve()}")

    # Avoid AMP+MMCV NMS dtype mismatch by default.
    optim_wrapper = cfg.get("optim_wrapper", None)
    before_type = optim_wrapper.get("type", "<missing>") if isinstance(optim_wrapper, dict) else type(optim_wrapper).__name__
    print(f"[train.py] optim_wrapper(before)={before_type}, amp_arg={args.amp}")
    if args.amp == "off" and isinstance(optim_wrapper, dict) and optim_wrapper.get("type") == "AmpOptimWrapper":
        new_optim_wrapper = dict(optim_wrapper)
        new_optim_wrapper["type"] = "OptimWrapper"
        new_optim_wrapper.pop("loss_scale", None)
        cfg.optim_wrapper = new_optim_wrapper
    after_optim_wrapper = cfg.get("optim_wrapper", None)
    after_type = (
        after_optim_wrapper.get("type", "<missing>")
        if isinstance(after_optim_wrapper, dict)
        else type(after_optim_wrapper).__name__
    )
    print(f"[train.py] optim_wrapper(after)={after_type}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launcher = args.launcher
    if launcher is None:
        launcher = "pytorch" if world_size > 1 else "none"

    cfg.launcher = launcher

    enable_find_unused = args.find_unused_parameters == "on" or (
        args.find_unused_parameters == "auto" and world_size > 1
    )
    if enable_find_unused:
        cfg.find_unused_parameters = True
        model_wrapper_cfg = dict(cfg.get("model_wrapper_cfg", {}))
        model_wrapper_cfg["type"] = "MMDistributedDataParallel"
        model_wrapper_cfg["find_unused_parameters"] = True
        cfg.model_wrapper_cfg = model_wrapper_cfg
        print("[train.py] Enabled find_unused_parameters for distributed training")

    if args.work_dir is not None:
        # Allow the same generated config to be reused across different runs.
        cfg.work_dir = str(Path(args.work_dir).resolve())

    if args.resume_from is not None:
        # MMEngine needs both the checkpoint path and the resume flag.
        cfg.resume_from = str(Path(args.resume_from).resolve())
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
