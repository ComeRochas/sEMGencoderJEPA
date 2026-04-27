from __future__ import annotations

import os


def init_wandb(args):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--wandb was set but wandb is not installed. Please run: pip install wandb") from exc

    os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=vars(args),
    )


def wandb_log(run, metrics: dict):
    if run is not None:
        run.log(metrics)


def finish_wandb(run):
    if run is not None:
        run.finish()
