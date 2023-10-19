#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import random
from typing import Optional

import numpy as np
import torch
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['MAGNUM_LOG'] = 'quiet'

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size

from src import (
    policy,
    environment,
    metric,
    sensor,
    trainer, 
    utils
)


def build_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "traverse"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
        help="debug using 1 scene"
    )
    parser.add_argument(
        "--note",
        default="",
        help="Add extra note for running file"
    )

    # distributed training parameters
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()
    run_exp(**vars(args))



def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """  
    
    
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported" 
    trainer = trainer_init(config)    # trainer : habitat_baselines.rl.ppo.ppo_trainer.PPOTrainer
    
    # if not os.path.exists(os.path.dirname(config.habitat_baselines.log_file)):
    #     os.makedirs(os.path.dirname(config.habitat_baselines.log_file))
    #     f=open(config.habitat_baselines.log_file,"w")
    #     f.close()
        

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "traverse":
        trainer.traverse()


def run_exp(exp_config: str, run_type: str, opts=None, model_dir=None, overwrite=False, note=None, debug=False, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval".
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    # print(opts)

    _, world_rank, _ = get_distrib_size()
    config = utils.get_config(exp_config, opts, run_type, model_dir, overwrite, note, debug, world_rank)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
