import logging
from functools import partial

import numpy as np
import torch
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn
from torch.nn import functional as F

from dauphin.image.config import TASK_INPUT_SIZE, TASK_METRIC, TASK_NUM_CLASS
from dauphin.image.models import ALL_MODELS
from dauphin.image.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss

logger = logging.getLogger(__name__)


SCE = SoftCrossEntropyLoss(reduction="none")


def sce_loss(module_name, intermediate_output_dict, Y, active):
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_name][0].new_zeros(
            intermediate_output_dict[module_name][0].size()
        )
        label.scatter_(1, Y.view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return SCE(intermediate_output_dict[module_name][0][active], label[active])


def output_classification(module_name, immediate_output_dict):
    return F.softmax(immediate_output_dict[module_name][0], dim=1)


def create_task(args):
    task_name = args.task
    n_class = TASK_NUM_CLASS[args.task]

    if args.model in ["wide_resnet"]:
        feature_extractor = ALL_MODELS[args.model](
            args.wide_resnet_depth,
            args.wide_resnet_width,
            args.wide_resnet_dropout,
            n_class,
            has_fc=False,
        )
        n_hidden_dim = feature_extractor(
            torch.randn(TASK_INPUT_SIZE[args.task])
        ).size()[-1]

    elif args.model == "mlp":
        n_hidden_dim = args.mlp_hidden_dim
        input_dim = np.prod(TASK_INPUT_SIZE[args.task])
        feature_extractor = ALL_MODELS[args.model](
            input_dim, n_hidden_dim, n_class, has_fc=False
        )
    elif args.model == "shake_shake":
        feature_extractor = ALL_MODELS[args.model](
            args.shake_shake_depth,
            args.shake_shake_base_channels,
            args.shake_shake_shake_forward,
            args.shake_shake_shake_backward,
            args.shake_shake_shake_image,
            TASK_INPUT_SIZE[args.task],
            n_class,
            has_fc=False,
        )
        n_hidden_dim = feature_extractor.feature_size
    elif args.model == "pyramidnet":
        feature_extractor = ALL_MODELS[args.model](
            args.task,
            args.pyramidnet_depth,
            args.pyramidnet_alpha,
            args.pyramidnet_bottleneck,
            has_fc=False,
        )
        n_hidden_dim = feature_extractor.final_featuremap_dim
    else:
        raise ValueError(f"Invalid model {args.model}")

    loss = sce_loss
    output = output_classification

    logger.info(f"Built model: {feature_extractor}")

    return EmmentalTask(
        name=args.task,
        module_pool=nn.ModuleDict(
            {
                "feature": feature_extractor,
                f"{task_name}_pred_head": nn.Linear(n_hidden_dim, n_class),
            }
        ),
        task_flow=[
            {"name": "feature", "module": "feature", "inputs": [("_input_", "image")]},
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("feature", 0)],
            },
        ],
        loss_func=partial(loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(metrics=TASK_METRIC[task_name]),
    )
