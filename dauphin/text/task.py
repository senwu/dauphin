import logging
from functools import partial

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn
from torch.nn import functional as F

from dauphin.image.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss
from dauphin.text.config import TASK_METRIC, TASK_NUM_CLASS
from dauphin.text.modules.bert_model import BertModule

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

    bert_module = BertModule(args.model)
    bert_output_dim = 768 if "base" in args.model else 1024

    loss = sce_loss
    output = output_classification

    logger.info(f"Built model: {bert_module}")

    return EmmentalTask(
        name=args.task,
        module_pool=nn.ModuleDict(
            {
                "feature": bert_module,
                f"{task_name}_pred_head": nn.Linear(bert_output_dim, n_class),
            }
        ),
        task_flow=[
            {
                "name": "feature",
                "module": "feature",
                "inputs": [
                    ("_input_", "token_ids"),
                    ("_input_", "token_segments"),
                    ("_input_", "token_masks"),
                ],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("feature", 1)],
            },
        ],
        loss_func=partial(loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(metrics=TASK_METRIC[task_name]),
    )
