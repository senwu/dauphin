import logging
import sys

import emmental
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args_to_config

from dauphin.text.data import get_dataloaders
from dauphin.text.scheduler import AugScheduler
from dauphin.text.task import create_task
from dauphin.utils import write_to_file, write_to_json_file

logger = logging.getLogger(__name__)


def main(args):
    # Initialize Emmental
    config = parse_args_to_config(args)
    emmental.init(log_dir=config["meta_config"]["log_path"], config=config)

    # Log configuration into files
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{emmental.Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {emmental.Meta.config}")
    write_to_file(f"{emmental.Meta.log_path}/config.txt", emmental.Meta.config)

    # Create dataloaders
    dataloaders = get_dataloaders(args)

    config["learner_config"]["task_scheduler_config"]["task_scheduler"] = AugScheduler(
        augment_k=args.augment_k, enlarge=args.augment_enlarge
    )
    emmental.Meta.config["learner_config"]["task_scheduler_config"][
        "task_scheduler"
    ] = config["learner_config"]["task_scheduler_config"]["task_scheduler"]

    # Specify parameter group for Adam BERT
    def grouped_parameters(model):
        no_decay = ["bias", "LayerNorm.weight"]
        return [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": emmental.Meta.config["learner_config"][
                    "optimizer_config"
                ]["l2"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    emmental.Meta.config["learner_config"]["optimizer_config"][
        "parameters"
    ] = grouped_parameters

    # Create tasks
    model = EmmentalModel(name=f"{args.task}_task")
    model.add_task(create_task(args))

    # Load the best model from the pretrained model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # Remove all extra augmentation policy
    for idx in range(len(dataloaders)):
        dataloaders[idx].dataset.transform_cls = None
        dataloaders[idx].dataset.k = 1

    scores = model.score(dataloaders)

    # Save metrics and models
    logger.info(f"Metrics: {scores}")
    scores["log_path"] = emmental.Meta.log_path
    write_to_json_file(f"{emmental.Meta.log_path}/metrics.txt", scores)
    model.save(f"{emmental.Meta.log_path}/last_model.pth")
