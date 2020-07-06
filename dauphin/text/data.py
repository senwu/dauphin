import logging

from emmental.data import EmmentalDataLoader

from dauphin.text.datasets import ALL_DATASETS

logger = logging.getLogger(__name__)


def get_dataloaders(args):
    dataloaders = []
    datasets = {}

    for split in ["train", "test"]:
        if split == "train":
            datasets[split] = ALL_DATASETS[args.task](
                args.task, args, split, index=None, k=args.augment_k, model=args.model
            )
        elif split == "test":
            datasets[split] = ALL_DATASETS[args.task](
                args.task, args, split, model=args.model
            )

    for split, dataset in datasets.items():
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={args.task: "labels"},
                dataset=dataset,
                split=split,
                shuffle=True if split in ["train"] else False,
                batch_size=args.batch_size
                if split in args.train_split or args.valid_batch_size is None
                else args.valid_batch_size,
                num_workers=1,
            )
        )
        logger.info(
            f"Built dataloader for {args.task} {split} set with {len(dataset)} "
            f"samples (Shuffle={split in args.train_split}, "
            f"Batch size={dataloaders[-1].batch_size})."
        )

    return dataloaders
