import logging

import torchvision
from emmental.data import EmmentalDataLoader

from dauphin.image.datasets import ALL_DATASETS

logger = logging.getLogger(__name__)


def get_dataloaders(args):
    if args.task.upper() != 'IMAGENET':
        train_dataset = torchvision.datasets.__dict__[args.task.upper()](
            root=args.data, train=True, download=True
        )
        test_dataset = torchvision.datasets.__dict__[args.task.upper()](
            root=args.data, train=False, download=True
        )
    else:
        train_dataset = ImageNet(root=os.path.join(args.data, 'imagenet-pytorch'))
        test_dataset = ImageNet(root=os.path.join(args.data, 'imagenet-pytorch'), split='val')

    dataloaders = []
    datasets = {}

    for split in ["train", "test"]:
        if split == "train":
            datasets[split] = ALL_DATASETS[args.task](
                args.task,
                train_dataset,
                split,
                index=None,
                prob_label=True,
                k=args.augment_k,
            )
        elif split == "test":
            datasets[split] = ALL_DATASETS[args.task](args.task, test_dataset, split)

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
                num_workers=4,
            )
        )
        logger.info(
            f"Built dataloader for {args.task} {split} set with {len(dataset)} "
            f"samples (Shuffle={split in args.train_split}, "
            f"Batch size={dataloaders[-1].batch_size})."
        )

    return dataloaders
