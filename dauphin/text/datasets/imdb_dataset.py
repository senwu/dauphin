import logging

import numpy as np
import torch
from emmental.data import EmmentalDataset
from emmental.utils.utils import pred_to_prob
from torchtext import data, datasets

from dauphin.text.augment_policy import Augmentation
from dauphin.text.config import TASK_NUM_CLASS
from dauphin.text.datasets.utils import build_vocab, clean_web_text, get_data_stats
from dauphin.text.transforms.compose import Compose
from dauphin.text.transforms.to_tensor import ToTensor

logger = logging.getLogger(__name__)


class ImdbDataset(EmmentalDataset):
    """Dataset to load Imdb dataset."""

    def __init__(
        self,
        name,
        args,
        split="train",
        transform_cls=None,
        index=None,
        k=1,
        model="bert-base-uncased",
    ):
        X_dict, Y_dict = {"text_name": [], "text": []}, {"labels": []}

        TEXT = data.Field()
        LABEL = data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        dataset = train_data if split == "train" else test_data

        if index is None:
            for i, sample in enumerate(dataset.examples):
                x = clean_web_text(" ".join(sample.text)).split(" ")
                y = 1 if sample.label == "pos" else 0
                X_dict["text_name"].append(f"{name}_{split}_{i}")
                X_dict["text"].append(x)
                Y_dict["labels"].append(y)
        else:
            for id in index:
                x = clean_web_text(" ".join(dataset.examples[id].text)).split(" ")
                y = 1 if dataset.examples[id].label == "pos" else 0
                X_dict["image_name"].append(f"{name}_{split}_{id}")
                X_dict["image"].append(x)
                Y_dict["labels"].append(y)

        labels = pred_to_prob(np.array(Y_dict["labels"]), TASK_NUM_CLASS[name])

        Y_dict["labels"] = torch.from_numpy(labels)

        self.data_stats = get_data_stats(dataset.examples)
        self.vocab = build_vocab(dataset.examples)

        self.transform_cls = transform_cls
        if split == "train":
            self.transform_cls = Augmentation(args, self.vocab, self.data_stats)
        self.transforms = None

        self.defaults = [ToTensor(model=model)]

        # How many augmented samples to augment for each sample
        self.k = k if k is not None else 1

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="text_name")

    def gen_transforms(self):
        if self.transform_cls is not None:
            return self.transform_cls()
        else:
            return []

    def __getitem__(self, index):
        """Get item by index.

        Args:
          index(index): The index of the item.

        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict

        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        new_x_dict = {}
        new_y_dict = {"labels": []}

        for name, feature in x_dict.items():
            if name not in new_x_dict:
                new_x_dict[name] = []
            if name == self.uid:
                for i in range(self.k):
                    new_x_dict[name].append(f"{feature}_{i}")
            elif name == "text":
                for i in range(self.k):
                    # Save the original tokens
                    new_x_dict[name].append(feature)

                    if self.transform_cls is None:
                        self.transforms = self.defaults
                    else:
                        self.transforms = self.gen_transforms() + self.defaults

                    new_text_dict, new_label = Compose(self.transforms)(
                        feature,
                        y_dict["labels"],
                        X_dict=self.X_dict,
                        Y_dict=self.Y_dict,
                        transforms=self.transforms,
                    )

                    if "transforms" not in new_x_dict:
                        new_x_dict["transforms"] = []
                    new_x_dict["transforms"].append(self.transforms)
                    if index == 0:
                        logger.info(self.transforms)
                    for k, v in new_text_dict.items():
                        if k not in new_x_dict:
                            new_x_dict[k] = []
                        new_x_dict[k].append(v)
                    new_y_dict["labels"].append(new_label)
            else:
                for i in range(self.k):
                    new_x_dict[name].append(feature)
        for name, feature in y_dict.items():
            if name not in new_y_dict:
                new_y_dict[name] = []
            if name != "labels":
                for i in range(self.k):
                    new_y_dict[name].append(feature)

        return new_x_dict, new_y_dict
