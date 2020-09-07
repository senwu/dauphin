import numpy as np
import torch
from emmental.data import EmmentalDataset
from emmental.utils.utils import pred_to_prob

from dauphin.image.config import TASK_NORMALIZE, TASK_NUM_CLASS
from dauphin.image.transforms.compose import Compose
from dauphin.image.transforms.normalize import Normalize
from dauphin.image.transforms.to_tensor import ToTensor


class ImageNetDataset(EmmentalDataset):
    """Dataset to load ImageNet dataset."""

    def __init__(
        self,
        name,
        dataset,
        split="train",
        transform_cls=None,
        index=None,
        prob_label=False,
        k=1,
    ):
        X_dict, Y_dict = {"image_name": [], "image": []}, {"labels": []}
        if index is None:
            for i, (x, y) in enumerate(dataset):
                X_dict["image_name"].append(f"{name}_{split}_{i}")
                X_dict["image"].append(x)
                Y_dict["labels"].append(y)
        else:
            for id in index:
                x, y = dataset[id]
                X_dict["image_name"].append(f"{name}_{split}_{id}")
                X_dict["image"].append(x)
                Y_dict["labels"].append(y)

        if prob_label:
            labels = pred_to_prob(np.array(Y_dict["labels"]), TASK_NUM_CLASS[name])
        else:
            labels = np.array(Y_dict["labels"])

        Y_dict["labels"] = torch.from_numpy(labels)

        self.transform_cls = transform_cls
        self.transforms = None

        self.defaults = [
            ToTensor(),
            Normalize(TASK_NORMALIZE[name]["mean"], TASK_NORMALIZE[name]["std"]),
        ]

        # How many augmented samples to augment for each sample
        self.k = k if k is not None else 1

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="image_name")

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
            elif name == "image":
                for i in range(self.k):
                    if self.transform_cls is None:
                        self.transforms = self.defaults
                    else:
                        self.transforms = self.gen_transforms() + self.defaults

                    new_img, new_label = Compose(self.transforms)(
                        feature,
                        y_dict["labels"],
                        X_dict=self.X_dict,
                        Y_dict=self.Y_dict,
                        transforms=self.transforms,
                    )
                    new_x_dict[name].append(new_img)
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
