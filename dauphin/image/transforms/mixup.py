import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor

from dauphin.image.transforms.compose import Compose
from dauphin.image.transforms.transform import DauphinTransform
from dauphin.image.utils import default_loader


class Mixup(DauphinTransform):
    def __init__(
        self,
        name=None,
        prob=1.0,
        level=0,
        alpha=1.0,
        same_class_ratio=-1.0,
        prob_label=False,
    ):
        self.alpha = alpha
        self.same_class_ratio = same_class_ratio
        self.prob_label = prob_label

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        X_dict = kwargs["X_dict"]
        Y_dict = kwargs["Y_dict"]
        transforms = kwargs["transforms"]

        if self.alpha > 0.0:
            mix_ratio = np.random.beta(self.alpha, self.alpha)
        else:
            mix_ratio = 1.0

        if "image" in X_dict:
            idx = np.random.randint(len(X_dict["image"]))
            tot_cnt = len(X_dict["image"])
        else:
            idx = np.random.randint(len(X_dict["image_path"]))
            tot_cnt = len(X_dict["image_path"])

        if self.same_class_ratio >= 0:
            same_class = True if np.random.rand() <= self.same_class_ratio else False
            for i in np.random.permutation(tot_cnt):
                if same_class == torch.equal(Y_dict["labels"][i], label):
                    idx = i
                    break

        # Calc all transforms before mixup
        prev_transforms = transforms[: kwargs["idx"]]

        # Apply all prev mixup transforms
        if "image" not in X_dict and "image_path" in X_dict:
            cand_img, cand_label = Compose(prev_transforms)(
                default_loader(X_dict["image_path"][idx]),
                Y_dict["labels"][idx],
                **kwargs,
            )
        else:
            cand_img, cand_label = Compose(prev_transforms)(
                X_dict["image"][idx], Y_dict["labels"][idx], **kwargs
            )

        mixup_img = ToPILImage()(
            mix_ratio * ToTensor()(pil_img) + (1 - mix_ratio) * ToTensor()(cand_img)
        )

        if label is not None:
            if self.prob_label:
                mixup_label = mix_ratio * label + (1 - mix_ratio) * cand_label
            else:
                mixup_label = label if np.random.random() < mix_ratio else cand_label
        else:
            mixup_label = label

        return mixup_img, mixup_label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"alpha={self.alpha}, same_class_ratio={self.same_class_ratio}, "
            f"prob_label={self.prob_label}>"
        )
