from torchvision import transforms as transforms

from dauphin.image.transforms.transform import DauphinTransform


class RandomCrop(DauphinTransform):
    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
        name=None,
        prob=1.0,
        level=0,
    ):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        self.transform_func = transforms.RandomCrop(
            self.size, self.padding, self.pad_if_needed, self.fill, self.padding_mode
        )

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}, padding={self.padding}, "
            f"pad_if_needed={self.pad_if_needed}, fill={self.fill}, "
            f"mode={self.padding_mode}>"
        )
