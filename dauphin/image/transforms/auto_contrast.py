from PIL import ImageOps

from dauphin.image.transforms.transform import DauphinTransform


class AutoContrast(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return ImageOps.autocontrast(pil_img), label
