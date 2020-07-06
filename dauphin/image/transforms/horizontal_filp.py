from PIL import Image

from dauphin.image.transforms.transform import DauphinTransform


class HorizontalFlip(DauphinTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT), label
