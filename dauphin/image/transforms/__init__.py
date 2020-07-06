from dauphin.image.transforms.auto_contrast import AutoContrast
from dauphin.image.transforms.blur import Blur
from dauphin.image.transforms.brightness import Brightness
from dauphin.image.transforms.center_crop import CenterCrop
from dauphin.image.transforms.color import Color
from dauphin.image.transforms.contrast import Contrast
from dauphin.image.transforms.cutout import Cutout
from dauphin.image.transforms.equalize import Equalize
from dauphin.image.transforms.horizontal_filp import HorizontalFlip
from dauphin.image.transforms.identity import Identity
from dauphin.image.transforms.invert import Invert
from dauphin.image.transforms.mixup import Mixup
from dauphin.image.transforms.posterize import Posterize
from dauphin.image.transforms.random_crop import RandomCrop
from dauphin.image.transforms.random_resize_crop import RandomResizedCrop
from dauphin.image.transforms.resize import Resize
from dauphin.image.transforms.rotate import Rotate
from dauphin.image.transforms.sharpness import Sharpness
from dauphin.image.transforms.shear_x import ShearX
from dauphin.image.transforms.shear_y import ShearY
from dauphin.image.transforms.smooth import Smooth
from dauphin.image.transforms.solarize import Solarize
from dauphin.image.transforms.translate_x import TranslateX
from dauphin.image.transforms.translate_y import TranslateY
from dauphin.image.transforms.vertical_flip import VerticalFlip

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Blur": Blur,
    "Brightness": Brightness,
    "CenterCrop": CenterCrop,
    "Color": Color,
    "Contrast": Contrast,
    "Cutout": Cutout,
    "Equalize": Equalize,
    "HorizontalFlip": HorizontalFlip,
    "Identity": Identity,
    "Invert": Invert,
    "Mixup": Mixup,
    "Posterize": Posterize,
    "RandomCrop": RandomCrop,
    "RandomResizedCrop": RandomResizedCrop,
    "Resize": Resize,
    "Rotate": Rotate,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Smooth": Smooth,
    "Solarize": Solarize,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "VerticalFlip": VerticalFlip,
}
