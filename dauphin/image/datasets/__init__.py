from dauphin.image.datasets.cifar_dataset import CIFARDataset
from dauphin.image.datasets.mnist_dataset import MNISTDataset
from dauphin.image.datasets.imagenet_dataset import ImageNetDataset

ALL_DATASETS = {
    "ImageNet": ImageNetDataset,
    "cifar10": CIFARDataset,
    "cifar100": CIFARDataset,
    "mnist": MNISTDataset,
}
