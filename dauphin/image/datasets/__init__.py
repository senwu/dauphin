from dauphin.image.datasets.cifar_dataset import CIFARDataset
from dauphin.image.datasets.mnist_dataset import MNISTDataset

ALL_DATASETS = {
    "cifar10": CIFARDataset,
    "cifar100": CIFARDataset,
    "mnist": MNISTDataset,
}
