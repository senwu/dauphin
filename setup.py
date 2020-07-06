from setuptools import find_packages, setup

setup(
    name="Dauphin",
    version="0.0.1",
    description=(
        "On the Generalization Effects of Linear Transformations in Data Augmentation."
    ),
    install_requires=[
        "emmental>=0.0.7,<0.1.0",
        "torch==1.4.0",
        "torchvision==0.5.0",
        "spacy==2.3.0",
        "torchtext==0.6.0",
        "pytorch-pretrained-bert==0.6.2",
    ],
    scripts=["bin/image", "bin/text"],
    packages=find_packages(),
)
