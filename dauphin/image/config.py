TASK_NORMALIZE = {
    "cifar10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std": (0.24703223, 0.24348513, 0.26158784),
    },
    "cifar100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std": (0.26733429, 0.25643846, 0.27615047),
    },
    "mnist": {"mean": [0.1307], "std": [0.3081]},
}

TASK_NUM_CLASS = {"cifar10": 10, "cifar100": 100, "mnist": 10}

TASK_INPUT_SIZE = {
    "cifar10": (1, 3, 32, 32),
    "cifar100": (1, 3, 32, 32),
    "mnist": (1, 28, 28),
}

TASK_METRIC = {"cifar10": ["accuracy"], "cifar100": ["accuracy"], "mnist": ["accuracy"]}

TASK_CUTOUT_SIZE = {"cifar10": 16, "cifar100": 16, "mnist": 14}
