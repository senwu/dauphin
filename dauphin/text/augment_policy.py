import logging
import random

from dauphin.text.transforms import ALL_TRANSFORMS
from dauphin.text.transforms.back_translate import BackTranslate
from dauphin.text.transforms.tfidf_word_rep import TfIdfWordRep
from dauphin.text.transforms.unif_rep import UnifRep

logger = logging.getLogger(__name__)

PRECISION = 3


def parse_sequence(x, type="int"):
    x = x.split(",")
    if len(x) == 1:
        return int(x[0]) if type == "int" else float(x[0])
    return tuple([int(_) if type == "int" else float(_) for _ in x])


def parse_transform(policy, TFs):
    parsed_transforms = []

    if policy is None:
        return parsed_transforms

    transforms = policy.split("@")
    for transform in transforms:
        name = transform.split("_")[0]
        settings = transform.split("_")[1:] if len(transform.split("_")) > 1 else []
        if name in ALL_TRANSFORMS:
            prob = random.random()
            level = random.randint(0, 10 ** PRECISION) / float(10 ** PRECISION)
            config = {"prob": prob, "level": level}
            for setting in settings:
                if setting.startswith("TP"):
                    config["token_prob"] = float(setting[2:])
                if setting.startswith("SP"):
                    config["select_prob"] = float(setting[2:])

            TFs[name].prob = config["prob"]
            TFs[name].level = config["level"]
            if "token_prob" in config:
                TFs[name].token_prob = config["token_prob"]
            if "select_prob" in config:
                TFs[name].select_prob = config["select_prob"]

            parsed_transforms.append(TFs[name])
        else:
            raise ValueError(f"Unrecognized transformation {transforms}")

    return parsed_transforms


class Augmentation(object):
    """Given the augment policy to generate the list of augmentation functions."""

    def __init__(self, args, vocab, data_stats):
        self.args = args
        self.vocab = vocab
        self.data_stats = data_stats
        self.TF_uni = UnifRep(vocab=vocab)
        self.TF_tfidf = TfIdfWordRep(data_stats=data_stats)
        self.TF_trans = BackTranslate()

        self.TFs = {"Tfidf": self.TF_tfidf, "Uni": self.TF_uni, "Trans": self.TF_trans}

    def __call__(self):
        if self.args.augment_policy == "uncertainty_sampling":
            all_transforms = ["Tfidf", "Uni", "Trans", "Trans_SP1.0"]

            transform_keys = "@".join(
                random.choices(all_transforms, k=self.args.num_comp)
            )
            transforms = parse_transform(transform_keys, self.TFs)
        else:
            transforms = parse_transform(self.args.augment_policy, self.TFs)

        return transforms
