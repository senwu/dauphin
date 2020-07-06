# coding=utf-8
# Copyright 2019 The Google UDA Team Authors and Sen Wu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import numpy as np

from dauphin.text.transforms.transform import DauphinTransform
from dauphin.text.transforms.utils import EfficientRandomGen


class UnifRep(EfficientRandomGen, DauphinTransform):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, name=None, prob=1.0, level=0, token_prob=0.4, vocab={}):
        super().__init__(name, prob, level)
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def transform(self, text, label, **kwargs):
        new_text = self.replace_tokens(copy.deepcopy(text))
        return new_text, label

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        for i in range(len(tokens)):
            if self.get_random_prob() < self.token_prob:
                tokens[i] = self.get_random_token()
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"token_prob={self.token_prob}>"
        )
