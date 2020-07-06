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
import collections
import copy

import numpy as np

from dauphin.text.transforms.transform import DauphinTransform
from dauphin.text.transforms.utils import EfficientRandomGen


class TfIdfWordRep(EfficientRandomGen, DauphinTransform):
    """TF-IDF Based Word Replacement."""

    def __init__(self, name=None, prob=1.0, level=0, token_prob=0.4, data_stats={}):
        super().__init__(name, prob, level)
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []

        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = self.normalized_tf_idf.max() - self.normalized_tf_idf
        self.normalized_tf_idf = self.normalized_tf_idf / self.normalized_tf_idf.sum()
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1.0 / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (
            replace_prob / replace_prob.sum() * self.token_prob * len(all_words)
        )
        return replace_prob

    def transform(self, text, label, **kwargs):
        all_words = copy.deepcopy(text)
        replace_prob = self.get_replace_prob(all_words)
        new_text = self.replace_tokens(copy.deepcopy(text), replace_prob)
        return new_text, label

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf
        )
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"token_prob={self.token_prob}>"
        )
