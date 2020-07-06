import torch
from pytorch_pretrained_bert import BertTokenizer

from dauphin.text.transforms.transform import DauphinTransform


class ToTensor(DauphinTransform):
    def __init__(
        self, name=None, prob=1.0, level=0, model="bert-base-uncased", max_len=512
    ):
        super().__init__(name, prob, level)
        do_lower_case = "uncased" in model
        self.tokenizer = BertTokenizer.from_pretrained(
            model, do_lower_case=do_lower_case
        )
        self.max_len = max_len

    def transform(self, text, label, **kwargs):
        tokenized_sentence = self.tokenizer.tokenize(" ".join(text))
        if len(tokenized_sentence) > self.max_len - 2:
            # Account for [CLS] and [SEP] with "- 2"
            tokenized_sentence = tokenized_sentence[-(self.max_len - 2) :]
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        token_segments = [0] * len(tokenized_sentence)
        token_masks = [1] * len(token_ids)

        # Zero-pad up to the sequence length.
        while len(token_ids) < self.max_len:
            token_ids.append(0)
            token_segments.append(0)
            token_masks.append(0)

        text_dict = {
            "text_tokens": tokenized_sentence,
            "token_ids": torch.LongTensor(token_ids),
            "token_masks": torch.LongTensor(token_masks),
            "token_segments": torch.LongTensor(token_segments),
        }

        return text_dict, label
