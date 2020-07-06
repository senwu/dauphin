from dauphin.text.transforms.back_translate import BackTranslate
from dauphin.text.transforms.tfidf_word_rep import TfIdfWordRep
from dauphin.text.transforms.unif_rep import UnifRep

ALL_TRANSFORMS = {"Uni": UnifRep, "Tfidf": TfIdfWordRep, "Trans": BackTranslate}
