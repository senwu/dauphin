import collections
import copy
import math


def get_data_stats(examples):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(examples)):
        cur_word_dict = {}
        text = examples[i].text
        text = clean_web_text(" ".join(text)).split(" ")
        cur_sent = copy.deepcopy(text)
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1.0 / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_word_dict = {}
        text = examples[i].text
        text = clean_web_text(" ".join(text)).split(" ")
        cur_sent = copy.deepcopy(text)
        # cur_sent = copy.deepcopy(examples[i].text)
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1.0 / len(cur_sent) * idf[word]
    return {"idf": idf, "tf_idf": tf_idf}


def build_vocab(examples):
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(examples)):
        text = examples[i].text
        text = clean_web_text(" ".join(text)).split(" ")
        add_to_vocab(text)
        # add_to_vocab(examples[i].text)
    return vocab


def clean_web_text(st):
    """Clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", '"')
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1 :]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
    #     st = st.replace("\\n", " ")
    #     st = st.replace("\\", " ")
    while "  " in st:
        st = st.replace("  ", " ")
    return st
