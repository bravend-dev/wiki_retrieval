import re
import string
import sqlite3
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(last_hidden_state, attention_mask):
    s = torch.sum(
        last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
    )
    d = attention_mask.sum(dim=1, keepdim=True).float()
    return s / d

def cls_pooling(last_hidden_state):
    return last_hidden_state[:, 0]

_WORD_SPLIT = re.compile("([.,!?\"/':;)(])")

def connect_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""

    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    tokens = [w.lower() for w in words if w != '' and w != ' ' and w not in string.punctuation]
    return " ".join(tokens)