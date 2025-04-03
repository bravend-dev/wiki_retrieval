import re
import string
import sqlite3

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