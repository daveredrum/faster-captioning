import os
import json
import pickle

import numpy as np

from itertools import chain
from collections import Counter

DATAROOT = "/cluster/balrog/dchen/ScanRefer/data"
DATASET = "ScanRefer"
VOCAB = os.path.join(DATAROOT, "{}_vocabulary.json".format(DATASET)) # dataset_name
GLOVE_PICKLE = os.path.join(DATAROOT, "glove.p")
GLOVE_PATH = os.path.join(DATAROOT, "glove_trimmed_{}.npy".format(DATASET)) # dataset_name

scanrefer_train = json.load(open(os.path.join(DATAROOT, "ScanRefer_filtered_train.json")))

print("building vocabulary...")
glove = pickle.load(open(GLOVE_PICKLE, "rb"))
all_words = chain(*[data["token"] for data in scanrefer_train])
word_counter = Counter(all_words)
word_counter = sorted([(k, v) for k, v in word_counter.items() if k in glove], key=lambda x: x[1], reverse=True)
word_list = [k for k, _ in word_counter]

# build vocabulary
word2idx, idx2word = {}, {}
spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
for i, w in enumerate(word_list):
    shifted_i = i + len(spw)
    word2idx[w] = shifted_i
    idx2word[shifted_i] = w

# add special words into vocabulary
for i, w in enumerate(spw):
    word2idx[w] = i
    idx2word[i] = w

speical_tokens = {
    "bos_token": "sos",
    "eos_token": "eos",
    "unk_token": "unk",
    "pad_token": "pad_"
}
vocabulary = {
    "word2idx": word2idx,
    "idx2word": idx2word,
    "special_tokens": speical_tokens
}
json.dump(vocabulary, open(VOCAB, "w"), indent=4)

print("built vocabulary with {} words".format(len(vocabulary["word2idx"])))

all_glove = pickle.load(open(GLOVE_PICKLE, "rb"))

embeddings = np.zeros((len(vocabulary["word2idx"]), 300))
for word, idx in vocabulary["word2idx"].items():
    try:
        emb = all_glove[word]
    except KeyError:
        emb = all_glove["unk"]
        
    embeddings[int(idx)] = emb

np.save(GLOVE_PATH, embeddings)

print("trimmed GLoVE embedding with {} words".format(embeddings.shape[0]))
