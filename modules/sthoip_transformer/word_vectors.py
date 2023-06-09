"""
Adapted from PyTorch's text library.
"""
import torch
import torchtext.vocab


def glove_embedding_vectors(names, wv_type="6B", wv_dir="weights/semantic/", wv_dim=200):
    glove_embeddings = torchtext.vocab.GloVe(name=wv_type, dim=wv_dim, cache=wv_dir, unk_init=torch.Tensor.normal_)
    vectors = torch.Tensor(len(names), wv_dim)

    for i, token in enumerate(names):
        vectors[i] = glove_embeddings.get_vecs_by_tokens(token.split("/")[0])

    return vectors
