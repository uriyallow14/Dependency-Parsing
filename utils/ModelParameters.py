from torchtext.vocab import Vocab
from collections import Counter
from DataLoader import SPECIAL_TOKENS_ADV


def model_parameters(basic_ver, word_dict):
    if basic_ver:
        # basic version
        word_embedding_dim = 100
        word_embedding = None
        pos_embedding_dim = 25
        hidden_dim = 125
        mlp_hidden_dim = 100
        num_layers = 2
        alpha = 0.25
        lr = 0.01
        weight_decay = 0
        EPOCHS = 30

    else:
        # advanced version
        word_embedding_dim = 100
        embedding_name = 'glove.6B.100d'
        glove = Vocab(Counter(word_dict), vectors=embedding_name, specials=SPECIAL_TOKENS_ADV)
        word_embedding = (glove.stoi, glove.itos, glove.vectors)
        pos_embedding_dim = 50
        hidden_dim = 125
        mlp_hidden_dim = 100
        num_layers = 3
        alpha = 0.25
        lr = 1e-2
        weight_decay = 1e-5
        EPOCHS = 50

    return word_embedding_dim, word_embedding, pos_embedding_dim, hidden_dim, mlp_hidden_dim, num_layers, alpha, lr,\
           weight_decay, EPOCHS
