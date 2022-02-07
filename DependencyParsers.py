from chu_liu_edmonds import decode_mst
import torch
import torch.nn as nn
import abc


class DependencyParser(nn.Module, abc.ABC):
    """
    A class abstracting the various implementations we can create based on Kiperwasser dependency parser paper.
    """

    def __init__(self, pos_embedding_dim, pos_vocab_size, hidden_dim, mlp_hidden_dim, num_layers, device):
        """
        :param pos_embedding_dim:
        :param pos_vocab_size:
        :param hidden_dim:
        :param mlp_hidden_dim:
        :param num_layers:
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)  # Implement embedding layer for POS tags
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function = nn.CrossEntropyLoss().to(device)

    def forward(self, sentence):
        """
        :param sentence: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A score matrix and a predicted tree corresponding to the input sentence.
        """
        word_idx_tensor, pos_idx_tensor, true_sentence_len = sentence
        """
        word_idx_tensor: index of all words in the sentence (from word vocab)
        pos_idx_tensor: index of all pos in the sentence (from pos vocab)
        sentence_len: int
        """
        word_idx_tensor = word_idx_tensor[0][:true_sentence_len]
        pos_idx_tensor = pos_idx_tensor[0][:true_sentence_len]

        word_vec = self.get_word_embedding_vec(word_idx_tensor.to(self.device)).to(self.device)
        pos_vec = self.pos_embedding(pos_idx_tensor.to(self.device)).to(self.device)

        # Concat both embedding outputs
        x = torch.cat([word_vec, pos_vec], 1)
        x = x.unsqueeze(0)
        # Get Bi-LSTM hidden representation for each word+pos in sentence
        hidden, _ = self.get_encoder(x)

        a = hidden.view(hidden.shape[1], hidden.shape[2]).unsqueeze(1).repeat(1, true_sentence_len, 1)
        b = hidden.repeat(true_sentence_len, 1, 1)
        c = torch.cat([a, b], -1)

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = self.get_edge_scorer(c, true_sentence_len)
        predicted_tree = self.decoder(score_matrix.cpu().detach().numpy(), true_sentence_len, has_labels=False)[0]
        return score_matrix, predicted_tree

    @abc.abstractmethod
    def get_word_embedding_vec(self, word_idx_tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_encoder(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_edge_scorer(self, c, true_sentence_len):
        raise NotImplementedError()


class KiperwasserDependencyParserBasic(DependencyParser):
    def __init__(self, word_embedding_dim, pos_embedding_dim, word_vocab_size, pos_vocab_size,
                 hidden_dim, mlp_hidden_dim, num_layers, device):
        super().__init__(pos_embedding_dim, pos_vocab_size, hidden_dim, mlp_hidden_dim, num_layers, device)
        # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.input_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim

        # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               bidirectional=True, batch_first=True)

        # the input is 2 vectors with dim=lstm_hidden_dim concatenated
        self.edge_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.mlp_hidden_dim, 1).to(device)
        )

    def get_word_embedding_vec(self, word_idx_tensor):
        return self.word_embedding(word_idx_tensor.to(self.device)).to(self.device)

    def get_encoder(self, x):
        return self.encoder(x)

    def get_edge_scorer(self, c, true_sentence_len):
        return self.edge_scorer(c).view(true_sentence_len, true_sentence_len).to(self.device)


class KiperwasserDependencyParserAdvanced(DependencyParser):
    def __init__(self, word_embeddings, pos_embedding_dim, pos_vocab_size, hidden_dim, mlp_hidden_dim, num_layers,
                 device):
        super().__init__(pos_embedding_dim, pos_vocab_size, hidden_dim, mlp_hidden_dim, num_layers, device)
        self.word_embedding = word_embeddings
        self.input_dim = self.word_embedding[2].size(-1) + self.pos_embedding.embedding_dim

        # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               bidirectional=True, batch_first=True)

        # the input is 2 vectors with dim=lstm_hidden_dim concatenated
        self.edge_scorer = nn.Sequential(
            nn.Linear(4*self.hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, 1).to(device)
        ).to(device)

    def get_word_embedding_vec(self, word_idx_tensor):
        return self.word_embedding[2][word_idx_tensor.to(self.device)].to(self.device)

    def get_encoder(self, x):
        return self.encoder(x)

    def get_edge_scorer(self, c, true_sentence_len):
        return self.edge_scorer(c).view(true_sentence_len, true_sentence_len).to(self.device)
