from torch.utils.data.dataset import Dataset, TensorDataset
from collections import defaultdict, OrderedDict
import torch
import random

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
NUMBER_TOKEN = "<number>"
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN, ROOT_TOKEN]
SPECIAL_TOKENS_ADV = SPECIAL_TOKENS + [NUMBER_TOKEN]


def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and pos2idx.
        :param list_of_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - pos2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == '\n':
                    continue
                items = line.split('\t')
                pos = items[3]
                word = items[1]
                word_dict[word] += 1
                pos_dict[pos] += 1

    return OrderedDict(word_dict), OrderedDict(pos_dict)


class ParseTreeDataReader:
    def __init__(self, file, word_dict, pos_dict, subset, basic_ver):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.rel_cols = [0, 1, 3, 6]
        self.subset = subset
        self.sentences = []
        self.basic_ver = basic_ver
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            data = f.read()
        raw_sentences = data.split('\n\n')
        if raw_sentences[-1] == '':
            raw_sentences = raw_sentences[:-1]
        self.sentences = []
        for raw_sentence in raw_sentences:
            init_tup = (0, ROOT_TOKEN, UNKNOWN_TOKEN, -1)
            sentence = [init_tup]
            for word_line in raw_sentence.split('\n'):
                if word_line == '':
                    break
                rel_word = []
                for i, item in enumerate(word_line.split('\t')):
                    if i in self.rel_cols:
                        if i in [0, 6]:  # idx or head_idx
                            if self.subset in ['comp'] and i == 6:
                                rel_word.append(-1)  # i=6 is not relevant in competition
                            else:
                                rel_word.append(int(item))
                        else:  # a string of word or POS
                            if not self.basic_ver and i == 1:
                                rel_word.append(self.transformed_word(item))
                            else:
                                rel_word.append(item)
                sentence.append(rel_word)
            self.sentences.append(sentence)

    @staticmethod
    def transformed_word(word):
        for c in word:
            if c.isdigit():
                return NUMBER_TOKEN
        return word.lower()

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class ParseTreeDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str, basic_ver,
                 alpha=None, padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test, comp]
        file_ext = ".labeled" if subset in ['train', 'test'] else ".unlabeled"
        self.file = dir_path + subset + file_ext
        self.SPECIAL_TOKENS = SPECIAL_TOKENS if basic_ver else SPECIAL_TOKENS_ADV
        self.datareader = ParseTreeDataReader(self.file, word_dict, pos_dict, subset, basic_ver)
        self.vocab_size = len(self.datareader.word_dict)
        self.alpha = alpha
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads

    def word2unknown(self, token):
        if self.alpha is None:
            return False
        if token in self.SPECIAL_TOKENS:
            return False
        p = self.alpha / (self.alpha + self.datareader.word_dict[token])
        rand = random.uniform(0, 1)
        return rand < p

    def init_word_embeddings(self, word_dict):
        idx_word_mappings = self.SPECIAL_TOKENS + list(word_dict.keys())
        word_idx_mapping = {w: i for i, w in enumerate(idx_word_mappings)}
        return word_idx_mapping, idx_word_mappings, None

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = self.SPECIAL_TOKENS + list(pos_dict.keys())
        pos_idx_mapping = {w: i for i, w in enumerate(idx_pos_mappings)}
        return pos_idx_mapping, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        sentence_heads_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            heads_list = []
            for idx, word, pos, head in sentence:
                # word
                word_idx = self.word_idx_mappings.get(word)
                if word_idx is None:
                    words_idx_list.append(self.word_idx_mappings[UNKNOWN_TOKEN])
                else:
                    if self.word2unknown(word):
                        words_idx_list.append(self.word_idx_mappings[UNKNOWN_TOKEN])
                    else:
                        words_idx_list.append(word_idx)
                # pos
                pos_idx = self.pos_idx_mappings.get(pos)
                if pos_idx is None:
                    pos_idx_list.append(self.pos_idx_mappings[UNKNOWN_TOKEN])
                else:
                    pos_idx_list.append(pos_idx)
                heads_list.append(head)
            sentence_len = len(words_idx_list)

            if padding:
                while len(words_idx_list) < self.max_seq_len:
                    words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
                    pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
                    heads_list.append(-1)
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)
            sentence_heads_list.append(torch.tensor(heads_list, dtype=torch.long, requires_grad=False))

        if padding:
            all_sentence_word_idx = torch.stack(sentence_word_idx_list)
            all_sentence_pos_idx = torch.stack(sentence_pos_idx_list)
            all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
            all_sentence_heads = torch.stack(sentence_heads_list)
            return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_heads, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_heads_list,
                                                                     sentence_len_list))}
