from Main import KiperwasserDependencyParserAdvanced, KiperwasserDependencyParserBasic
from DataLoader import ParseTreeDataset
import torch
import pickle
from torch.utils.data.dataloader import DataLoader
import sys
import time

WORD = 1
POS = 3
HEAD = 6


def append_sentence2csv(path, orig_sentence, predicted_tree, last_row_flag):
    """
    Write row to csv in the .labeled format
    """
    orig_sentence_words = orig_sentence.split('\n')
    orig_sentence_words = [orig_sentence_words[i].split('\t') for i in range(len(orig_sentence_words))]
    with open(path, 'a') as f:
        size = len(orig_sentence_words)
        for i in range(size):
            word = orig_sentence_words[i][WORD]
            pos = orig_sentence_words[i][POS]
            head = predicted_tree[i]
            row = f'{i+1}\t{word}\t_\t{pos}\t_\t_\t{head}\t_\t_\t_\n'
            f.write(row)
        if not last_row_flag:
            f.write('\n')


if __name__ == '__main__':
    version = sys.argv[1]  # either 'basic' or 'advanced'
    if version == 'basic':
        basic_ver = True
    elif version == 'advanced':
        basic_ver = False
    else:
        KeyError("argument can be either 'basic' or 'advanced'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load pre-trained
    with open(f'results\\vocabs_{version}.pkl', 'rb') as f:
        vocabs = pickle.load(f)
    model = torch.load(f'results\\nn_model_{version}')

    # load comp data
    comp_path = 'comp'
    dir_path = 'data\\'

    if basic_ver:
        word_embeddings = None
        version_str = 'm1'
    else:
        word_embeddings = model.word_embedding
        version_str = 'm2'
    t0 = time.time()
    comp = ParseTreeDataset(word_dict=vocabs[0], pos_dict=vocabs[1], dir_path=dir_path, basic_ver=basic_ver,
                            subset=comp_path, padding=True, word_embeddings=word_embeddings)
    comp_dataloader = DataLoader(comp, shuffle=False)

    path = f'results\\comp_{version_str}.labeled'
    with open(path, 'w'):
        pass

    with open(dir_path + comp_path + '.unlabeled', 'r') as f:
        data = f.read()
    raw_sentences = data.split('\n\n')
    if raw_sentences[-1] == '':
        raw_sentences = raw_sentences[:-1]

    comp_len = len(comp)
    for i, (input_data, orig_sentence) in enumerate(zip(comp_dataloader, raw_sentences)):
        words_idx_tensor, pos_idx_tensor, __, sentence_length = input_data
        words_idx_tensor.to(device)
        pos_idx_tensor.to(device)
        sentence_length.to(device)
        score_matrix, predicted_tree = model((words_idx_tensor, pos_idx_tensor, sentence_length))
        append_sentence2csv(path, orig_sentence, predicted_tree[1:], i == comp_len - 1)

    print(f'Finished labeling the comp file. results are in {path}')
    print(f'labeling comp file took {time.time()-t0:.4f} sec')


