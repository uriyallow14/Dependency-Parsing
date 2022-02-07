import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import time
import matplotlib.pyplot as plt
import pickle
import sys
import warnings
from DataLoader import get_vocabs, ParseTreeDataset
from DependencyParsers import KiperwasserDependencyParserBasic, KiperwasserDependencyParserAdvanced
from utils.PlotMeasurments import plot_measurement_graph
from utils.Validation import evaluate
from utils.ModelParameters import model_parameters
from utils.ReadData import get_data
from utils.Trainer import training

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
if use_cuda:
    torch.cuda.empty_cache()
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    version = sys.argv[1]  # either 'basic' or 'advanced'
    paths_list, basic_ver, data_dir = get_data(version)
    word_dict, pos_dict = get_vocabs(paths_list)

    # hyper-parameters
    word_embedding_dim, word_embedding, pos_embedding_dim, hidden_dim, mlp_hidden_dim, num_layers, alpha, lr, \
    weight_decay, EPOCHS = model_parameters(basic_ver, word_dict)

    train = ParseTreeDataset(word_dict, pos_dict, data_dir, 'train', basic_ver=basic_ver,
                             alpha=alpha, padding=True, word_embeddings=word_embedding)
    train_dataloader = DataLoader(train, shuffle=True)
    test = ParseTreeDataset(word_dict, pos_dict, data_dir, 'test', basic_ver=basic_ver,
                            padding=True, word_embeddings=word_embedding)
    test_dataloader = DataLoader(test, shuffle=False)

    # save train object for comp py script
    vocabs = (word_dict, pos_dict)
    with open(f'results\\vocabs_{version}.pkl', 'wb') as f:
        pickle.dump(vocabs, f)

    word_vocab_size = len(train.word_idx_mappings)
    pos_vocab_size = len(train.pos_idx_mappings)

    if basic_ver:
        model = KiperwasserDependencyParserBasic(word_embedding_dim, pos_embedding_dim, word_vocab_size, pos_vocab_size,
                                                 hidden_dim, mlp_hidden_dim, num_layers, device).to(device)
    else:
        model = KiperwasserDependencyParserAdvanced(word_embedding, pos_embedding_dim, pos_vocab_size,
                                                    hidden_dim, mlp_hidden_dim, num_layers, device).to(device)

    print(f'number of parameters = {sum(param.numel() for param in model.parameters())}')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    acumulate_grad_steps = 50
    print("Training Started")

    t0, train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = training(model, EPOCHS,
                                                                                            train_dataloader, train,
                                                                                            test_dataloader, test,
                                                                                            device, acumulate_grad_steps
                                                                                            , optimizer)

    print('Training is done')
    print(f'time took for training phase is {(time.time() - t0)/60:.2f} minutes\n')
    # compute final results:
    print('Final results:')
    t2 = time.time()
    train_final_loss, train_final_acc = evaluate(train_dataloader, len(train), model, device)
    print(f'time took to evaluate train: {time.time() - t2:.4f} sec')
    t2 = time.time()
    test_final_loss, test_final_acc = evaluate(test_dataloader, len(test), model, device)
    print(f'time took to evaluate test: {time.time() - t2:.4f} sec')
    print(f'train_loss= {train_final_loss:.4f}\ttrain_UAS= {train_final_acc:.4f}\n'
          f'test_loss=  {test_final_loss:.4f}\ttest_UAS=  {test_final_acc:.4f}')

    plot_measurement_graph([train_accuracy_list, test_accuracy_list], ['train', 'test'], 'UAS', version)
    plot_measurement_graph([train_loss_list, test_loss_list], ['train', 'test'], 'Loss', version)

    torch.save(model, f'results\\nn_model_{version}')
