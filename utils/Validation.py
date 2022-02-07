import torch


def evaluate(dataset_dataloader, num_sentences, model, device):
    correct_edges = 0
    total_edges = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for _, input_data in enumerate(dataset_dataloader):
            # get input
            words_idx_tensor, pos_idx_tensor, true_tree_heads, sentence_length = input_data
            words_idx_tensor.to(device)
            pos_idx_tensor.to(device)
            true_tree_heads.to(device)
            sentence_length.to(device)
            true_tree_heads = true_tree_heads[0][:sentence_length]
            score_matrix, predicted_tree = model((words_idx_tensor, pos_idx_tensor, sentence_length))
            # calculate loss
            loss = model.loss_function(score_matrix.T[1:].to(device), true_tree_heads[1:].to(device))

            total_loss += loss.item()
            # predict and evaluate
            predicted_tree = predicted_tree[1:]
            true_tree_heads = true_tree_heads.squeeze(0)
            true_tree_heads = true_tree_heads[1: sentence_length]
            for j in range(len(true_tree_heads)):
                if true_tree_heads[j] == predicted_tree[j]:
                    correct_edges += 1
                total_edges += 1
    # aggregation
    total_loss = total_loss / num_sentences
    acc = correct_edges / total_edges

    return total_loss, acc
