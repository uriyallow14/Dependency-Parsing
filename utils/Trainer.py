import time
from utils.Validation import evaluate


def training(model, epochs, train_dataloader, train, test_dataloader, test, device, acumulate_grad_steps, optimizer):
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    # epochs = EPOCHS
    t0 = time.time()  # start of training
    for epoch in range(epochs):
        t1 = time.time()  # start of epoch
        correct_edges = 0
        total_edges = 0
        # train_acc = 0  # to keep track of accuracy
        train_loss = 0  # To keep track of the loss value
        i = 0

        model.train()
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            # get input
            words_idx_tensor, pos_idx_tensor, true_tree_heads, sentence_length = input_data
            words_idx_tensor.to(device)
            pos_idx_tensor.to(device)
            true_tree_heads.to(device)
            sentence_length.to(device)

            true_tree_heads = true_tree_heads[0][:sentence_length]
            score_matrix, predicted_tree = model((words_idx_tensor, pos_idx_tensor, sentence_length))
            # calculate loss
            loss = model.loss_function(score_matrix.T[1:].to(device), true_tree_heads[1:].to(device)).to(device)
            loss = loss / acumulate_grad_steps
            loss.backward()

            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            train_loss += loss.item()

            # predict and evaluate
            predicted_tree = predicted_tree[1:]
            true_tree_heads = true_tree_heads.squeeze(0)
            true_tree_heads = true_tree_heads[1: sentence_length]
            for j in range(len(true_tree_heads)):
                if true_tree_heads[j] == predicted_tree[j]:
                    correct_edges += 1
                total_edges += 1

        train_loss = acumulate_grad_steps * (train_loss / len(train))
        train_acc = correct_edges / total_edges
        train_loss_list.append(float(train_loss))
        train_accuracy_list.append(float(train_acc))
        test_loss, test_acc = evaluate(test_dataloader, len(test), model, device)
        test_loss_list.append(float(test_loss))
        test_accuracy_list.append(float(test_acc))

        print(f'EPOCH {epoch + 1}:\t({time.time() - t1:.4f} sec)\t'
              f'train_loss= {train_loss:.4f}\ttrain_UAS= {train_acc:.4f}\t'
              f'test_loss= {test_loss:.4f}\ttest_UAS= {test_acc:.4f}')

    return t0, train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list
