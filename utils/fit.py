def fit(model, X, y, epochs=5, batch_size=128, loss_func=torch.nn.CrossEntropyLoss()):
    """
    :param model:
    :param X: torch.FloatTensor
    :param y: torch.LongTensor
    :param epochs:
    :param batch_size:
    :param loss_func:
    :param optimizer:
    :return:
    """
    from tqdm import tqdm_notebook
    from sklearn.metrics import roc_auc_score as auc
    
    optimizer = torch.optim.Adam(model.parameters())
    bar_epochs = tqdm_notebook(range(epochs))
    for e in bar_epochs:
        bar_epochs.set_description(f"Epoch {e}:")
        t = tqdm_notebook(range(0, X.size(0), batch_size))
        for b in t:  # for each training step
            # train your data...
            b_X = X[b:b + batch_size]
            b_y = y[b:b + batch_size]
            output = model(b_X)  # rnn output
            loss = loss_func(
                output,
                b_y)  # cross entropy loss and y is not one-hotted
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            if b % 10000 == 0:
                t.set_description(
                    f"Epoch {e}:"
                    f"Loss: {loss.data.numpy():.5f} | "
                    f"Auc: {auc(b_y.numpy(), output.data.numpy()[:, 1]):.5}")
