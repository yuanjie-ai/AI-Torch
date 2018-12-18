import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score as auc

def train_desc(epoch, **kw):
    _ = ' ｜ '.join([f'{k.title()} = {v:.5f}' for k, v in kw.items()])
    return f'Epoch {epoch+1: >2}: ' + _

def fit(model, X, y, epochs=5, batch_size=128, loss_func=torch.nn.CrossEntropyLoss()):
    assert isinstance(y, torch.LongTensor)

    optimizer = torch.optim.Adam(model.parameters())

    for e in range(epochs):
        # X, y = map(torch.LongTensor, shuffle(X.numpy(), y.numpy()))
        batchs = tqdm(range(0, X.size(0), batch_size))
        for b in batchs:  # for each training step
            # train your data...
            b_X = X[b:b + batch_size]
            b_y = y[b:b + batch_size]
            _y = model(b_X)  # rnn output
            loss = loss_func(_y, b_y)  # cross entropy loss and y is not one-hotted
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
            if b % 50 == 0:
                _ = train_desc(e, 
                               loss=loss, 
                               auc=auc(b_y.numpy(), _y[:, 1].data.numpy()),
                               acc=(b_y==torch.max(_y, 1)[1]).sum().numpy()/len(_y))
                batchs.set_description(_)
