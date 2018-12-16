import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook
from ml_metrics import auc
from sklearn.datasets import make_classification


class LogsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.logstic = nn.Linear(in_dim, n_class)
 
    def forward(self, x):
        out = self.logstic(x)
        return F.softmax(out, -1)

epochs = 5
batch_size = 128
X, y = make_classification(1000000)
t_X, t_y = map(torch.FloatTensor, (X, y))

net = LogsticRegression(20, 2)
loss_func = torch.nn.modules.loss.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

bar_epochs = tqdm_notebook(range(epochs))
for e in bar_epochs:
    bar_epochs.set_description(f"Epoch {e}:")
    t = tqdm_notebook(range(0, t_X.size(0), batch_size))
    for b in t:  # for each training step
        # train your data...
        b_X = t_X[b:b + batch_size]
        b_y = t_y[b:b + batch_size]
        output = net(b_X)  # rnn output
        loss = loss_func(output, b_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()
        if b % 10000 == 0:
            t.set_description(
                f"Epoch {e}:"
                f"Loss: {loss.data.numpy():.5f} | "
                f"Auc: {auc(b_y.numpy(), output.data.numpy()[:, 1]):.5}"
            )
