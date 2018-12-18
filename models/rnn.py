import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """https://www.itcodemonkey.com/article/9008.html"""
    # Data
    class_num = 2

    # Embedding
    pretrained_embedding = None  # pretrained_weight
    vocab_dim = 5 if pretrained_embedding is None else pretrained_embedding.shape[1]
    vocab_size = 100

    # RNN
    rnn_layers_num = 2
    rnn_dropout = 0

    # Linear
    fc_dropout = 0


opt = Config()


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        # self.embed = nn.Embedding(V, D, max_norm=config.max_norm)
        self.embed = nn.Embedding(opt.vocab_size, opt.vocab_dim)
        # pretrained  embedding
        if opt.pretrained_embedding:
            self.embed.weight.data.copy_(opt.pretrained_embedding)
            # self.embed.weight.requires_grad = False  # 冻结词向量

        self.bilstm = nn.LSTM(
            opt.vocab_dim,
            opt.vocab_dim // 2,
            dropout=opt.rnn_dropout,
            num_layers=opt.rnn_layers_num,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True)

        self.fc1 = nn.Linear(opt.vocab_dim // 2 * 2, opt.vocab_dim // 2)
        self.fc2 = nn.Linear(opt.vocab_dim // 2, opt.class_num)
        self.dropout = nn.Dropout(opt.fc_dropout)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.bilstm(x)
        r_out = F.relu(r_out)

        # r_out = F.max_pool1d(r_out, r_out.size(2)).squeeze(2)
        # y = self.fc1(r_out)
        # y = self.fc2(y)
        # choose r_out at the last time step
        y = self.fc1(r_out[:, -1, :])
        y = self.fc2(y)
        return y
 


"""http://www.cnblogs.com/wangduo/p/6773601.html?utm_source=itdadao&utm_medium=referral
遗忘层门： 
作用对象：细胞状态 
作用：将细胞状态中的信息选择性的遗忘 
让我们回到语言模型的例子中来基于已经看到的预测下一个词。在这个问题中，细胞状态可能包含当前主语的类别，因此正确的代词可以被选择出来。当我们看到新的主语，我们希望忘记旧的主语。 
例如，他今天有事，所以我。。。当处理到‘’我‘’的时候选择性的忘记前面的’他’，或者说减小这个词对后面词的作用。

输入层门： 
作用对象：细胞状态 
作用：将新的信息选择性的记录到细胞状态中 
在我们语言模型的例子中，我们希望增加新的主语的类别到细胞状态中，来替代旧的需要忘记的主语。 
例如：他今天有事，所以我。。。。当处理到‘’我‘’这个词的时候，就会把主语我更新到细胞中去。

输出层门： 
作用对象：隐层ht 
在语言模型的例子中，因为他就看到了一个 代词，可能需要输出与一个 动词 相关的信息。例如，可能输出是否代词是单数还是负数，这样如果是动词的话，我们也知道动词需要进行的词形变化。 
例如：上面的例子，当处理到‘’我‘’这个词的时候，可以预测下一个词，是动词的可能性较大，而且是第一人称。 
会把前面的信息保存到隐层中去。

Gated Recurrent Unit (GRU)就是lstm的一个变态，这是由 Cho, et al. (2014) 提出。它将忘记门和输入门合成了一个单一的 更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。
"""
