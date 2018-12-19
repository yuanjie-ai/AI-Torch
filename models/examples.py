pos = pd.read_excel('/home/yuanjie/desktop/情感分析/pos.xls', names=['s'], header=None).assign(label=1)
neg = pd.read_excel('/home/yuanjie/desktop/情感分析/neg.xls', names=['s'], header=None).assign(label=0)
df = pd.concat([pos, neg], ignore_index=True)
df['words'] = df.s.apply(jieba.lcut)
bow = BOW(1000, opt.max_len)

X = bow.fit_transform(df.words)
y = df.label.values

_X, _y = map(torch.LongTensor, shuffle(X, y))
