- 词向量嵌入
```python
self.embed = nn.Embedding(vocab_size, embedding_dim)
self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
self.embed.weight.requires_grad = False  # 冻结词向量

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=opt.weight_decay)
```
