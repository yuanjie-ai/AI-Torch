def train_desc(epoch, **kw):
    """
    import random
    from tqdm import tqdm
    epochs = 10
    batch_size = 128
    for e in range(epochs):
        bar_batchs = tqdm(range(0, 1280, batch_size))
        for b in bar_batchs:
            loss = random.random()
            if b % 10 == 0:
                auc = random.random()
                acc = random.random()
                f1 = random.random()
                desc = train_desc(e + 1, loss=loss, auc=auc, acc=acc, f1=f1)
                bar_batchs.set_description(desc)
    """
    _ = ' ｜ '.join([f'{k.title()} = {v:.5f}' for k, v in kw.items()])
    return f'Epoch {epoch+1: >2}: ' + _



