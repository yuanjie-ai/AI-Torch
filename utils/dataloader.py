def data_loader(*tensors, batch_size=5, shuffle=True, num_workers=0):
    """
    x = torch.linspace(1, 10, 10)  # this is x data (torch tensor)
    y = torch.linspace(10, 1, 10)  # this is y data (torch tensor)

    loder = data_loa
    der(x, y)
    epochs = 5
    for epoch in range(epochs):  # train entire dataset 3 times
        for step, (batch_x,
                   batch_y) in tqdm(enumerate(loder)):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
    """
    from torch.utils import data

    _ = data.DataLoader(
        dataset=data.TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False)
    return _
