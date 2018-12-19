def predict(model, X, batch_size=128):
    from tqdm import trange
    
    _model = model.eval()
    def func(x):
        with torch.no_grad():
            return _model(x)
        
    _ = [X[b:b + batch_size] for b in trange(0, len(X), batch_size, desc='Predict')]
    with ThreadPoolExecutor(8) as pool:
        _ = np.row_stack(pool.map(func, _))
        del _model
        return _[:, 1]
