import numpy as np

from layers import DenseLayer


def createNetwork(input_size, layer_specs, seed=42):
    rng = np.random.default_rng(seed)
    network = []
    prev_size = input_size
    for spec in layer_specs:
        if not isinstance(spec, DenseLayer):
            spec = DenseLayer(**spec)
        spec.build(prev_size, rng)
        network.append(spec)
        prev_size = spec.units
    return network


def _to_onehot(y, n_classes=2):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def _forward(network, x):
    a = x
    for layer in network:
        a = layer.forward(a)
    return a


def _backward(network, y_onehot):
    proba = network[-1].a
    grad = (proba - y_onehot) / y_onehot.shape[0]
    for layer in reversed(network):
        grad = layer.backward(grad)


def _update(network, learning_rate):
    for layer in network:
        layer.update(learning_rate)


def _binary_crossentropy(proba, y_onehot):
    eps = 1e-12
    return float(-np.mean(np.sum(y_onehot * np.log(proba + eps), axis=1)))


def _accuracy(proba, y_onehot):
    pred = np.argmax(proba, axis=1)
    true = np.argmax(y_onehot, axis=1)
    return float(np.mean(pred == true))


def fit(network, data_train, data_valid, loss="binaryCrossentropy",
        learning_rate=0.0314, batch_size=8, epochs=84, seed=42, verbose=True):
    x_train, y_train = data_train
    x_val, y_val = data_valid

    y_train_oh = _to_onehot(y_train)
    y_val_oh = _to_onehot(y_val)

    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        x_sh = x_train[perm]
        y_sh = y_train_oh[perm]

        for start in range(0, n, batch_size):
            xb = x_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            _forward(network, xb)
            _backward(network, yb)
            _update(network, learning_rate)

        train_proba = _forward(network, x_train)
        val_proba = _forward(network, x_val)

        train_loss = _binary_crossentropy(train_proba, y_train_oh)
        val_loss = _binary_crossentropy(val_proba, y_val_oh)
        train_acc = _accuracy(train_proba, y_train_oh)
        val_acc = _accuracy(val_proba, y_val_oh)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - "
                  f"val_acc: {val_acc:.4f}")

    return history


def predict(network, x):
    proba = _forward(network, x)
    return np.argmax(proba, axis=1)


def predict_proba(network, x):
    return _forward(network, x)


def binary_crossentropy_loss(network, x, y):
    proba = _forward(network, x)
    return _binary_crossentropy(proba, _to_onehot(y))


def accuracy_score(network, x, y):
    proba = _forward(network, x)
    return _accuracy(proba, _to_onehot(y))


def save(network, path, norm_stats=None):
    payload = {
        "layers": [layer.get_state() for layer in network],
        "norm_stats": norm_stats,
    }
    np.save(path, payload, allow_pickle=True)


def load(path):
    payload = np.load(path, allow_pickle=True).item()
    network = []
    for state in payload["layers"]:
        layer = DenseLayer(state["units"], state["activation"],
                           state["weights_initializer"])
        layer.set_state(state)
        network.append(layer)
    return network, payload["norm_stats"]