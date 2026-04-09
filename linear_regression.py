import numpy as np
import pandas as pd

MIN_STD = 1e-12
MIN_ALPHA = 1e-12

def _normalize_features(X, mean=None, std=None):
    X = X.copy()
    mean = X.mean() if mean is None else mean
    std = X.std(ddof=0) if std is None else std
    std = std.replace(0, 1.0)
    X = (X - mean) / std
    return X, mean, std

def _add_bias(X):
    X = X.copy()
    X.insert(0, 'X0', 1.0)
    return X

def _compute_loss(X, y, w, lambda_reg):
    err = X @ w - y
    mse = np.mean(err ** 2)
    reg_penalty = lambda_reg * np.sum(w[1:] ** 2)
    return mse + reg_penalty, err

def _compute_gradient(X, err, w, lambda_reg):
    m = X.shape[0]
    grad = (2.0 / m) * (X.T @ err)
    reg_grad = 2.0 * lambda_reg * w
    reg_grad[0] = 0.0
    return grad + reg_grad

def _select_alpha(X, y, w, grad, alpha, lambda_reg):
    base_loss, _ = _compute_loss(X, y, w, lambda_reg)
    step_alpha = alpha

    while step_alpha >= MIN_ALPHA:
        candidate_w = w - step_alpha * grad
        candidate_loss, _ = _compute_loss(X, y, candidate_w, lambda_reg)
        if candidate_loss <= base_loss:
            return step_alpha
        step_alpha *= 0.5

    return alpha

#%% UCENJE
def learn(
    data,
    target_att=None,
    alpha=0.01,
    max_iter=10000,
    tolerance=1e-6,
    lambda_reg=0.0,
    stochastic=False,
    auto_alpha=False,
    random_state=0,
):
    target_att = target_att or data.columns[-1]
    X = data.drop(columns=target_att)
    y = data[[target_att]].to_numpy(dtype=float)

    X_norm, X_mean, X_std = _normalize_features(X)
    X_train = _add_bias(X_norm).to_numpy(dtype=float)

    m, n = X_train.shape
    w = np.zeros((n, 1), dtype=float)
    rng = np.random.default_rng(random_state)
    history = []
    current_alpha = alpha

    for it in range(max_iter):
        if stochastic:
            idx = rng.integers(m)
            X_step = X_train[idx:idx + 1]
            y_step = y[idx:idx + 1]
            _, step_err = _compute_loss(X_step, y_step, w, 0.0)
            grad = _compute_gradient(X_step, step_err, w, lambda_reg)
            step_alpha = current_alpha / np.sqrt(it + 1) if auto_alpha else current_alpha
            w = w - step_alpha * grad

            loss, err = _compute_loss(X_train, y, w, lambda_reg)
            grad_norm = np.abs(_compute_gradient(X_train, err, w, lambda_reg)).sum()
        else:
            loss, err = _compute_loss(X_train, y, w, lambda_reg)
            grad = _compute_gradient(X_train, err, w, lambda_reg)
            step_alpha = _select_alpha(X_train, y, w, grad, current_alpha, lambda_reg) if auto_alpha else current_alpha
            w = w - step_alpha * grad

            loss, err = _compute_loss(X_train, y, w, lambda_reg)
            grad_norm = np.abs(grad).sum()
            current_alpha = step_alpha

        history.append(
            {
                'iteration': it,
                'alpha': float(step_alpha),
                'loss': float(loss),
                'grad_norm': float(grad_norm),
            }
        )

        if grad_norm < tolerance:
            break

    return {
        '_target_att': target_att,
        '_feature_names': X.columns.tolist(),
        '_mean': X_mean,
        '_std': X_std,
        '_weights': w,
        '_lambda_reg': lambda_reg,
        '_history': pd.DataFrame(history),
        '_stochastic': stochastic,
    }

#%% PREDVIDJANJE
def predict(model, new_data):
    if isinstance(new_data, pd.Series):
        X_new = new_data.reindex(model['_feature_names']).to_frame().T
        return float(predict(model, X_new).iloc[0])

    X_new = new_data.copy()
    if model['_target_att'] in X_new.columns:
        X_new = X_new.drop(columns=model['_target_att'])

    X_new = X_new[model['_feature_names']]
    X_new, _, _ = _normalize_features(X_new, model['_mean'], model['_std'])
    X_new = _add_bias(X_new).to_numpy(dtype=float)

    predictions = (X_new @ model['_weights']).ravel()
    return pd.Series(predictions, index=new_data.index, name='prediction')

#%% TEST
if __name__ == '__main__':
    data = pd.read_csv('data/boston.csv')

    model = learn(
        data,
        alpha=0.05,
        max_iter=5000,
        tolerance=1e-4,
        lambda_reg=0.01,
        stochastic=False,
        auto_alpha=True,
    )

    y_name = data.columns[-1]
    preview = data[[y_name]].head().copy()
    preview['prediction'] = predict(model, data.drop(columns=y_name).head()).values

    print(preview)
    print(model['_history'].tail(1))