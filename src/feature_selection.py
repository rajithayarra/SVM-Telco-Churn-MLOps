from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

def select_features(X_train, y_train, alpha=0.001, correlation_threshold=0.95):
    # Step 1: remove one feature from each highly correlated pair.
    corr = X_train.corr().abs()
    upper = corr.where(
        __import__("numpy").triu(__import__("numpy").ones(corr.shape), k=1).astype(bool)
    )
    correlated_to_drop = [
        column for column in upper.columns
        if (upper[column] > correlation_threshold).any()
    ]
    reduced = X_train.drop(columns=correlated_to_drop, errors="ignore")

    # Step 2: LASSO-based selection on the remaining training features.
    selector = SelectFromModel(
        Lasso(alpha=alpha, max_iter=10000),
        threshold="median"
    )
    selector.fit(reduced, y_train)
    selected = reduced.columns[selector.get_support()].tolist()

    # Keep a usable feature set if the selected coefficients are all zero.
    return selected if selected else reduced.columns.tolist()
