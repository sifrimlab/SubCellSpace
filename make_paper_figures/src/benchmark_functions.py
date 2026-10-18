import numpy as np
import pandas as pd

def compute_silhouette(adata, obs_key="pattern", use_rep="X_umap"):
    from sklearn.metrics import silhouette_score
    """
    Compute silhouette score for a given obs_key in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    obs_key : str
        Column in adata.obs with categorical labels.
    use_rep : str
        Key in adata.obsm to use as representation (e.g. "X_umap", "latent").

    Returns
    -------
    score : float
        Mean silhouette score.
    """
    if obs_key not in adata.obs:
        raise ValueError(f"{obs_key} not found in adata.obs")
    if use_rep not in adata.obsm:
        raise ValueError(f"{use_rep} not found in adata.obsm")

    X = adata.obsm[use_rep]
    labels = adata.obs[obs_key].astype("category").cat.codes  # convert to integers

    # silhouette_score expects at least 2 clusters
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{obs_key} has fewer than 2 unique categories")

    return silhouette_score(X, labels)

def compute_silhouette_per_label(adata, obs_key="pattern", use_rep="X_umap"):
    from sklearn.metrics import silhouette_samples
    """
    Silhouette score for obs_key, overall and per category (Table 1).

    Unlike compute_silhouette, observations with a missing label are dropped
    instead of being scored as one extra cluster.

    Returns
    -------
    score : float
        Mean silhouette score over all labelled observations.
    per_label : dict
        Mean silhouette score of the observations in each category.
    """
    labels = adata.obs[obs_key]
    keep = labels.notna().values
    X = adata.obsm[use_rep][keep]
    labels = labels[keep].astype(str).values

    scores = silhouette_samples(X, labels, n_jobs=-1)
    per_label = pd.Series(scores).groupby(labels).mean()
    return scores.mean(), per_label.to_dict()

def kappa_probe(X, labels, groups=None, n_splits=5, n_repeats=3, max_depth=None, shuffle_labels=False, seed=101):
    from sklearn import ensemble, metrics, model_selection
    """
    Cohen's kappa of a random forest predicting labels from X, cross-validated (Table 2).

    With groups, the folds are grouped (e.g. by cell_id), so no group is in both the
    training and the test fold. With shuffle_labels, the labels are permuted first,
    which gives the chance level of the same probe.

    Returns
    -------
    kappas : list of float
        One kappa per fold, n_splits * n_repeats in total.
    """
    labels = np.asarray(labels)
    kappas = []
    for repeat in range(n_repeats):
        rs = seed + repeat
        y = np.random.RandomState(rs).permutation(labels) if shuffle_labels else labels
        if groups is None:
            splits = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs).split(X, y)
        else:
            splits = model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs).split(X, y, groups)
        for train_idx, test_idx in splits:
            clf = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=rs, n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            kappas.append(metrics.cohen_kappa_score(y[test_idx], clf.predict(X[test_idx])))
    return kappas

def benchmark_classify_patterns(adata_test, labels, group="cell_id"):
    from sklearn import ensemble, metrics, model_selection

    booster = ensemble.RandomForestClassifier(n_jobs=-1)
    splitter = model_selection.StratifiedGroupKFold()


        # Generate cross-validated predictions
    y_pred = model_selection.cross_val_predict(
        booster,
        X=adata_test.X,
        y=labels,
        groups=adata_test.obs[group],
        cv=splitter,
        n_jobs=-1
    )

    # Global metrics
    acc = metrics.accuracy_score(labels, y_pred)
    kappa = metrics.cohen_kappa_score(labels, y_pred)
    f1_weighted = metrics.f1_score(labels, y_pred, average="weighted")
    precision_weighted = metrics.precision_score(labels, y_pred, average="weighted")
    recall_weighted = metrics.recall_score(labels, y_pred, average="weighted")

    # Per-class F1
    f1_per_class = metrics.f1_score(labels, y_pred, average=None)
    classes = np.unique(labels)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Kappa:    {kappa:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}")
    print(f"Recall (weighted): {recall_weighted:.3f}")
    print(f"Precision (weighted): {precision_weighted:.3f}")
    print("F1 per class:")
    for cls, f1 in zip(classes, f1_per_class):
        print(f"  {cls}: {f1:.3f}")

    return {
        "accuracy": acc,
        "kappa": kappa,
        "f1_weighted": f1_weighted,
        "recall_weighted": recall_weighted,
        "precision_weighted": precision_weighted,
        "f1_per_class": dict(zip(classes, f1_per_class))
    }
