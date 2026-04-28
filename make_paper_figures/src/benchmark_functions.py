import numpy as np

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
