def cluster_features(df, numeric_cols, method="kmeans", n_clusters=4, eps=0.9, min_samples=5):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler

    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    method = str(method).lower()

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        labels = np.zeros(len(X_scaled), dtype=int)

    return pd.Series(labels, index=df.index, name="cluster")
