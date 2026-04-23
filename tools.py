from langchain_core.tools import tool
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[t] for t in iris.target]

_knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
_knn.fit(iris.data)

_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
_dt.fit(iris.data, iris.target)


@tool
def get_flower_stats(species: str) -> str:
    species = species.lower().strip()
    if species not in iris.target_names:
        return f"Unknown species '{species}'. Available: {list(iris.target_names)}"

    subset = df[df["species"] == species]
    stats = subset.drop(columns="species").agg(["min", "mean", "max"]).round(2)
    return f"Statistics for {species}:\n{stats.to_string()}"

@tool
def find_nearest_neighbors(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    distances, indices = _knn.kneighbors(sample)

    result = "Nearest samples from the dataset:\n"
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        row = df.iloc[idx]
        result += (
            f"  #{rank} | {row['species']} | "
            f"sepal={row['sepal length (cm)']:.1f}/{row['sepal width (cm)']:.1f} "
            f"petal={row['petal length (cm)']:.1f}/{row['petal width (cm)']:.1f} "
            f"| dist={dist:.3f}\n"
        )
    return result

@tool
def classify_by_decision_tree(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = _dt.predict(sample)[0]
    proba = _dt.predict_proba(sample)[0]
    species = iris.target_names[prediction]
    confidence = round(proba[prediction] * 100, 1)

    node_indicator = _dt.decision_path(sample)
    node_ids = node_indicator.indices

    feature_names = iris.feature_names
    threshold = _dt.tree_.threshold
    feature = _dt.tree_.feature

    path_description = "Decision path:\n"
    for node_id in node_ids[:-1]:
        feat = feature_names[feature[node_id]]
        thresh = round(threshold[node_id], 2)
        actual_value = sample[0][feature[node_id]]
        direction = "<=" if actual_value <= thresh else ">"
        path_description += (
            f"  {feat} {direction} {thresh} "
            f"(sample value: {actual_value:.1f})\n"
        )

    return (
        f"Decision Tree prediction: {species} ({confidence}% confidence)\n"
        f"  setosa={proba[0]:.2f}, versicolor={proba[1]:.2f}, virginica={proba[2]:.2f}\n"
        f"{path_description}"
    )
@tool
def model_performance() -> str:
    knn_scores = cross_val_score(_knn, iris.data, iris.target, cv=5)
    dt_scores = cross_val_score(_dt, iris.data, iris.target, cv=5)

    result = "Model performance:\n"
    result += f"KNN accuracy: mean={knn_scores.mean():.3f}, std={knn_scores.std():.3f}\n"
    result += f"Decision Tree accuracy: mean={dt_scores.mean():.3f}, std={dt_scores.std():.3f}\n"

    return result

@tool
def feature_importance() -> str:
    importances = _dt.feature_importances_
    features = iris.feature_names

    pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    result = "Feature importance (Decision Tree):\n"
    for feat, val in pairs:
        result += f"  {feat}: {val:.3f}\n"

    return result

@tool
def compare_models(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    dt_pred = iris.target_names[_dt.predict(sample)[0]]

    _, indices = _knn.kneighbors(sample)
    neighbor_species = df.iloc[indices[0]]["species"]
    knn_pred = neighbor_species.mode()[0]

    agreement = dt_pred == knn_pred

    return (
        f"Model comparison:\n"
        f"  Decision Tree: {dt_pred}\n"
        f"  KNN: {knn_pred}\n"
        f"  Agreement: {'YES' if agreement else 'NO'}\n"
    )


@tool
def validate_input(
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
) -> str:
    sample = np.array([sepal_length, sepal_width, petal_length, petal_width])

    mins = df.drop(columns="species").min().values
    maxs = df.drop(columns="species").max().values

    warnings = []
    for i, (val, min_v, max_v) in enumerate(zip(sample, mins, maxs)):
        if val < min_v or val > max_v:
            warnings.append(
                f"{iris.feature_names[i]}={val:.1f} outside range [{min_v:.1f}, {max_v:.1f}]"
            )

    if not warnings:
        return "Input is within normal dataset range."

    return "Potential out-of-distribution input:\n" + "\n".join(warnings)

@tool
def combined_decision(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    dt_pred = iris.target_names[_dt.predict(sample)[0]]
    dt_proba = _dt.predict_proba(sample)[0]
    dt_conf = dt_proba.max()

    distances, indices = _knn.kneighbors(sample)
    neighbor_species = df.iloc[indices[0]]["species"]
    knn_pred = neighbor_species.mode()[0]

    agreement = dt_pred == knn_pred

    if agreement:
        final = dt_pred
        reason = "Both models agree"
    elif dt_conf > 0.8:
        final = dt_pred
        reason = "Decision Tree confident (>80%)"
    else:
        final = knn_pred
        reason = "Fallback to KNN"

    return (
        f"Final decision: {final}\n"
        f"Reason: {reason}\n"
        f"  Decision Tree: {dt_pred} (conf={dt_conf:.2f})\n"
        f"  KNN: {knn_pred}\n"
    )

IRIS_TOOLS = [
    get_flower_stats,
    find_nearest_neighbors,
    classify_by_decision_tree,
    model_performance,
    feature_importance,
    compare_models,
    validate_input,
    combined_decision,
]