from argparse import ArgumentParser
import json
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from optuna import create_study
import bentoml.sklearn


class ClassifierModel:
    def __init__(self, classifier):
        super(ClassifierModel, self).__init__()

        self._classifier = classifier
        self._true_index = list(classifier.classes_).index(True)

    def predict(self, X):
        predictions = self._classifier.predict_proba(X)

        return predictions[:, self._true_index]


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("dataset_file")
    parser.add_argument("--n-trials", type=int, default=100)

    return parser.parse_args()


def objective(trial, X_train, X_test, y_train, y_test):
    max_df = trial.suggest_float("max_df", 0.5, 0.9)
    min_df = trial.suggest_int("min_df", 1, 5)
    percentile = trial.suggest_int("percentile", 10, 90)
    penalty = trial.suggest_categorical("penalty", ["l2"])
    c = trial.suggest_float("c", 0.001, 100.0)

    pipeline = Pipeline([
        ("feature_extractor", CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df)),
        ("feature_selector", SelectPercentile(chi2, percentile=percentile)),
        ("classifier", LogisticRegression(
            penalty=penalty,
            C=c,
            random_state=42,
            max_iter=1000,
            solver="liblinear"
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return f1_score(y_test, predictions)


def main():
    args = get_arguments()

    urls = set()
    documents = []
    classes = []
    with open(args.dataset_file) as f:
        for line in f:
            document = json.loads(line)
            text = document.get("text") or ""
            if not len(text) >= 1000:
                continue
            if document["url"] in urls:
                continue
            documents.append(document["text"])
            classes.append(document["category"] == args.category)
            urls.add(document["url"])

    X_train, X_test, y_train, y_test = train_test_split(
        documents, classes, test_size=0.33, random_state=42, stratify=classes
    )

    study = create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=args.n_trials)

    max_df = study.best_params["max_df"]
    min_df = study.best_params["min_df"]
    percentile = study.best_params["percentile"]
    c = study.best_params["c"]
    penalty = study.best_params["penalty"]

    pipeline = Pipeline([
        ("feature_extractor", CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df)),
        ("feature_selector", SelectPercentile(chi2, percentile=percentile)),
        ("classifier", LogisticRegression(
            penalty=penalty,
            C=c,
            random_state=42,
            max_iter=1000,
            solver="liblinear"
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    class_counts = Counter(classes)
    model_tag = bentoml.sklearn.save_model(
        f"topicaxis-categories-{args.category}-basic",
        pipeline,
        metadata={
            "classification_report": classification_report(y_test, predictions),
            "f1": study.best_trial.values[0],
            "category": args.category,
            "n_trials": args.n_trials,
            "positive_samples": class_counts[True],
            "total_samples": len(classes),
            "best_params": study.best_params,
            "confusion_matrix": cm,
            "true_index": list(pipeline["classifier"].classes_).index(True)
        },
        signatures={
            "predict_proba": {
                "batchable": True,
                "batch_dim": 0,
            },
        }
    )

    print(f"{args.category} category model saved: {model_tag}")


if __name__ == '__main__':
    main()
