from typing import List

import bentoml
from bentoml.io import JSON
from torch.nn import Sigmoid


available_categories = [
    ("cat__programming", "nsvnfutajco3y4kj"),
    ("cat__science", "j45pcetajko3y4kj"),
    ("cat__politics", "bbynmytajgo3y4kj"),
    ("cat__business", "vmn7fldajgo3y4kj"),
    ("cat__technology", "5w4wfcdajko3y4kj")
]
category_models = {
    category: bentoml.sklearn.get(f"topicaxis-categories-{category}-basic:{tag}")
    for category, tag in available_categories
}
category_runners = {
    category: category_models[category].to_runner()
    for category, _ in available_categories
}

bert_classifier_model_tag = "wtpsegs7t2wdizjy"
bert_classifier_model = bentoml.transformers.get(f"categories-bert-model:{bert_classifier_model_tag}")
bert_classifier_model_runner = bert_classifier_model.to_runner()

bert_classifier_tokenizer_tag = "wwozgos7t2wdizjy"
bert_classifier_tokenizer = bentoml.transformers.get(
    f"categories-bert-tokenizer:{bert_classifier_tokenizer_tag}"
)
bert_classifier_tokenizer_runner = bert_classifier_tokenizer.to_runner()


runners = list(category_runners.values())
runners.append(bert_classifier_model_runner)
runners.append(bert_classifier_tokenizer_runner)

sigmoid = Sigmoid()
bert_classifier_available_categories = [
    "cat__programming",
    "cat__science",
    "cat__business",
    "cat__politics",
    "cat__technology"
]

svc = bentoml.Service(
    "topicaxis-classifiers",
    runners=runners
)


@svc.api(
    input=JSON(),
    output=JSON(),
    route="/categories/v1"
)
def categories_v1(documents: List[str]) -> List[dict]:
    results = {
        category: category_runners[category].predict_proba.run(documents)
        for category, _ in available_categories
    }

    return [
        {
            category: results[category][index][category_models[category].info.metadata["true_index"]]
            for category, _ in available_categories
        }
        for index in range(len(documents))
    ]


@svc.api(
    input=JSON(),
    output=JSON(),
    route="/categories/v2"
)
def categories_v2(documents: List[str]) -> List[dict]:
    encoded_input = bert_classifier_tokenizer_runner.run(
        documents,
        return_tensors='pt',
        padding="max_length",
        truncation=True
    )
    predictions = bert_classifier_model_runner.run(**encoded_input)
    predictions = sigmoid(predictions.logits).detach().numpy()

    return [
        dict(list(zip(bert_classifier_available_categories, result)))
        for result in predictions
    ]
