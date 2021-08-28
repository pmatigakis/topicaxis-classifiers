from argparse import ArgumentParser
import json
import re

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import bentoml.transformers


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("categories")

    return parser.parse_args()


def get_dataset(dataset_file, categories):
    documents = []
    labels = []

    with open(dataset_file) as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text") or ""
            text = text.lower()
            text = re.sub(r'\b\d+\b', "$", text)
            text = re.sub(r'\s+', " ", text)
            if len(text) >= 100 and data["category"] in categories:
                documents.append(text)
                y = [0.0 for _ in range(len(categories))]
                y[categories.index(data["category"])] = 1.0
                labels.append(y)

    return train_test_split(
        documents,
        labels,
        test_size=0.33,
        random_state=42,
        stratify=labels
    )


def main():
    args = get_arguments()
    categories = args.categories.split(",")

    X_train, X_test, y_train, y_test = get_dataset(args.dataset_file, categories)
    print(f"Number of training instances: {len(X_train)}")
    print(f"Number of testing instances: {len(X_test)}")

    def train_generator():
        for document, label in zip(X_train, y_train):
            yield {"text": document, "label": label}

    def test_generator():
        for document, label in zip(X_test, y_test):
            yield {"text": document, "label": label}

    train_dataset = Dataset.from_generator(train_generator)
    test_dataset = Dataset.from_generator(test_generator)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True
        )

    small_train_dataset = train_dataset.map(tokenize_function, batched=True)
    small_eval_dataset = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=5,
        problem_type="multi_label_classification"
    )
    metric = evaluate.load("mse")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        return metric.compute(
            predictions=logits.flatten(),
            references=labels.flatten()
        )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        no_cuda=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.to("cpu")
    saved_model = bentoml.transformers.save_model("categories-bert-model", model)
    saved_tokenizer = bentoml.transformers.save_model("categories-bert-tokenizer", tokenizer)
    print(f"bentomo model saved at {saved_model}")
    print(f"bentomo tokenizer saved at {saved_tokenizer}")


if __name__ == "__main__":
    main()
