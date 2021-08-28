# Topicaxis classifiers

This repository contains the scripts that are used to train the classifiers of Topicaxis.

## Installation

Use poetry to install the requirements. First activate the environment.

```bash
poetry shell
```

Then install the requirements.

```bash
poetry install
```

## Dataset

The datasets used by Topicaxis can't be shared for obvious reasons. To train the
classifiers you have to provide your own datasets. The document classifiers require
a file with one json encoded line for every sample document. The JSON schema should look
like this.

```json
{
  "category": "category-name",
  "text": "the document text"
}
```

Topicaxis was used to classify documents into the following categories. You have to use the category names
listed here.

| Category    | Category name    |
|-------------|------------------|
| programming | cat__programming |
| science     | cat__science     |
| politics    | cat__politics    |
| business    | cat__business    |
| technology  | cat__technology  |

You can use your category names however in that case you need to modify the bentoml classifier file to make it
use your categories instead.

## Training

Use the training script to train the classifiers.

```bash
./train_category_classifiers.sh
```

When training is complete there should be several bentoml models.

```bash
bentoml models list
```

You need to copy the model tags and add them in the `classifiers.py` script. After this is done build the bento.

```bash
bentoml build -f bentofile.yaml
```

## Docker

When training is complete you should have a bento model.

```bash
bento list
```

Build a docker image using that model tag

```bash
bentoml containerize topicaxis-classifiers:exampletag
```

Start the classifier using docker

```bash
docker run -it --rm -p 3000:3000 topicaxis-classifiers:exampletag serve --production --api-workers=1 --backlog=64
```
