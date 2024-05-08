# German news quotation attribution dataset

This repository contains the dataset and evaluation code for the paper "Dataset of Quotation Attribution in German News Articles" by Fynn Petersen-Frey and Chris Biemann (2024). 


## Data Format:

The data is available as JSON files where document (news article) is a individual JSON file.

Each JSON file has the following format:

```json
{
    "annotations": [
        {
            "addressee": [],
            "cue": {
                "spans": [
                    {
                        "begin": 1,
                        "charBegin": 5,
                        "charEnd": 9,
                        "end": 2
                    }
                ],
                "text": "said",
                "tokenIds": [
                    1
                ]   
            },
            "frame": {
                "spans": [
                    {
                        "begin": 0,
                        "charBegin": 0,
                        "charEnd": 9,
                        "end": 2
                    }
                ],
                "text": "They said",
                "tokenIds": [
                    0, 1
                ]   
            },
            "hasNested": false,
            "id": 0,
            "isNested": false,
            "medium": "SW",
            "quote": ["0:2", "0:3", "0:4", "0:5"],
            "quote": {
                "spans": [
                    {
                        "begin": 2,
                        "charBegin": 10,
                        "charEnd": 29,
                        "end": 7
                    }
                ],
                "text": "this is a sentence.",
                "tokenIds": [
                    2, 3, 4, 5
                ]   
            },
            "speaker": {
                "spans": [
                    {
                        "begin": 0,
                        "charBegin": 0,
                        "charEnd": 4,
                        "end": 1
                    }
                ],
                "text": "They",
                "tokenIds": [
                    0
                ]   
            },
            "type": "Indirect"
        },
    ],
    "documentName": "Title of the document",
    "originalText": "They said this is a sentence.",
    "sentences": [
        {
            "begin": 0,
            "charBegin": 0,
            "charEnd": 29,
            "end": 7,
            "id": 0,
            "text": "They said this is a sentence.",
            "tokens": ["They", "said", "this", "is", "a", "sentence", "."]
        },
    ]
}
```

## Data Source

The data for includes news articles from the German [WIKINEWS](https://de.wikinews.org/), extracted from the [articles XML dump](https://dumps.wikimedia.org/dewikinews/).
The entire dump from April 2022 consists of 13,001 published articles, from which we sampled 1000 articles to annotate.
These annotated articles contain almost 250,000 tokens.

### License

The data is provided under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Evaluation

The evaluation script can be found under the `eval` directory.
