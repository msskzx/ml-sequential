# Machine Learning for Sequential Data

## Outline

- Hidden Markov Model 
- Word2Vec
- Neural Temporal Point Processes
- Hate speech detection
- Toxic speech classification

## Hidden Markov Model

- `Hidden Markov Model (HMM)` to generate and classify reviews

## Word2Vec

- `Word2Vec` to obtain word embeddings

## Neural Temporal Point Processes

- `Neural Temporal Point Processes` to model the time of occurrence of events

## Hate Speech Detection

- Tweets labels: `RACIST`, `SEXIST`, `NEITHER`
- Data Preprocessing

  - `LabelEncoder` for labels
  - `Universal Sentence Encoder` to get text embeddings

- Model: Fully Connected Neural Network

## Toxic Speech Classification

- Tweets labels: `none`, `racism`, `sexism`
- Data Preprocessing

  - `LabelEncoder` for labels
  - `BertTokenizer` to get text tokens
  - padding
  - CustomDataset
  - Split to `Train/Val/Test` `60/20/20`

- Model: `DistilBERT`
- Explanation using `SHAP`

![shap sexism](/ex06/shap-sexism.png)
