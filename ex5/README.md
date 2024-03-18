# Social Computing


## Outline

- Toxic Speech Classification with BERT and Explanation using SHAP
- Hate Speech Detection
- Graph Clustering
- Centrality Measures
- Twitch Streamers' Friends Network

## Toxic Speech Classification with BERT and Explanation using SHAP

![shap sexism](/ex06/shap-sexism.png)

- Tweets labels: `none`, `racism`, `sexism`
- Data Preprocessing: 
    - `LabelEncoder` for labels
    - `BertTokenizer` to get text tokens
    - padding
    - CustomDataset
    - Split to `Train/Val/Test` `60/20/20`
- Model: `DistilBERT`
- Explanation using `SHAP`


## Hate Speech Detection with

![base model](/ex05/img/base_model.png)

- Tweets labels: `RACIST`, `SEXIST`, `NEITHER`
- Data Preprocessing: 
    - `LabelEncoder` for labels
    - `Universal Sentence Encoder` to get text embeddings
- Model: Fully Connected Neural Network

![embeddings](/ex05/img/embeddings.png)

## Clustering

- `Louvain` clustering
- `K-Means` clustering
- `Gaussian Mixture Model` clustering

![kmeans vs. gmm](/ex03/kmeans_vs_gmm.png)
<center>Two Gaussian Distributions and one Normal Distribution</center>


## Centrality Measures

Implementation of `Betweenness Centrality` and `PageRank` centrality measures is performed on a `Krackhardt Kite Graph`.

- `Kite centrality`
- `Betweenness centrality`
- `Epsilon Betweenness centrality`
- `PageRank` centrality measure
- Personalized `PageRank` centrality

## Twitch Streamers' Friends Network

A graph of twitch friends network is constructed and analyzed.

- Sparse Vector Representation: convert dense vector to sparse vector
- Data Preprocessing
    - filter streamers
    - merge nodes with edges

### Graph

- use `networkX` to construct graph from preprocessed data frame
- visualize graph using `Spring Layout` which positions nodes using `Fruchterman-Reingold force-directed algorithm`

![spring layout](/ex01/spring_layout.png)
<center>Spring Layout</center>