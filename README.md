<div align="center">

# AI & ML Knowledge Base — Algorithms Guide

**Notes from the Master's in AI & Programming — Theory, algorithms and practical implementations**

![Status](https://img.shields.io/badge/status-en%20progreso-yellow)
![Made with](https://img.shields.io/badge/made%20with-%E2%98%95%20%2B%20python-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

</div>

---

## About this repository

This repository is my personal knowledge base on the main concepts, algorithms, architectures, and techniques of **Artificial Intelligence, Machine Learning, Deep Learning, Generative AI**, and related fields.

The goal is not only to describe *how* each algorithm works, but also to answer the practical questions that really matter when choosing one:

- What problem does this algorithm solve?
- Why would I use it?
- When should I use it?
- When should I avoid it?
- What are its main advantages and limitations?
- What alternatives should I consider?
- How do I implement it in practice?

Each section links (or will progressively link) to **practical implementations** in dedicated GitHub repositories, with notebooks, datasets, and results.

> **How ​​to use this repo:** This README serves as a general theoretical index. Each block of algorithms has (or will have) its own folder with extended notes and its own linked code repo in the section [Repositorios prácticos](#-repositorios-prácticos).

---

## Table of Contents

1. [AI Overview](#1-artificial-intelligence-overview)
2. [Choosing an Algorithm](#2-choosing-an-algorithm)
3. [Supervised Learning](#3-supervised-learning)
4. [Regression](#4-linear-regression) — Linear, Ridge, Lasso, Elastic Net
5. [Classical Classification](#8-logistic-regression) — Logistic Regression, Decision Trees, Random Forest
6. [Gradient Boosting](#11-gradient-boosting) — XGBoost, LightGBM, CatBoost
7. [SVM, KNN, Naive Bayes, LDA](#15-support-vector-machines)
8. [Unsupervised Learning](#19-unsupervised-learning) — K-Means, Hierarchical, DBSCAN, GMM
9. [Reduction of dimensionality](#24-principal-component-analysis) — PCA, t-SNE, UMAP
10. [Anomaly Detection](#27-anomaly-detection) — Isolation Forest, One-Class SVM, LOF
11. [Time series](#31-time-series-algorithms) — ARIMA, SARIMA, Prophet, VAR, Markov Chains, Monte Carlo, Kalman Filter
12. [Deep Learning](#41-neural-networks) — MLP, CNN, RNN, LSTM, GRU
13. [Transformers & LLMs](#47-transformers) — Transformers, LLMs, Embeddings, RAG
14. [Generative models](#51-autoencoders) — Autoencoders, VAE, GAN, Diffusion
15. [Graph Neural Networks](#55-graph-neural-networks)
16. [Classic NLP](#56-natural-language-processing) — TF-IDF, Word2Vec, Topic Modeling
17. [Recommender systems](#60-recommender-systems)
18. [Reinforcement Learning](#64-reinforcement-learning) — Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO
19. [Optimization](#71-genetic-algorithms) — Genetic Algorithms, PSO
20. [Probabilistic models](#73-bayesian-networks) — Bayesian Networks, MCMC
21. [Symbolic AI](#75-expert-systems) — Expert Systems, Fuzzy Logic
22. [Semi-supervised learning](#77-semi-supervised-learning) — Semi-supervised learning, transfer learning, fine-tuning
24. [Explainable AI](#80-explainable-ai) — SHAP, LIME
25. [Model evaluation](#83-model-evaluation)
26. [Algorithm selection cheat sheet](#84-algorithm-selection-cheat-sheet)
27. [Practical flow for tabular data](#85-tabular-data-practical-model-selection)
28. [The most important rule](#86-the-most-important-rule)
29. [Recommended learning path](#87-recommended-learning-path)
30. [Repository structure](#88-repository-structure)
31. Repositories practicals](#-practical-repositories)

---

# 1. Artificial Intelligence Overview

Artificial Intelligence is a broad field concerned with building systems capable of performing tasks that normally require human intelligence.

```text
Artificial Intelligence
│
├── Symbolic AI
│   ├── Expert Systems
│   ├── Rule-Based Systems
│   ├── Knowledge Graphs
│   └── Search / Planning
│
├── Machine Learning
│   │
│   ├── Supervised Learning
│   │   ├── Regression
│   │   └── Classification
│   │
│   ├── Unsupervised Learning
│   │   ├── Clustering
│   │   ├── Dimensionality Reduction
│   │   └── Anomaly Detection
│   │
│   ├── Semi-Supervised Learning
│   │
│   └── Reinforcement Learning
│
├── Deep Learning
│   ├── MLP
│   ├── CNN
│   ├── RNN
│   ├── LSTM / GRU
│   ├── Transformers
│   ├── Autoencoders
│   ├── GANs
│   ├── Diffusion Models
│   └── Graph Neural Networks
│
├── Natural Language Processing
│
├── Computer Vision
│
├── Generative AI
│
└── Robotics & Autonomous Systems
```

---

# 2. Choosing an Algorithm

Before choosing an algorithm, the first question should be:

> **What kind of problem am I trying to solve?**

A simplified decision process is:

```text
Do I have a target variable?
│
├── YES → Supervised Learning
│   │
│   ├── Continuous target → Regression
│   │
│   └── Categorical target → Classification
│
└── NO → Unsupervised Learning
    │
    ├── Find groups → Clustering
    ├── Reduce variables → Dimensionality Reduction
    └── Detect unusual observations → Anomaly Detection
```

Other problem types require more specialized approaches:

```text
Sequential data / future values
→ Time Series Models

Images
→ CNN / Vision Transformers

Text
→ NLP / Transformers / LLMs

Actions + rewards
→ Reinforcement Learning

Generate new content
→ GAN / VAE / Diffusion / LLM

Graph relationships
→ Graph Algorithms / GNN

Recommendation
→ Recommender Systems
```

---

# 3. Supervised Learning

Supervised learning uses labeled data:

```text
X → Model → y
```

The model learns the relationship between input variables `X` and a known target `y`.

The two main supervised tasks are:

- **Regression** → predict numerical values.
- **Classification** → predict categories.

---

# 4. Linear Regression

## Purpose

Predict a continuous numerical variable.

Examples: house prices, revenue, temperature, energy consumption, travel time.

The basic model is:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

## When to use it

Use Linear Regression when:

- the target is numerical;
- relationships are approximately linear;
- interpretability is important;
- you need a strong baseline.

## Advantages

- Extremely fast.
- Easy to interpret.
- Easy to implement.
- Works well with relatively small datasets.
- Useful as a baseline.

## Limitations

- Assumes approximately linear relationships.
- Sensitive to outliers.
- Can suffer from multicollinearity.
- Usually performs poorly on complex nonlinear patterns.

---

# 5. Ridge Regression

Ridge Regression extends Linear Regression with **L2 regularization**. It penalizes large model coefficients.

## Use it when

- many variables are correlated;
- Linear Regression is overfitting;
- you want to keep all features.

Ridge tends to reduce coefficients but rarely makes them exactly zero.

---

# 6. Lasso Regression

Lasso uses **L1 regularization**. Unlike Ridge, Lasso can reduce some coefficients exactly to zero.

## Use it when

- feature selection is useful;
- the dataset contains many potentially irrelevant variables;
- you want a simpler model.

---

# 7. Elastic Net

Elastic Net combines:

```text
L1 regularization + L2 regularization
```

It combines characteristics of both Lasso and Ridge.

## Use it when

- many predictors are correlated;
- you want feature selection;
- pure Lasso is unstable.

---

# 8. Logistic Regression

Despite its name, Logistic Regression is primarily a **classification algorithm**. It estimates probabilities such as:

```text
P(y = 1 | X)
```

Typical applications: fraud / non-fraud, churn / no churn, disease / healthy, default / no default, spam / not spam.

## When to use it

Use Logistic Regression when:

- you need a classification baseline;
- interpretability is important;
- relationships are relatively simple;
- probabilities are useful.

## Advantages

- Fast.
- Interpretable.
- Produces probabilities.
- Works well with high-dimensional sparse data.

## Limitations

- Linear decision boundary.
- Cannot naturally model highly complex nonlinear relationships.

---

# 9. Decision Trees

Decision Trees repeatedly split the dataset using rules.

```text
Age > 35?
│
├── Yes
│   └── Income > 40K?
│       ├── Yes → Customer likely to buy
│       └── No  → Customer unlikely to buy
│
└── No → Customer unlikely to buy
```

They can perform classification or regression.

## When to use them

Use Decision Trees when:

- relationships are nonlinear;
- interpretability is important;
- interactions between variables exist;
- numerical and categorical features are present.

## Advantages

- Very intuitive.
- No feature scaling required.
- Handles nonlinear relationships.
- Captures feature interactions.

## Limitations

A single tree can easily overfit. For this reason, ensemble tree algorithms are often preferred.

---

# 10. Random Forest

Random Forest creates many Decision Trees.

```text
Data
│
├── Tree 1
├── Tree 2
├── Tree 3
├── ...
└── Tree N

        ↓

Aggregation

Classification → majority vote
Regression     → average
```

## When to use it

Random Forest is an excellent general-purpose algorithm for **tabular data**.

Use it when:

- relationships are nonlinear;
- interactions exist;
- interpretability is useful but not critical;
- you need a strong model with relatively little tuning.

## Advantages

- Robust.
- Handles nonlinear data.
- Less prone to overfitting than individual trees.
- Can estimate feature importance.
- No scaling usually required.

## Limitations

- Large forests can consume significant memory.
- Less interpretable than a single Decision Tree.
- Often outperformed by Gradient Boosting on structured tabular datasets.

---

# 11. Gradient Boosting

Gradient Boosting builds trees sequentially. Each new tree tries to correct mistakes made by previous trees.

```text
Model 1 → Errors → Model 2 → Remaining errors → Model 3 → ...
```

Important implementations include GradientBoosting, XGBoost, LightGBM, and CatBoost.

---

# 12. XGBoost

**Extreme Gradient Boosting** is one of the most popular algorithms for structured/tabular datasets.

Common applications: fraud detection, credit risk, churn, customer scoring, demand prediction, traffic prediction, competitions.

## When to use XGBoost

Use it when:

- the dataset is tabular;
- relationships are nonlinear;
- predictive performance is important;
- feature interactions matter;
- missing values may exist.

## Advantages

- Excellent predictive performance.
- Handles nonlinear relationships.
- Built-in regularization.
- Supports missing values.
- Works with SHAP for explainability.

## Limitations

- More hyperparameters than Random Forest.
- Can overfit if improperly configured.
- Training can become expensive on very large datasets.

---

# 13. LightGBM

LightGBM is a highly optimized gradient boosting framework, especially useful for very large datasets, datasets with many features, and situations requiring fast training.

## Advantages

- Extremely fast.
- Memory efficient.
- Strong predictive performance.

## Limitation

Its leaf-wise growth strategy can overfit small datasets if not carefully configured.

---

# 14. CatBoost

CatBoost is another gradient boosting algorithm designed particularly for **categorical variables**.

## Use it when

- your dataset contains many categorical features;
- you want to reduce complex preprocessing;
- you have high-cardinality categorical variables.

Typical applications: customer behavior, marketing, e-commerce, financial risk.

---

# 15. Support Vector Machines

Support Vector Machines try to find a decision boundary that maximizes the separation between classes.

```text
Class A     |     Class B

 ● ● ●      |      ○ ○ ○
 ● ●        |       ○ ○
            |
      Maximum Margin
```

With kernels, SVM can create nonlinear decision boundaries (Linear, Polynomial, RBF).

## When to use SVM

Use SVM when:

- datasets are small or medium-sized;
- dimensionality is high;
- clear separation exists between classes.

## Advantages

- Powerful mathematical foundation.
- Effective in high-dimensional spaces.
- Kernel trick supports nonlinear relationships.

## Limitations

- Training becomes expensive with large datasets.
- Requires feature scaling.
- Hyperparameter tuning can be important.

---

# 16. K-Nearest Neighbors

KNN predicts based on the closest observations in the dataset.

```text
Find K nearest observations → Majority class → Prediction
```

## Use KNN when

- the dataset is relatively small;
- similar observations should have similar outputs;
- the decision boundary is irregular.

## Advantages

- Extremely simple.
- No real training phase.
- Naturally nonlinear.

## Limitations

- Prediction becomes slow with large datasets.
- Sensitive to feature scaling.
- Performs poorly in very high dimensions (curse of dimensionality).

---

# 17. Naive Bayes

Naive Bayes is based on Bayes' theorem:

```text
P(A|B) = P(B|A) × P(A) / P(B)
```

It assumes conditional independence between features. Common variants: Gaussian, Multinomial, Bernoulli Naive Bayes.

## When to use it

Particularly useful for spam detection, sentiment classification, document classification, text categorization.

## Advantages

- Extremely fast.
- Works well with small datasets.
- Strong baseline for NLP.

## Limitation

The independence assumption is often unrealistic.

---

# 18. Linear Discriminant Analysis

LDA finds combinations of features that maximize separation between classes. It can be used for classification or dimensionality reduction.

Use it when classes are reasonably well separated and the statistical assumptions are acceptable.

---

# 19. Unsupervised Learning

Unsupervised learning operates without a target variable.

```text
X → Algorithm → Hidden Structure
```

Typical objectives: discovering groups, finding patterns, reducing dimensionality, detecting unusual observations.

---

# 20. K-Means Clustering

K-Means divides observations into `K` clusters, minimizing distances between observations and their cluster centroid.

Typical uses: customer segmentation, mobility patterns, behavioral segmentation, geographic analysis, product segmentation.

## Use it when

- the number of clusters can be estimated;
- clusters are roughly spherical;
- features are numerical.

## Advantages

- Fast.
- Simple.
- Scales well.

## Limitations

- `K` must be specified.
- Sensitive to outliers.
- Sensitive to feature scaling.
- Performs poorly with irregularly shaped clusters.

---

# 21. Hierarchical Clustering

Hierarchical clustering progressively merges or separates observations. The result can be visualized using a **dendrogram**.

```text
        ┌── A
    ┌───┤
    │   └── B
────┤
    │   ┌── C
    └───┤
        └── D
```

## Use it when

- you want to understand relationships between clusters;
- the dataset is relatively small;
- the number of clusters is initially unknown.

---

# 22. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise** finds dense regions of observations. Unlike K-Means, DBSCAN can discover arbitrarily shaped clusters.

## Advantages

- Does not require specifying the number of clusters.
- Identifies noise.
- Finds irregularly shaped clusters.

## Useful for

GPS data, geospatial clustering, anomaly detection, mobility data.

## Limitations

Selecting appropriate values for `eps` and `min_samples` can be difficult.

---

# 23. Gaussian Mixture Models

GMM treats the data as a combination of multiple Gaussian distributions. Unlike K-Means, it provides **probabilistic cluster membership**.

```text
Customer A
Cluster 1 → 70%
Cluster 2 → 25%
Cluster 3 → 5%
```

Use GMM when cluster boundaries overlap and probabilistic membership is useful.

---

# 24. Principal Component Analysis

PCA reduces the number of dimensions while preserving as much variance as possible.

```text
100 variables → PCA → 10 principal components
```

## Use PCA for

Dimensionality reduction, visualization, noise reduction, preprocessing, handling correlated variables.

## Limitations

Principal components are combinations of original variables and therefore may be difficult to interpret.

---

# 25. t-SNE

t-SNE is primarily a visualization technique. It projects high-dimensional data into 2D or 3D.

## Useful for

Visualizing embeddings, exploring clusters, visualizing neural network representations.

It should generally **not be treated as a predictive model**.

---

# 26. UMAP

UMAP is another nonlinear dimensionality reduction algorithm. Compared with t-SNE, UMAP is often faster, more scalable, and better at preserving some global structure.

Commonly used for: embedding visualization, clustering, high-dimensional biological data, NLP embeddings.

---

# 27. Anomaly Detection

Anomaly detection attempts to identify unusual observations.

Typical applications: fraud, cyberattacks, industrial failures, unusual transactions, sensor failures.

---

# 28. Isolation Forest

Isolation Forest isolates observations using random splits. Anomalies usually require fewer splits to isolate.

## Use it when

- labels for anomalies are unavailable;
- the dataset is large;
- the data is tabular.

It is one of the most practical general-purpose anomaly detection algorithms.

---

# 29. One-Class SVM

One-Class SVM learns the region containing normal observations. Points outside that region are considered anomalies.

Useful for small or medium datasets, although it can become computationally expensive.

---

# 30. Local Outlier Factor

LOF compares the local density of an observation with the density of its neighbors. Useful when anomalies are unusual **relative to their local neighborhood**.

---

# 31. Time Series Algorithms

Time-series problems involve observations ordered in time.

Examples: traffic, demand, sales, energy, stock prices, sensor measurements, passenger flows.

---

# 32. ARIMA

ARIMA stands for:

```text
AR = AutoRegression
I  = Integrated
MA = Moving Average
```

It models future observations using past observations and previous prediction errors.

## Use ARIMA when

- data is univariate;
- relationships are mostly linear;
- the dataset is not extremely large;
- interpretability matters.

---

# 33. SARIMA

SARIMA extends ARIMA by including seasonality. Useful for daily demand, weekly traffic, monthly sales, yearly seasonality.

---

# 34. Exponential Smoothing

Methods such as Holt and Holt-Winters model level, trend, and seasonality. Useful for relatively stable time series with clear seasonal patterns.

---

# 35. Prophet

Prophet models time series using components such as:

```text
Trend + Seasonality + Holidays + Special Events
```

Useful for business forecasting where calendar effects are important.

---

# 36. Vector Autoregression

VAR is used when several time series influence each other.

```text
Traffic volume, Speed, Occupancy, Weather
```

Each variable can depend on previous values of the others.

---

# 37. Markov Chains

A Markov Chain models transitions between states. The fundamental assumption is:

> The next state primarily depends on the current state.

```text
Free Traffic → Moderate → Dense → Congestion
```

A transition matrix may look like:

```text
              Next State

Current      Free  Moderate Dense Congested

Free          .90    .08     .02     .00
Moderate      .10    .75     .12     .03
Dense         .02    .15     .65     .18
Congested     .01    .04     .20     .75
```

## Use Markov Chains when

- the problem naturally contains states;
- state transitions are important;
- probabilistic future scenarios are required.

Typical applications: traffic, customer journeys, reliability, finance, weather, queue systems.

---

# 38. Hidden Markov Models

HMM extends Markov models by assuming that the true state cannot be directly observed. Instead, observations are generated by hidden states.

Typical applications: speech recognition, activity recognition, biological sequences, regime detection.

---

# 39. Monte Carlo Simulation

Monte Carlo methods use repeated random simulations to estimate possible outcomes.

```text
Initial State
     │
     ├── Simulation 1
     ├── Simulation 2
     ├── ...
     └── Simulation N
```

After thousands of simulations, probabilities can be estimated.

## Use Monte Carlo when

- uncertainty is important;
- many future scenarios are possible;
- probabilities are more useful than a single prediction.

Typical applications: risk, finance, traffic forecasting, engineering, project management.

Monte Carlo is often combined with predictive models:

```text
Machine Learning → Probability Distribution → Markov Model → Monte Carlo Simulation → Future Risk Distribution
```

---

# 40. Kalman Filter

The Kalman Filter estimates the hidden state of a dynamic system using noisy measurements.

Common applications: GPS, robotics, navigation, tracking, sensor fusion.

---

# 41. Neural Networks

Artificial Neural Networks are inspired by interconnected biological neurons.

```text
Inputs → Weighted Sum → Activation Function → Output
```

A neural network usually contains:

```text
Input Layer → Hidden Layers → Output Layer
```

---

# 42. Multilayer Perceptron

The MLP is the classical fully connected neural network.

```text
Input → Dense Layer → ReLU → Dense Layer → ReLU → Output
```

## Use MLPs for

Tabular classification, tabular regression, general nonlinear modeling.

However, for many tabular datasets, algorithms such as XGBoost or CatBoost may perform better with less tuning.

---

# 43. Convolutional Neural Networks

CNNs are designed to identify spatial patterns and became especially important in Computer Vision.

Typical tasks: image classification, object detection, medical image analysis, face recognition, segmentation.

Convolution filters can progressively identify:

```text
Edges → Shapes → Objects → Complex structures
```

---

# 44. Recurrent Neural Networks

RNNs process sequential information. Their hidden state carries information from previous observations.

Used for text, speech, sensor signals, time series. However, traditional RNNs struggle with long-term dependencies.

---

# 45. LSTM

Long Short-Term Memory networks improve RNNs by introducing memory gates, designed to preserve relevant information over longer sequences.

Typical applications: time-series prediction, text generation, speech, sequential anomaly detection.

---

# 46. GRU

Gated Recurrent Units are a simplified alternative to LSTMs, usually containing fewer parameters and training faster.

Use GRU when sequential modeling is needed but a slightly simpler architecture is preferred.

---

# 47. Transformers

Transformers introduced the **attention mechanism** as the central component for processing sequences. Instead of processing tokens strictly one at a time, the model learns which parts of the input should receive attention.

```text
Input Tokens → Embeddings → Self-Attention → Feed Forward Network → Multiple Transformer Layers
```

Examples: BERT, GPT, T5, LLaMA-style models, Vision Transformers.

---

# 48. Large Language Models

LLMs are large neural networks, usually based on Transformer architectures, trained on enormous text datasets.

Typical capabilities: text generation, summarization, translation, reasoning, code generation, information extraction, conversational systems.

Use LLMs when a problem involves complex natural language understanding or generation.

---

# 49. Embeddings

Embeddings convert complex information into numerical vectors.

```text
"King"  → [0.22, -0.51, 0.84, ...]
"Queen" → [0.25, -0.48, 0.81, ...]
```

Semantically similar objects tend to have similar vectors. Embeddings can represent text, images, users, products, audio, graphs.

Fundamental for: semantic search, recommendation systems, clustering, Retrieval-Augmented Generation.

---

# 50. Retrieval-Augmented Generation

RAG combines information retrieval with generative models.

```text
User Question → Embedding → Vector Database → Relevant Documents → LLM → Answer
```

Use RAG when an LLM needs access to company documents, private knowledge, frequently updated information, or specialized domain knowledge.

---

# 51. Autoencoders

Autoencoders learn to reconstruct their own inputs.

```text
Input → Encoder → Latent Representation → Decoder → Reconstructed Input
```

Use them for dimensionality reduction, anomaly detection, representation learning, denoising.

---

# 52. Variational Autoencoders

VAEs extend Autoencoders by learning a probability distribution in the latent space. This makes them generative models, capable of creating new samples similar to the training data.

---

# 53. Generative Adversarial Networks

GANs contain two competing neural networks:

```text
Generator → Fake Data → Discriminator → Real or Fake?
```

The Generator tries to fool the Discriminator; the Discriminator tries to distinguish generated data from real data.

Typical applications: image generation, image enhancement, synthetic datasets, style transfer.

---

# 54. Diffusion Models

Diffusion models gradually add noise to data and train a neural network to reverse the process.

```text
Image → add noise → Noise → learn reverse process → Generated Image
```

Modern image generation systems heavily rely on diffusion-based approaches, useful for image generation, image editing, video generation, audio generation.

---

# 55. Graph Neural Networks

GNNs operate on graph-structured data (Nodes + Edges).

```text
Social Network

Person ─── Person
   │          │
 Person ─── Person
```

Typical applications: social networks, fraud detection, recommendation, molecular analysis, transport networks, knowledge graphs.

---

# 56. Natural Language Processing

NLP focuses on processing and understanding human language.

Classical techniques: Bag of Words, TF-IDF, N-grams, Word2Vec, GloVe, Topic Modeling.

Modern NLP is dominated by embeddings, Transformers, and LLMs.

---

# 57. TF-IDF

TF-IDF measures how important a word is to a document relative to a collection of documents.

Useful for document classification, keyword extraction, classical search, text similarity. It remains a very strong baseline for many text classification problems.

---

# 58. Word2Vec

Word2Vec learns numerical representations of words using two major architectures:

```text
CBOW: Context → Target Word
Skip-Gram: Target Word → Context
```

Word2Vec was an important step toward modern embedding systems.

---

# 59. Topic Modeling

Algorithms such as **Latent Dirichlet Allocation** attempt to discover hidden topics in collections of documents.

```text
Topic 1: football, player, match, team
Topic 2: market, stock, investment, bank
Topic 3: model, neural, training, data
```

Useful for exploring large document collections.

---

# 60. Recommender Systems

Recommendation algorithms predict what a user may prefer.

Typical applications: Netflix, Amazon, Spotify, social networks, e-commerce.

---

# 61. Collaborative Filtering

Recommendations are based on the behavior of similar users.

```text
User A likes X and Y
User B likes X
→ Recommend Y to User B
```

---

# 62. Content-Based Filtering

Recommendations are based on item characteristics.

```text
User likes: Science Fiction + Space + Thrillers
→ Recommend similar movies
```

---

# 63. Matrix Factorization

Matrix Factorization decomposes a large user-item matrix into latent representations. Popular techniques: SVD, ALS. Widely used in recommender systems.

---

# 64. Reinforcement Learning

Reinforcement Learning is based on an agent interacting with an environment.

```text
Agent → Action → Environment → Reward + New State → Agent
```

The objective is to maximize cumulative reward.

Applications: robotics, games, autonomous vehicles, resource allocation, optimization.

---

# 65. Q-Learning

Q-Learning learns the value of performing an action in a specific state, learning a function `Q(state, action)`. The agent chooses actions that maximize long-term reward.

---

# 66. SARSA

SARSA is similar to Q-Learning but learns using the action actually selected by the current policy. It is an **on-policy** algorithm.

---

# 67. Deep Q-Network

DQN replaces the traditional Q-table with a neural network.

```text
State → Neural Network → Q value for each action
```

This allows reinforcement learning to operate in much larger state spaces.

---

# 68. Policy Gradient

Instead of estimating action values, Policy Gradient algorithms directly learn a policy:

```text
State → Probability of each Action
```

Useful for complex or continuous action spaces.

---

# 69. Actor-Critic

Actor-Critic methods combine two models:

```text
Actor  → decides what action to take
Critic → evaluates the action
```

Important algorithms: A2C, A3C, PPO, SAC, DDPG.

---

# 70. PPO

Proximal Policy Optimization is one of the most commonly used modern reinforcement learning algorithms. It provides relatively stable training and works for many different environments.

---

# 71. Genetic Algorithms

Genetic Algorithms are inspired by biological evolution.

```text
Initial Population → Fitness Evaluation → Selection → Crossover → Mutation → New Generation
```

Use Genetic Algorithms for difficult optimization problems where gradient-based optimization is unavailable.

---

# 72. Particle Swarm Optimization

PSO is inspired by collective behavior such as bird flocks or fish schools. A population of candidate solutions moves through the search space. Useful for optimization problems with complex search spaces.

---

# 73. Bayesian Networks

Bayesian Networks represent probabilistic relationships between variables using a directed graph.

```text
Weather → Traffic → Arrival Delay
```

Useful when uncertainty and causal or conditional relationships need to be explicitly modeled.

---

# 74. Monte Carlo Markov Chain

MCMC techniques generate samples from complex probability distributions. Popular algorithms: Metropolis-Hastings, Gibbs Sampling, Hamiltonian Monte Carlo. Widely used in Bayesian statistics.

---

# 75. Expert Systems

Expert Systems represent knowledge using explicit rules.

```text
IF temperature > 40
AND pressure > threshold
THEN alert = critical
```

Useful when business or technical knowledge can be explicitly represented. Advantages: explainability, deterministic behavior, easy auditing. Can also be combined with Machine Learning.

---

# 76. Fuzzy Logic

Traditional logic uses `TRUE` / `FALSE`. Fuzzy Logic allows partial truth:

```text
0.0 → completely false
0.5 → partially true
1.0 → completely true
```

Useful when concepts are inherently vague (e.g. temperature is "hot", traffic is "heavy", risk is "high").

Applications: control systems, industrial automation, decision support.

---

# 77. Semi Supervised Learning

Semi-supervised learning combines a small amount of labeled data with a large amount of unlabeled data. Useful when labels are expensive to obtain.

Common approaches: pseudo-labeling, self-training, consistency regularization.

---

# 78. Transfer Learning

Transfer Learning takes knowledge from a pretrained model and adapts it to another task.

```text
Large Image Model → Pretrained Knowledge → Fine-tune with Medical Images
```

This dramatically reduces the amount of data and computation required. Fundamental in Computer Vision, NLP, and Generative AI.

---

# 79. Fine-Tuning

Fine-tuning adapts a pretrained model using task-specific data. For LLMs, approaches include full fine-tuning, LoRA, QLoRA, and instruction tuning.

Use fine-tuning when the model must learn new behavior, style, or domain-specific patterns.

---

# 80. Explainable AI

High-performing models are not always easy to interpret. Explainable AI techniques help understand model predictions.

Important techniques: Feature Importance, SHAP, LIME, Partial Dependence Plots, ICE plots, Saliency Maps, Grad-CAM.

---

# 81. SHAP

SHAP explains how much each feature contributed to a model prediction.

```text
Fraud Probability = 82%

Transaction Amount    +20%
Foreign Location      +15%
Night Transaction     +11%
Trusted Merchant      -8%
Account Age           -5%
```

Particularly useful with XGBoost, LightGBM, CatBoost, Random Forest, and other complex ML models.

---

# 82. LIME

LIME explains an individual model prediction by approximating the model locally with a simpler interpretable model. It is model-agnostic. Use it when individual predictions need to be explained.

---

# 83. Model Evaluation

Choosing an algorithm is only part of the problem — models must also be evaluated correctly.

## Regression

```text
MAE, MSE, RMSE, R², MAPE
```

## Classification

```text
Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC, Log Loss
```

For imbalanced datasets, accuracy can be misleading. Metrics such as Precision, Recall, F1, and PR-AUC are often more informative.

---

# 84. Algorithm Selection Cheat Sheet

| Problem                         | Good Starting Algorithms                       |
| -------------------------------- | ----------------------------------------------- |
| Simple regression                | Linear Regression                               |
| Nonlinear tabular regression     | Random Forest / XGBoost                         |
| Simple binary classification     | Logistic Regression                             |
| Complex tabular classification   | XGBoost / LightGBM / CatBoost                   |
| Small high-dimensional dataset   | SVM                                             |
| Text classification baseline     | TF-IDF + Logistic Regression                    |
| Customer segmentation            | K-Means                                         |
| Geographic clustering            | DBSCAN                                          |
| Probabilistic clustering         | GMM                                             |
| Dimensionality reduction         | PCA                                             |
| Data visualization               | PCA / UMAP / t-SNE                              |
| Anomaly detection                | Isolation Forest                                |
| Seasonal time series             | SARIMA / Exponential Smoothing                  |
| Complex time series              | XGBoost / LSTM / Transformer                    |
| State transitions                | Markov Chains                                   |
| Future probability scenarios     | Monte Carlo                                     |
| Images                           | CNN / Vision Transformer                        |
| Sequential data                  | LSTM / GRU / Transformer                        |
| Language understanding           | Transformer                                     |
| Text generation                  | LLM                                             |
| Semantic search                  | Embeddings                                      |
| Private knowledge + LLM          | RAG                                             |
| Image generation                 | Diffusion Models                                |
| Synthetic data                   | GAN / VAE / Diffusion                           |
| Graph relationships              | GNN                                             |
| Recommendation                   | Collaborative Filtering / Matrix Factorization  |
| Autonomous decision making       | Reinforcement Learning                          |
| Complex optimization             | Genetic Algorithms / PSO                        |
| Transparent deterministic rules  | Expert Systems                                  |
| Uncertain relationships          | Bayesian Networks                               |

---

# 85. Tabular Data: Practical Model Selection

For structured datasets, a practical workflow is often:

```text
Linear / Logistic Regression
        ↓
   Decision Tree
        ↓
   Random Forest
        ↓
XGBoost / LightGBM / CatBoost
        ↓
   Neural Network
```

This progression helps determine whether the additional complexity actually improves performance.

> **Gradient Boosting algorithms are often stronger starting points than Deep Neural Networks** for many real-world tabular datasets.

Deep Learning becomes especially attractive when working with large quantities of images, text, audio, video, complex sequential data, or multimodal data.

---

# 86. The Most Important Rule

There is no universally best algorithm. The best choice depends on:

```text
Problem Type + Dataset Size + Number of Features + Data Quality
+ Interpretability Requirements + Latency Requirements
+ Computational Resources + Business Constraints
```

A sophisticated model is not automatically a better model. A good Machine Learning workflow usually starts with a simple baseline and progressively increases complexity only when necessary.

---

# 87. Recommended Learning Path

```text
1. Python
2. NumPy / Pandas
3. Statistics & Probability
4. Data Preprocessing
5. Linear Regression
6. Logistic Regression
7. Decision Trees
8. Random Forest
9. XGBoost / LightGBM
10. Clustering
11. PCA
12. Time Series
13. Neural Networks
14. CNN
15. RNN / LSTM / GRU
16. Transformers
17. LLMs
18. RAG
19. Generative AI
20. Reinforcement Learning
```

---

# 88. Repository Structure

```text
AI-ML-Knowledge-Base/
│
├── README.md
│
├── 01_python/
├── 02_statistics_probability/
├── 03_data_preprocessing/
│
├── 04_supervised_learning/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── decision_trees/
│   ├── random_forest/
│   ├── svm/
│   └── gradient_boosting/
│
├── 05_unsupervised_learning/
│   ├── kmeans/
│   ├── dbscan/
│   ├── hierarchical_clustering/
│   └── pca/
│
├── 06_anomaly_detection/
│
├── 07_time_series/
│   ├── arima/
│   ├── prophet/
│   ├── markov_chains/
│   └── monte_carlo/
│
├── 08_deep_learning/
│   ├── mlp/
│   ├── cnn/
│   ├── rnn/
│   ├── lstm/
│   └── transformers/
│
├── 09_nlp/
├── 10_llms/
├── 11_rag/
│
├── 12_generative_ai/
│   ├── autoencoders/
│   ├── vae/
│   ├── gan/
│   └── diffusion/
│
├── 13_reinforcement_learning/
├── 14_recommender_systems/
├── 15_graph_neural_networks/
│
├── 16_explainable_ai/
│   ├── shap/
│   └── lime/
│
└── projects/
    ├── project_01/
    ├── project_02/
    └── project_03/
```

---

# Practical Repositories

Each theoretical section links to its corresponding practical implementation. Suggested format for each entry:

```markdown
## Random Forest

Teoría → [Notas de Random Forest](./04_supervised_learning/random_forest/README.md)
Implementación → [Repo del proyecto](https://github.com/<username>/<repositorio>)
Dataset → [Dataset en Kaggle](...)
Notebook → [random_forest.ipynb](...)
```

The goal is to connect:

```text
THEORY → CODE → DATASET → EXPERIMENT → RESULTS → EXPLAINABILITY → REAL-WORLD APPLICATION
```

> ✏️ **Pendiente:** ir añadiendo aquí los enlaces reales a cada uno de tus repos a medida que los subas (uno por algoritmo/proyecto, o agrupados por bloque temático).

| Bloque | Teoría | Código | Estado |
|---|---|---|---|
| Regresión Lineal / Logística | [enlace](#) | [repo](#) | 🔲 pendiente |
| Árboles / Random Forest / XGBoost | [enlace](#) | [repo](#) | 🔲 pendiente |
| Clustering (K-Means, DBSCAN, GMM) | [enlace](#) | [repo](#) | 🔲 pendiente |
| Series temporales | [enlace](#) | [repo](#) | 🔲 pendiente |
| Deep Learning (CNN/RNN/LSTM) | [enlace](#) | [repo](#) | 🔲 pendiente |
| Transformers / LLMs / RAG | [enlace](#) | [repo](#) | 🔲 pendiente |
| Generative AI (GAN/VAE/Diffusion) | [enlace](#) | [repo](#) | 🔲 pendiente |
| Reinforcement Learning | [enlace](#) | [repo](#) | 🔲 pendiente |

---

# Perspectiva final

La Inteligencia Artificial no debería entenderse como una colección de algoritmos aislados, sino como una **caja de herramientas**.

```text
Problema de negocio / investigación
            ↓
       Entender los datos
            ↓
     Definir el problema de ML
            ↓
      Seleccionar un baseline
            ↓
         Entrenar
            ↓
        Evaluar
            ↓
        Explicar
            ↓
        Mejorar
            ↓
        Desplegar
            ↓
        Monitorizar
```

**True skill lies not in knowing how to execute each algorithm, but in knowing which approach to use, why it is appropriate, what assumptions it makes, what its limitations are, and how to evaluate whether it actually solves the problem.**

---

<div align="center">

Apuntes creados durante el Máster en Inteligencia Artificial & Programación

</div>

