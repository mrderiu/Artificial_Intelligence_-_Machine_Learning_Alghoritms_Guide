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

[Artificial Intelligence Overview](#artificial-intelligence-overview)
[Choosing an Algorithm](#choosing-an-algorithm)

1. [Symbolic AI](#1-symbolic-ai)
   1.1 [Expert Systems](#11-expert-systems)
   1.2 [Rule-Based Systems](#12-rule-based-systems)
   1.3 [Knowledge Graphs](#13-knowledge-graphs)
   1.4 [Search / Planning](#14-search--planning)

2. [Machine Learning](#2-machine-learning)
   2.1 [Supervised Learning](#21-supervised-learning)
      2.1.1 [Regression](#211-regression)
         2.1.1.1 [Linear Regression](#2111-linear-regression)
         2.1.1.2 [Ridge](#2112-ridge)
         2.1.1.3 [Lasso](#2113-lasso)
         2.1.1.4 [Elastic Net](#2114-elastic-net)
      2.1.2 [Classification](#212-classification)
         2.1.2.1 [Logistic Regression](#2121-logistic-regression)
         2.1.2.2 [Decision Trees](#2122-decision-trees)
         2.1.2.3 [Random Forest](#2123-random-forest)
         2.1.2.4 [Support Vector Machines (SVM)](#2124-support-vector-machines-svm)
         2.1.2.5 [K-Nearest Neighbors (KNN)](#2125-k-nearest-neighbors-knn)
         2.1.2.6 [Naive Bayes](#2126-naive-bayes)
         2.1.2.7 [Linear Discriminant Analysis (LDA)](#2127-linear-discriminant-analysis-lda)
      2.1.3 [Gradient Boosting](#213-gradient-boosting)
         2.1.3.1 [XGBoost](#2131-xgboost)
         2.1.3.2 [LightGBM](#2132-lightgbm)
         2.1.3.3 [CatBoost](#2133-catboost)
   2.2 [Unsupervised Learning](#22-unsupervised-learning)
      2.2.1 [Clustering](#221-clustering)
         2.2.1.1 [K-Means](#2211-k-means)
         2.2.1.2 [Hierarchical Clustering](#2212-hierarchical-clustering)
         2.2.1.3 [DBSCAN](#2213-dbscan)
         2.2.1.4 [Gaussian Mixture Models (GMM)](#2214-gaussian-mixture-models-gmm)
      2.2.2 [Dimensionality Reduction](#222-dimensionality-reduction)
         2.2.2.1 [PCA](#2221-pca)
         2.2.2.2 [t-SNE](#2222-t-sne)
         2.2.2.3 [UMAP](#2223-umap)
      2.2.3 [Anomaly Detection](#223-anomaly-detection)
         2.2.3.1 [Isolation Forest](#2231-isolation-forest)
         2.2.3.2 [One-Class SVM](#2232-one-class-svm)
         2.2.3.3 [Local Outlier Factor (LOF)](#2233-local-outlier-factor-lof)
   2.3 [Semi-Supervised Learning](#23-semi-supervised-learning)
      2.3.1 [Semi-Supervised Algorithms](#231-semi-supervised-algorithms)
      2.3.2 [Transfer Learning](#232-transfer-learning)
      2.3.3 [Fine-Tuning](#233-fine-tuning)
   2.4 [Reinforcement Learning](#24-reinforcement-learning)
      2.4.1 [Q-Learning](#241-q-learning)
      2.4.2 [SARSA](#242-sarsa)
      2.4.3 [Deep Q-Network (DQN)](#243-deep-q-network-dqn)
      2.4.4 [Policy Gradient](#244-policy-gradient)
      2.4.5 [Actor-Critic](#245-actor-critic)
      2.4.6 [Proximal Policy Optimization (PPO)](#246-proximal-policy-optimization-ppo)
   2.5 [Time Series](#25-time-series)
      2.5.1 [ARIMA](#251-arima)
      2.5.2 [SARIMA](#252-sarima)
      2.5.3 [Exponential Smoothing (Holt-Winters)](#253-exponential-smoothing-holt-winters)
      2.5.4 [Prophet](#254-prophet)
      2.5.5 [Vector Autoregression (VAR)](#255-vector-autoregression-var)
      2.5.6 [Markov Chains](#256-markov-chains)
      2.5.7 [Hidden Markov Models (HMM)](#257-hidden-markov-models-hmm)
      2.5.8 [Monte Carlo Simulation](#258-monte-carlo-simulation)
      2.5.9 [Kalman Filter](#259-kalman-filter)
   2.6 [Optimization](#26-optimization)
      2.6.1 [Genetic Algorithms](#261-genetic-algorithms)
      2.6.2 [Particle Swarm Optimization (PSO)](#262-particle-swarm-optimization-pso)
   2.7 [Probabilistic Models](#27-probabilistic-models)
      2.7.1 [Bayesian Networks](#271-bayesian-networks)
      2.7.2 [MCMC (Markov Chain Monte Carlo)](#272-mcmc-markov-chain-monte-carlo)
   2.8 [Recommender Systems](#28-recommender-systems)
      2.8.1 [Collaborative Filtering](#281-collaborative-filtering)
      2.8.2 [Content-Based Filtering](#282-content-based-filtering)
      2.8.3 [Matrix Factorization](#283-matrix-factorization)
   2.9 [Explainable AI](#29-explainable-ai)
      2.9.1 [SHAP](#291-shap)
      2.9.2 [LIME](#292-lime)

3. [Deep Learning](#3-deep-learning)
   3.1 [MLP (MultiLayer Perceptron)](#31-mlp-multilayer-perceptron)
   3.2 [CNN (Convolutional Neural Network)](#32-cnn-convolutional-neural-network)
   3.3 [RNN (Recurrent Neural Network)](#33-rnn-recurrent-neural-network)
   3.4 [LSTM](#34-lstm)
   3.5 [GRU](#35-ru)
   3.6 [Transformers](#36-transformers)
   3.7 [Large Language Models (LLMs)](#37-large-language-models-llms)
   3.8 [Embeddings](#38-embeddings)
   3.9 [RAG (Retrieval-Augmented Generation)](#39-rag-retrieval-augmented-generation)
   3.10 [Autoencoders](#310-autoencoders)
   3.11 [Variational Autoencoders (VAE)](#311-variational-autoencoders-vae)
   3.12 [GANs](#312-gans)
   3.13 [Diffusion Models](#313-diffusion-models)
   3.14 [Graph Neural Networks](#314-graph-neural-networks)

4. [Natural Language Processing](#4-natural-language-processing)
   4.1 [TF-IDF](#41-tf-idf)
   4.2 [Word2Vec](#42-word2vec)
   4.3 [Topic Modeling (LDA)](#43-topic-modeling-lda)

5. [Computer Vision](#5-computer-vision)

6. [Generative AI](#6-generative-ai)

7. [Robotics & Autonomous Systems](#7-robotics--autonomous-systems)

8. [Model Evaluation](#8-model-evaluation)

9. [Algorithm Selection Cheat Sheet](#9-algorithm-selection-cheat-sheet)

10. [Practical Workflow for Tabular Data](#10-practical-workflow-for-tabular-data)

11. [The Most Important Rule](#11-the-most-important-rule)

12. [Algorithm Selection Cheat Sheet](#11-algorithm-selection-cheat-sheet)
    
13. [Tabular Data: Practical Model Selection](#13-tabular-data-practical-model-selection)

---

# Artificial Intelligence Overview

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
│   │   │   ├── Linear Regression
│   │   │   ├── Ridge
│   │   │   ├── Lasso
│   │   │   └── Elastic Net
│   │   ├── Classification
│   │   │   ├── Logistic Regression
│   │   │   ├── Decision Trees
│   │   │   ├── Random Forest
│   │   │   ├── Support Vector Machines (SVM)
│   │   │   ├── K-Nearest Neighbors (KNN)
│   │   │   ├── Naive Bayes
│   │   │   └── Linear Discriminant Analysis (LDA)
│   │   └── Gradient Boosting
│   │       ├── XGBoost
│   │       ├── LightGBM
│   │       └── CatBoost
│   │
│   ├── Unsupervised Learning
│   │   ├── Clustering
│   │   │   ├── K-Means
│   │   │   ├── Hierarchical Clustering
│   │   │   ├── DBSCAN
│   │   │   └── Gaussian Mixture Models (GMM)
│   │   ├── Dimensionality Reduction
│   │   │   ├── PCA
│   │   │   ├── t-SNE
│   │   │   └── UMAP
│   │   └── Anomaly Detection
│   │       ├── Isolation Forest
│   │       ├── One-Class SVM
│   │       └── Local Outlier Factor (LOF)
│   │
│   ├── Semi-Supervised Learning
│   │   ├── Semi-Supervised Algorithms
│   │   ├── Transfer Learning
│   │   └── Fine-Tuning
│   │
│   ├── Reinforcement Learning
│   │   ├── Q-Learning
│   │   ├── SARSA
│   │   ├── Deep Q-Network (DQN)
│   │   ├── Policy Gradient
│   │   ├── Actor-Critic
│   │   └── Proximal Policy Optimization (PPO)
│   │
│   ├── Time Series
│   │   ├── ARIMA
│   │   ├── SARIMA
│   │   ├── Prophet
│   │   ├── VAR
│   │   ├── Markov Chains
│   │   ├── Monte Carlo
│   │   └── Kalman Filter
│   │
│   ├── Optimization
│   │   ├── Genetic Algorithms
│   │   └── Particle Swarm Optimization (PSO)
│   │
│   ├── Probabilistic Models
│   │   ├── Bayesian Networks
│   │   └── MCMC
│   │
│   ├── Recommender Systems
│   │
│   └── Explainable AI
│       ├── SHAP
│       └── LIME
│
├── Deep Learning
│   ├── MLP (MultiLayer Perceptron)
│   ├── CNN (Convolutional Neural Network)
│   ├── RNN (Recurrent Neural Network)
│   ├── LSTM / GRU
│   │   ├── LSTM
│   │   └── GRU
│   ├── Transformers
│   │   ├── Transformers (base architecture)
│   │   ├── Large Language Models (LLMs)
│   │   ├── Embeddings
│   │   └── RAG (Retrieval-Augmented Generation)
│   ├── Autoencoders
│   │   ├── Autoencoders (classic)
│   │   └── Variational Autoencoders (VAE)
│   ├── GANs
│   ├── Diffusion Models
│   └── Graph Neural Networks
│
├── Natural Language Processing
│   ├── TF-IDF
│   ├── Word2Vec
│   └── Topic Modeling
│
├── Computer Vision
│
├── Generative AI
│
└── Robotics & Autonomous Systems
```

---

# Choosing an Algorithm

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

## 1. Symbolic AI

### 1.1 Expert Systems
- **Problem it solves:** encoding explicit human expertise into automated decisions.
- **Why use it:** full transparency — every decision can be traced back to a rule.
- **When to use it:** regulated domains (auditing, compliance, medical protocols) where the logic is already known and stable.
- **When to avoid it:** when the domain knowledge is fuzzy, constantly changing, or too large to encode as rules.
- **Advantages:** explainable, deterministic, easy to audit and modify rule by rule.
- **Limitations:** doesn't scale to complex or ambiguous knowledge; rules must be maintained manually.
- **Alternatives:** Decision Trees (learn rules from data instead of hand-coding them), Fuzzy Logic (for partial truth).
- **How to implement it:** rule engines like Drools, CLIPS, or a simple `if/else` / rules-as-data structure in Python.

### 1.2 Rule-Based Systems
- **Problem it solves:** automating decisions that follow a fixed, known logic.
- **Why use it:** predictable, fast, and requires no training data.
- **When to use it:** validation logic, alerting systems, business rule engines.
- **When to avoid it:** when patterns are learned better from data than defined by hand, or rules would become unmanageably numerous.
- **Advantages:** simple, fast, fully interpretable.
- **Limitations:** brittle — doesn't generalize beyond the rules defined; hard to maintain at scale.
- **Alternatives:** Decision Trees, Expert Systems.
- **How to implement it:** conditional logic in code, or a rules engine (Drools, business rule management systems).

### 1.3 Knowledge Graphs
- **Problem it solves:** representing entities and their relationships explicitly, enabling reasoning over connections.
- **Why use it:** captures relational structure that flat tables can't.
- **When to use it:** semantic search, recommendation, RAG pipelines, knowledge management.
- **When to avoid it:** when relationships between entities aren't important to the problem, or maintaining the graph is too costly.
- **Advantages:** rich relational reasoning, reusable across many downstream tasks.
- **Limitations:** costly to build and keep updated; querying can be complex.
- **Alternatives:** relational databases, vector embeddings.
- **How to implement it:** graph databases such as Neo4j, or RDF/SPARQL stacks.

### 1.4 Search / Planning
- **Problem it solves:** finding a sequence of actions from an initial state to a goal state.
- **Why use it:** guarantees an optimal or near-optimal path when the state space is well defined.
- **When to use it:** routing, logistics, puzzle solving, game AI.
- **When to avoid it:** when the state space is too large or continuous without good heuristics.
- **Advantages:** provably correct/optimal under the right conditions; well-understood theory.
- **Limitations:** computationally expensive for large state spaces; needs a good heuristic function.
- **Alternatives:** Reinforcement Learning (when the environment is too complex to search explicitly).
- **How to implement it:** libraries like `networkx` (graph search), custom A* implementations, or planning frameworks (PDDL solvers).

---

## 2. Machine Learning

### 2.1 Supervised Learning

#### 2.1.1 Regression

##### 2.1.1.1 Linear Regression
- **Problem it solves:** predicting a continuous numerical value from input variables.
- **Why use it:** it's fast, transparent, and gives a strong baseline before trying anything more complex.
- **When to use it:** target is numerical, relationships are roughly linear, interpretability matters.
- **When to avoid it:** when relationships are strongly nonlinear or there's heavy multicollinearity you haven't addressed.
- **Advantages:** extremely fast, easy to interpret, works with small datasets.
- **Limitations:** assumes linearity, sensitive to outliers and multicollinearity.
- **Alternatives:** Ridge/Lasso/Elastic Net (regularized versions), Random Forest/XGBoost (nonlinear).
- **How to implement it:** `sklearn.linear_model.LinearRegression`.

##### 2.1.1.2 Ridge
- **Problem it solves:** regression with many correlated features, where plain linear regression overfits.
- **Why use it:** stabilizes coefficients without removing any feature.
- **When to use it:** multicollinearity is present and you want to keep all predictors.
- **When to avoid it:** when you actually need feature selection (use Lasso instead).
- **Advantages:** reduces variance, more stable than OLS with correlated features.
- **Limitations:** doesn't perform feature selection; coefficients rarely reach exactly zero.
- **Alternatives:** Lasso, Elastic Net.
- **How to implement it:** `sklearn.linear_model.Ridge`.

##### 2.1.1.3 Lasso
- **Problem it solves:** regression with many potentially irrelevant features.
- **Why use it:** performs automatic feature selection by zeroing out unimportant coefficients.
- **When to use it:** you suspect many features are irrelevant and want a sparser, simpler model.
- **When to avoid it:** with highly correlated features, where it tends to pick one arbitrarily and drop the rest (Elastic Net handles this better).
- **Advantages:** built-in feature selection, simpler resulting model.
- **Limitations:** unstable when features are highly correlated.
- **Alternatives:** Ridge, Elastic Net.
- **How to implement it:** `sklearn.linear_model.Lasso`.

##### 2.1.1.4 Elastic Net
- **Problem it solves:** regression with correlated predictors where you also want feature selection.
- **Why use it:** balances Ridge's stability with Lasso's sparsity.
- **When to use it:** many correlated predictors and Lasso alone is unstable.
- **When to avoid it:** simple, low-dimensional problems where plain Linear Regression is enough.
- **Advantages:** more robust than pure Lasso with correlated features.
- **Limitations:** two hyperparameters to tune instead of one (L1/L2 ratio and regularization strength).
- **Alternatives:** Ridge, Lasso.
- **How to implement it:** `sklearn.linear_model.ElasticNet`.

#### 2.1.2 Classification

##### 2.1.2.1 Logistic Regression
- **Problem it solves:** binary (or multiclass) classification with probability estimates.
- **Why use it:** simple, fast, interpretable, and outputs calibrated-ish probabilities.
- **When to use it:** need a classification baseline, interpretability matters, relationships are relatively simple.
- **When to avoid it:** the decision boundary is highly nonlinear or involves complex feature interactions.
- **Advantages:** fast, interpretable, works well on high-dimensional sparse data (e.g., text).
- **Limitations:** linear decision boundary; struggles with complex nonlinear patterns.
- **Alternatives:** Decision Trees, Random Forest, SVM (for nonlinear boundaries via kernels).
- **How to implement it:** `sklearn.linear_model.LogisticRegression`.

##### 2.1.2.2 Decision Trees
- **Problem it solves:** classification or regression with nonlinear relationships and feature interactions.
- **Why use it:** intuitive, visualizable, handles mixed numerical/categorical data without scaling.
- **When to use it:** need an interpretable model that captures nonlinear patterns and interactions.
- **When to avoid it:** when a single tree's tendency to overfit is unacceptable — use an ensemble instead.
- **Advantages:** intuitive, no feature scaling needed, captures interactions naturally.
- **Limitations:** prone to overfitting; unstable (small data changes can produce very different trees).
- **Alternatives:** Random Forest, Gradient Boosting (both fix the overfitting problem via ensembling).
- **How to implement it:** `sklearn.tree.DecisionTreeClassifier` / `DecisionTreeRegressor`.

##### 2.1.2.3 Random Forest
- **Problem it solves:** robust classification/regression on tabular data without heavy tuning.
- **Why use it:** strong out-of-the-box performance with far less overfitting than a single tree.
- **When to use it:** nonlinear tabular problems where you want a reliable model with minimal tuning.
- **When to avoid it:** very large datasets where training/memory cost is prohibitive, or when Gradient Boosting's extra accuracy is worth the tuning effort.
- **Advantages:** robust, handles nonlinearity, estimates feature importance, little preprocessing required.
- **Limitations:** large forests can be memory-heavy; less interpretable than one tree; often beaten by Gradient Boosting on structured data.
- **Alternatives:** XGBoost / LightGBM / CatBoost (usually higher accuracy), Decision Trees (more interpretable).
- **How to implement it:** `sklearn.ensemble.RandomForestClassifier` / `RandomForestRegressor`.

##### 2.1.2.4 Support Vector Machines (SVM)
- **Problem it solves:** classification (or regression) with a clear margin of separation between classes.
- **Why use it:** effective in high-dimensional spaces, and kernels let it capture nonlinear boundaries.
- **When to use it:** small-to-medium, high-dimensional datasets with reasonably separable classes.
- **When to avoid it:** very large datasets (training time grows fast) or when features aren't scaled.
- **Advantages:** strong theoretical foundation, effective in high dimensions, flexible via kernels.
- **Limitations:** expensive to train at scale, requires feature scaling, sensitive to hyperparameter choice.
- **Alternatives:** Logistic Regression (linear, faster), Random Forest / Gradient Boosting (large tabular data).
- **How to implement it:** `sklearn.svm.SVC` / `SVR`.

##### 2.1.2.5 K-Nearest Neighbors (KNN)
- **Problem it solves:** classification/regression where similar inputs should have similar outputs.
- **Why use it:** extremely simple, no training phase, naturally nonlinear.
- **When to use it:** small datasets where local similarity is meaningful and the decision boundary is irregular.
- **When to avoid it:** large datasets (slow prediction) or high-dimensional data (curse of dimensionality).
- **Advantages:** simple, no explicit training, naturally captures nonlinear boundaries.
- **Limitations:** slow at prediction time on large data, sensitive to feature scaling, degrades in high dimensions.
- **Alternatives:** Decision Trees, SVM, clustering-based approaches.
- **How to implement it:** `sklearn.neighbors.KNeighborsClassifier` / `KNeighborsRegressor`.

##### 2.1.2.6 Naive Bayes
- **Problem it solves:** fast probabilistic classification, especially for text.
- **Why use it:** extremely fast to train and predict, works well even with limited data.
- **When to use it:** spam detection, sentiment analysis, document classification, quick baselines for NLP.
- **When to avoid it:** when features are strongly correlated (violates the independence assumption badly) and accuracy is critical.
- **Advantages:** fast, simple, strong text-classification baseline, works with small datasets.
- **Limitations:** the feature-independence assumption is often unrealistic.
- **Alternatives:** Logistic Regression, TF-IDF + Logistic Regression, Transformer-based text classifiers.
- **How to implement it:** `sklearn.naive_bayes.MultinomialNB` / `GaussianNB` / `BernoulliNB`.

##### 2.1.2.7 Linear Discriminant Analysis (LDA)
- **Problem it solves:** classification and dimensionality reduction when classes are linearly separable.
- **Why use it:** simple, fast, and doubles as a dimensionality-reduction technique.
- **When to use it:** classes are reasonably well separated and Gaussian-ish assumptions hold.
- **When to avoid it:** classes overlap heavily or the covariance structure differs strongly between classes.
- **Advantages:** fast, interpretable, useful for both classification and dimensionality reduction.
- **Limitations:** relies on statistical assumptions (normality, equal covariance) that often don't hold.
- **Alternatives:** Logistic Regression, PCA (for reduction only, unsupervised).
- **How to implement it:** `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.

#### 2.1.3 Gradient Boosting

##### 2.1.3.1 XGBoost
- **Problem it solves:** high-accuracy classification/regression on structured/tabular data.
- **Why use it:** consistently among the best-performing algorithms for tabular problems.
- **When to use it:** tabular data, nonlinear relationships, feature interactions matter, missing values may exist.
- **When to avoid it:** you need maximum interpretability with zero tuning effort, or your data is unstructured (images, raw text, audio).
- **Advantages:** excellent accuracy, handles nonlinearity and missing values, built-in regularization, pairs well with SHAP.
- **Limitations:** many hyperparameters, can overfit if misconfigured, training cost grows on very large data.
- **Alternatives:** LightGBM (faster on huge data), CatBoost (better with categorical features), Random Forest (simpler, less tuning).
- **How to implement it:** `xgboost` Python package (`xgboost.XGBClassifier` / `XGBRegressor`).

##### 2.1.3.2 LightGBM
- **Problem it solves:** gradient boosting at scale — very large datasets or many features.
- **Why use it:** much faster and more memory-efficient than classic gradient boosting.
- **When to use it:** large-scale tabular problems where training speed matters.
- **When to avoid it:** small datasets, where its leaf-wise growth can overfit.
- **Advantages:** very fast, memory-efficient, strong accuracy.
- **Limitations:** leaf-wise growth can overfit small datasets without careful tuning.
- **Alternatives:** XGBoost, CatBoost.
- **How to implement it:** `lightgbm` Python package (`lightgbm.LGBMClassifier` / `LGBMRegressor`).

##### 2.1.3.3 CatBoost
- **Problem it solves:** gradient boosting on data with many categorical variables.
- **Why use it:** handles categorical features natively, reducing preprocessing work.
- **When to use it:** datasets with many (especially high-cardinality) categorical features.
- **When to avoid it:** purely numerical data where XGBoost/LightGBM already perform well and are more widely supported.
- **Advantages:** minimal preprocessing for categoricals, strong default performance.
- **Limitations:** can be slower to train than LightGBM on very large numerical datasets.
- **Alternatives:** XGBoost, LightGBM.
- **How to implement it:** `catboost` Python package (`catboost.CatBoostClassifier` / `CatBoostRegressor`).

---

### 2.2 Unsupervised Learning

#### 2.2.1 Clustering

##### 2.2.1.1 K-Means
- **Problem it solves:** partitioning data into K groups based on similarity.
- **Why use it:** fast, simple, scales well to large datasets.
- **When to use it:** the number of clusters can be estimated and clusters are roughly spherical (customer segmentation).
- **When to avoid it:** clusters have irregular shapes or very different densities, or you don't know K.
- **Advantages:** fast, simple, scalable.
- **Limitations:** requires specifying K, sensitive to outliers and feature scaling, assumes spherical clusters.
- **Alternatives:** DBSCAN (irregular shapes), Hierarchical Clustering (unknown K), GMM (soft assignment).
- **How to implement it:** `sklearn.cluster.KMeans`.

##### 2.2.1.2 Hierarchical Clustering
- **Problem it solves:** understanding nested/hierarchical relationships between groups.
- **Why use it:** doesn't require specifying the number of clusters upfront; produces an interpretable dendrogram.
- **When to use it:** small-to-medium datasets where the relationship between clusters matters.
- **When to avoid it:** large datasets (computationally expensive, roughly O(n²) or worse).
- **Advantages:** no need to fix K in advance, visual and interpretable via dendrograms.
- **Limitations:** doesn't scale well to large datasets; sensitive to the linkage/distance metric chosen.
- **Alternatives:** K-Means (faster at scale), DBSCAN.
- **How to implement it:** `scipy.cluster.hierarchy` or `sklearn.cluster.AgglomerativeClustering`.

##### 2.2.1.3 DBSCAN
- **Problem it solves:** clustering with irregular shapes and automatic noise/outlier detection.
- **Why use it:** finds arbitrarily shaped clusters and doesn't require specifying the number of clusters.
- **When to use it:** geospatial/mobility data, or when clusters aren't spherical and noise detection is useful.
- **When to avoid it:** clusters have very different densities (DBSCAN struggles with this) or choosing `eps`/`min_samples` is too hard.
- **Advantages:** no need to specify cluster count, identifies noise, handles irregular shapes.
- **Limitations:** sensitive to `eps` and `min_samples`, struggles with varying-density clusters.
- **Alternatives:** HDBSCAN (handles varying density), K-Means, GMM.
- **How to implement it:** `sklearn.cluster.DBSCAN`.

##### 2.2.1.4 Gaussian Mixture Models (GMM)
- **Problem it solves:** clustering with overlapping groups where a probabilistic membership is more informative than a hard label.
- **Why use it:** gives a probability of belonging to each cluster instead of a single hard assignment.
- **When to use it:** cluster boundaries genuinely overlap and probabilistic membership is useful.
- **When to avoid it:** clusters are clearly separated and simplicity/speed matters more (K-Means is enough).
- **Advantages:** soft/probabilistic clustering, models elliptical (not just spherical) clusters.
- **Limitations:** can be sensitive to initialization, assumes Gaussian-shaped clusters.
- **Alternatives:** K-Means, DBSCAN.
- **How to implement it:** `sklearn.mixture.GaussianMixture`.

#### 2.2.2 Dimensionality Reduction

##### 2.2.2.1 PCA
- **Problem it solves:** reducing the number of variables while retaining as much variance as possible.
- **Why use it:** speeds up downstream models, reduces noise, helps visualize high-dimensional data.
- **When to use it:** many correlated numerical variables, or as a preprocessing step before another model.
- **When to avoid it:** you need interpretable features (principal components are linear combinations, hard to interpret) or relationships are strongly nonlinear.
- **Advantages:** fast, well-understood, reduces noise and correlated redundancy.
- **Limitations:** components are hard to interpret; only captures linear structure.
- **Alternatives:** UMAP / t-SNE (nonlinear, mainly for visualization), Autoencoders (nonlinear, learned).
- **How to implement it:** `sklearn.decomposition.PCA`.

##### 2.2.2.2 t-SNE
- **Problem it solves:** visualizing high-dimensional data in 2D/3D while preserving local neighborhoods.
- **Why use it:** reveals cluster structure visually better than linear methods like PCA.
- **When to use it:** exploring embeddings or neural network representations visually.
- **When to avoid it:** as a preprocessing step for a predictive model, or with very large datasets (it's slow).
- **Advantages:** captures nonlinear local structure, great for visual exploration.
- **Limitations:** slow, not meant for downstream prediction, distances between distant clusters aren't meaningful.
- **Alternatives:** UMAP (faster, better global structure), PCA (linear, faster).
- **How to implement it:** `sklearn.manifold.TSNE`.

##### 2.2.2.3 UMAP
- **Problem it solves:** nonlinear dimensionality reduction and visualization at larger scale than t-SNE.
- **Why use it:** faster than t-SNE and preserves more global structure.
- **When to use it:** visualizing embeddings, clustering pipelines, high-dimensional biological or NLP data.
- **When to avoid it:** when you need a simple, linear, easily explainable reduction (use PCA).
- **Advantages:** fast, scalable, preserves both local and some global structure.
- **Limitations:** results can vary with hyperparameters; less established theoretically than PCA.
- **Alternatives:** t-SNE, PCA.
- **How to implement it:** `umap-learn` Python package (`umap.UMAP`).

#### 2.2.3 Anomaly Detection

##### 2.2.3.1 Isolation Forest
- **Problem it solves:** detecting anomalies in tabular data without labeled examples.
- **Why use it:** simple, fast, and works well as a general-purpose anomaly detector.
- **When to use it:** no anomaly labels are available and the dataset is tabular and reasonably large.
- **When to avoid it:** anomalies are defined relative to local density rather than global isolation (use LOF instead).
- **Advantages:** fast, scalable, doesn't require labeled anomalies, handles high dimensions reasonably well.
- **Limitations:** can struggle with very subtle or local anomalies.
- **Alternatives:** One-Class SVM, Local Outlier Factor (LOF), Autoencoders.
- **How to implement it:** `sklearn.ensemble.IsolationForest`.

##### 2.2.3.2 One-Class SVM
- **Problem it solves:** learning the boundary of "normal" data to flag anything outside it.
- **Why use it:** effective in small/medium datasets where a clear "normal" region can be learned.
- **When to use it:** small-to-medium datasets with a well-defined notion of normal behavior.
- **When to avoid it:** large datasets (computationally expensive).
- **Advantages:** solid theoretical foundation, flexible via kernels.
- **Limitations:** expensive on large data, sensitive to hyperparameters.
- **Alternatives:** Isolation Forest (faster, scales better), LOF.
- **How to implement it:** `sklearn.svm.OneClassSVM`.

##### 2.2.3.3 Local Outlier Factor (LOF)
- **Problem it solves:** detecting anomalies that are only unusual relative to their local neighborhood.
- **Why use it:** captures anomalies that a global method (like Isolation Forest) might miss.
- **When to use it:** anomalies are context-dependent (unusual in their local area but not globally).
- **When to avoid it:** you need to score new/unseen points efficiently at scale (LOF is mainly designed for the training set).
- **Advantages:** captures local density variation well.
- **Limitations:** computationally heavier, less suited to very large or streaming data.
- **Alternatives:** Isolation Forest, One-Class SVM.
- **How to implement it:** `sklearn.neighbors.LocalOutlierFactor`.

---

### 2.3 Semi-Supervised Learning

##### 2.3.1 Semi-Supervised Algorithms
- **Problem it solves:** learning when only a small fraction of the data is labeled.
- **Why use it:** makes use of abundant unlabeled data to improve a model trained on limited labels.
- **When to use it:** labeling is expensive/slow but unlabeled data is plentiful.
- **When to avoid it:** you already have enough labeled data, or unlabeled data doesn't share the same distribution.
- **Advantages:** better performance than pure supervised learning under label scarcity.
- **Limitations:** can propagate errors if pseudo-labels are wrong; sensitive to distribution mismatch.
- **Alternatives:** Transfer Learning, active learning (selectively labeling the most useful examples).
- **How to implement it:** `sklearn.semi_supervised` (`SelfTrainingClassifier`, `LabelPropagation`).

##### 2.3.2 Transfer Learning
- **Problem it solves:** building a strong model without training from scratch, using knowledge from a related task.
- **Why use it:** dramatically reduces the data and compute needed.
- **When to use it:** you have a pretrained model in a related domain and limited task-specific data.
- **When to avoid it:** the target task is very different from the source domain (negative transfer risk).
- **Advantages:** faster training, better performance with limited data.
- **Limitations:** performance depends heavily on how related source and target tasks are.
- **Alternatives:** training from scratch (if enough data/compute), Fine-Tuning.
- **How to implement it:** pretrained models via `torchvision.models`, `transformers` (Hugging Face), freezing/unfreezing layers as needed.

##### 2.3.3 Fine-Tuning
- **Problem it solves:** adapting a pretrained model to a specific task, style, or domain.
- **Why use it:** cheaper and faster than training a large model from scratch.
- **When to use it:** the model needs to learn task-specific behavior beyond what pretraining covers.
- **When to avoid it:** the task-specific dataset is tiny and risks overfitting or catastrophic forgetting.
- **Advantages:** efficient, leverages large pretrained models, works well even with modest data.
- **Limitations:** risk of catastrophic forgetting, still requires compute for large models.
- **Alternatives:** Prompt engineering / in-context learning (for LLMs, no weight updates needed), Transfer Learning with frozen layers.
- **How to implement it:** `transformers` (Hugging Face `Trainer`), parameter-efficient methods like LoRA/QLoRA via `peft`.

---

### 2.4 Reinforcement Learning

##### 2.4.1 Q-Learning
- **Problem it solves:** learning optimal actions in an environment through trial and error, with discrete states/actions.
- **Why use it:** simple, model-free, guaranteed to converge under standard conditions.
- **When to use it:** small, discrete state/action spaces.
- **When to avoid it:** large or continuous state spaces (a Q-table becomes infeasible).
- **Advantages:** simple, well understood, no model of the environment needed.
- **Limitations:** doesn't scale to large state spaces.
- **Alternatives:** DQN (neural network instead of a table), SARSA.
- **How to implement it:** custom Q-table implementation, or `gymnasium` environments with a simple Q-learning loop.

##### 2.4.2 SARSA
- **Problem it solves:** same as Q-Learning, but learning the value of the policy actually being followed.
- **Why use it:** more conservative/safer learning since it accounts for the exploration policy's actual behavior.
- **When to use it:** environments where the cost of exploratory mistakes is high and on-policy learning is preferred.
- **When to avoid it:** you want the fastest possible convergence to the optimal policy (Q-Learning is off-policy and often converges faster).
- **Advantages:** accounts for the actual exploration strategy, can be safer in risky environments.
- **Limitations:** slower to reach the truly optimal policy than off-policy methods.
- **Alternatives:** Q-Learning, DQN.
- **How to implement it:** custom implementation similar to Q-Learning, updating with the actually-taken next action.

##### 2.4.3 Deep Q-Network (DQN)
- **Problem it solves:** reinforcement learning in large or high-dimensional state spaces (e.g., pixels).
- **Why use it:** replaces the Q-table with a neural network, scaling RL to complex environments.
- **When to use it:** state spaces too large for tabular methods (video games, robotics with sensor input).
- **When to avoid it:** simple, small environments where a Q-table is simpler and sufficient.
- **Advantages:** scales to large/continuous state spaces, learns useful representations directly from raw input.
- **Limitations:** unstable training, needs tricks (experience replay, target networks) to converge.
- **Alternatives:** Policy Gradient methods, Actor-Critic, PPO.
- **How to implement it:** `stable-baselines3` (`DQN`), or a custom PyTorch/TensorFlow implementation.

##### 2.4.4 Policy Gradient
- **Problem it solves:** learning a policy directly, especially useful for continuous action spaces.
- **Why use it:** works naturally where action spaces are continuous or very large.
- **When to use it:** continuous control problems (robotics, physical simulations).
- **When to avoid it:** simple discrete-action problems where value-based methods (DQN) are simpler and more sample-efficient.
- **Advantages:** handles continuous actions naturally, can learn stochastic policies.
- **Limitations:** high variance in gradient estimates, sample-inefficient.
- **Alternatives:** Actor-Critic (reduces variance), PPO.
- **How to implement it:** `stable-baselines3` (`A2C`, `PPO` build on this idea), or custom REINFORCE implementation.

##### 2.4.5 Actor-Critic
- **Problem it solves:** combining the strengths of value-based and policy-based RL to reduce variance and improve stability.
- **Why use it:** more stable and sample-efficient training than pure policy gradient methods.
- **When to use it:** as the foundation for most modern deep RL algorithms (A2C, A3C, PPO, SAC, DDPG).
- **When to avoid it:** very simple problems where a basic Q-Learning table would suffice.
- **Advantages:** more stable than pure Policy Gradient, works for both discrete and continuous actions.
- **Limitations:** more complex to implement and tune (two networks instead of one).
- **Alternatives:** PPO, SAC, DDPG (all Actor-Critic variants).
- **How to implement it:** `stable-baselines3` (`A2C`, `SAC`, `DDPG`).

##### 2.4.6 Proximal Policy Optimization (PPO)
- **Problem it solves:** stable, reliable policy optimization across a wide range of RL problems.
- **Why use it:** one of the most robust and widely used modern RL algorithms, works well "out of the box."
- **When to use it:** as a strong default choice for most RL tasks, from games to robotics.
- **When to avoid it:** extremely simple environments where a Q-table is overkill, or when sample efficiency is more critical than stability (SAC may be better for continuous control).
- **Advantages:** stable training, good default hyperparameters, works across many environment types.
- **Limitations:** still requires substantial compute and tuning for hard problems.
- **Alternatives:** SAC, DDPG, A2C.
- **How to implement it:** `stable-baselines3` (`PPO`).

---

### 2.5 Time Series

##### 2.5.1 ARIMA
- **Problem it solves:** forecasting a univariate time series with autocorrelation.
- **Why use it:** simple, interpretable, well-established statistical approach.
- **When to use it:** relationships are mostly linear, no strong seasonality, dataset isn't huge.
- **When to avoid it:** strong seasonality (use SARIMA), or highly nonlinear/complex patterns (use ML/DL).
- **Advantages:** interpretable, fast, strong statistical foundation.
- **Limitations:** assumes linear relationships, requires careful stationarity handling and parameter tuning (p, d, q).
- **Alternatives:** SARIMA (seasonality), Exponential Smoothing, Prophet, LSTM/Transformer (nonlinear/complex data).
- **How to implement it:** `statsmodels.tsa.arima.model.ARIMA`.

##### 2.5.2 SARIMA
- **Problem it solves:** forecasting seasonal time series.
- **Why use it:** extends ARIMA to explicitly model recurring seasonal patterns.
- **When to use it:** clear seasonal patterns exist (daily, weekly, yearly cycles).
- **When to avoid it:** non-seasonal data (plain ARIMA is simpler) or multiple overlapping seasonalities (consider Prophet).
- **Advantages:** captures seasonality explicitly, interpretable.
- **Limitations:** more parameters to tune than ARIMA, still assumes mostly linear dynamics.
- **Alternatives:** Prophet, Exponential Smoothing, ARIMA.
- **How to implement it:** `statsmodels.tsa.statespace.sarimax.SARIMAX`.

##### 2.5.3 Exponential Smoothing (Holt-Winters)
- **Problem it solves:** forecasting stable series with trend and/or seasonality.
- **Why use it:** simple, fast, and effective for well-behaved series.
- **When to use it:** series is relatively stable with clear trend/seasonal components.
- **When to avoid it:** series has irregular, non-repeating patterns or strong external drivers.
- **Advantages:** simple, fast, few parameters, robust for many business series.
- **Limitations:** limited ability to capture complex or irregular dynamics.
- **Alternatives:** SARIMA, Prophet.
- **How to implement it:** `statsmodels.tsa.holtwinters.ExponentialSmoothing`.

##### 2.5.4 Prophet
- **Problem it solves:** business forecasting with trend, seasonality, and calendar effects (holidays, events).
- **Why use it:** designed to be easy to use and robust to missing data and outliers, with built-in handling of holidays.
- **When to use it:** business forecasting where calendar effects matter and ease of use is a priority.
- **When to avoid it:** when you need fine control over the statistical model, or the series has no meaningful seasonality/trend structure.
- **Advantages:** easy to use, handles holidays/events, robust to missing data.
- **Limitations:** less flexible than custom statistical or ML models for unusual patterns.
- **Alternatives:** SARIMA, Exponential Smoothing, ML-based forecasting (XGBoost with time features).
- **How to implement it:** `prophet` Python package (`Prophet().fit()`).

##### 2.5.5 Vector Autoregression (VAR)
- **Problem it solves:** forecasting multiple interdependent time series simultaneously.
- **Why use it:** captures how variables influence each other over time, not just their own history.
- **When to use it:** several related series (traffic, speed, weather) that affect one another.
- **When to avoid it:** a single series with no meaningful cross-variable relationships (use ARIMA/SARIMA instead).
- **Advantages:** models interdependencies between series, useful for multivariate forecasting.
- **Limitations:** assumes linear relationships, number of parameters grows quickly with more series.
- **Alternatives:** Multivariate LSTM/Transformer models, separate univariate models per series.
- **How to implement it:** `statsmodels.tsa.api.VAR`.

##### 2.5.6 Markov Chains
- **Problem it solves:** modeling transitions between discrete states over time.
- **Why use it:** simple and interpretable way to reason about probabilistic future states.
- **When to use it:** the problem has natural discrete states and transitions matter (traffic states, customer journeys).
- **When to avoid it:** the true state isn't directly observable (use Hidden Markov Models instead).
- **Advantages:** simple, interpretable, good for probabilistic scenario analysis.
- **Limitations:** assumes the next state depends only on the current one (Markov property), which may be too simplistic.
- **Alternatives:** Hidden Markov Models, Monte Carlo simulation.
- **How to implement it:** custom transition-matrix implementation with NumPy, or `pomegranate` / `hmmlearn`.

##### 2.5.7 Hidden Markov Models (HMM)
- **Problem it solves:** modeling systems where the true state is hidden and only indirect observations are available.
- **Why use it:** captures the structure of sequential data with unobserved underlying states.
- **When to use it:** speech recognition, activity recognition, biological sequences, regime detection.
- **When to avoid it:** when states are directly observable (use a plain Markov Chain), or relationships are too complex for the HMM's assumptions.
- **Advantages:** principled framework for hidden-state sequential problems.
- **Limitations:** assumes discrete hidden states with Markov dynamics; can struggle with very complex sequences.
- **Alternatives:** RNN/LSTM (learn representations directly without explicit state assumptions).
- **How to implement it:** `hmmlearn` Python package.

##### 2.5.8 Monte Carlo Simulation
- **Problem it solves:** estimating a distribution of possible future outcomes under uncertainty.
- **Why use it:** doesn't require an analytical solution — just repeated random sampling.
- **When to use it:** uncertainty matters and you want a distribution of outcomes, not a single point prediction (risk, finance, project planning).
- **When to avoid it:** you need a single deterministic forecast and computation budget for many simulations is limited.
- **Advantages:** flexible, works for very complex or analytically intractable problems, quantifies uncertainty directly.
- **Limitations:** computationally expensive at high precision; results depend on the quality of the underlying model/assumptions.
- **Alternatives:** analytical probability models (when tractable), Bayesian methods.
- **How to implement it:** custom simulation loops in NumPy/Python, or specialized libraries depending on domain (e.g., `PyMC` for Bayesian Monte Carlo).

##### 2.5.9 Kalman Filter
- **Problem it solves:** estimating the hidden state of a dynamic system from noisy sensor measurements.
- **Why use it:** optimal (under linear-Gaussian assumptions) real-time state estimation.
- **When to use it:** GPS, robotics, navigation, sensor fusion, tracking.
- **When to avoid it:** the system dynamics are strongly nonlinear and non-Gaussian (consider Extended/Unscented Kalman Filter or particle filters).
- **Advantages:** efficient, real-time, well-understood theory, optimal under its assumptions.
- **Limitations:** assumes linear dynamics and Gaussian noise (standard form); needs extensions for nonlinear systems.
- **Alternatives:** Extended Kalman Filter, Particle Filters, HMM.
- **How to implement it:** `filterpy` Python package (`KalmanFilter`), or `pykalman`.

---

### 2.6 Optimization

##### 2.6.1 Genetic Algorithms
- **Problem it solves:** optimization problems with no usable gradient, complex or discrete search spaces.
- **Why use it:** flexible — works on almost any objective function, even non-differentiable ones.
- **When to use it:** combinatorial optimization, scheduling, design problems where gradients aren't available.
- **When to avoid it:** the problem is differentiable and gradient-based methods (much faster) apply.
- **Advantages:** doesn't require gradients, can escape local optima, very flexible.
- **Limitations:** computationally expensive, no convergence guarantees, many hyperparameters (population size, mutation rate).
- **Alternatives:** Particle Swarm Optimization, gradient-based optimization (when applicable), simulated annealing.
- **How to implement it:** `DEAP` Python package, or `scipy.optimize.differential_evolution`.

##### 2.6.2 Particle Swarm Optimization (PSO)
- **Problem it solves:** optimization over complex search spaces using a population of candidate solutions.
- **Why use it:** simple to implement and often converges faster than Genetic Algorithms on continuous problems.
- **When to use it:** continuous optimization problems with complex, multi-modal search spaces.
- **When to avoid it:** the problem has a usable gradient (gradient descent will be faster and more precise).
- **Advantages:** simple, few parameters, works well on continuous optimization.
- **Limitations:** can converge prematurely to local optima; less suited to discrete/combinatorial problems than Genetic Algorithms.
- **Alternatives:** Genetic Algorithms, gradient-based methods.
- **How to implement it:** `pyswarm` or `pyswarms` Python packages.

---

### 2.7 Probabilistic Models

##### 2.7.1 Bayesian Networks
- **Problem it solves:** modeling explicit probabilistic and causal/conditional relationships between variables.
- **Why use it:** makes uncertainty and dependencies between variables explicit and reasoned about formally.
- **When to use it:** domains where causal/conditional structure is known or important to reason about (medical diagnosis, risk modeling).
- **When to avoid it:** the causal structure is unknown and hard to specify, or you just need predictive accuracy without interpretability.
- **Advantages:** explicit representation of uncertainty and dependencies, supports causal reasoning.
- **Limitations:** building the network structure can be hard; inference can be computationally expensive in large networks.
- **Alternatives:** Logistic Regression (simpler predictive model), MCMC (for inference in complex probabilistic models).
- **How to implement it:** `pgmpy` Python package.

##### 2.7.2 MCMC (Markov Chain Monte Carlo)
- **Problem it solves:** sampling from complex probability distributions with no closed-form solution.
- **Why use it:** enables Bayesian inference even when the posterior distribution can't be computed analytically.
- **When to use it:** Bayesian statistics, complex hierarchical models.
- **When to avoid it:** simple problems where an analytical solution or a faster approximate method (variational inference) is available.
- **Advantages:** general-purpose, works for very complex distributions.
- **Limitations:** computationally expensive, convergence can be slow and hard to diagnose.
- **Alternatives:** Variational Inference (faster, approximate).
- **How to implement it:** `PyMC` or `Stan` (via `cmdstanpy`).

---

### 2.8 Recommender Systems

##### 2.8.1 Collaborative Filtering
- **Problem it solves:** recommending items based on patterns across many users' behavior.
- **Why use it:** doesn't require knowing item content — just interaction data.
- **When to use it:** enough historical user-item interactions exist.
- **When to avoid it:** new users/items with no interaction history (the "cold start" problem).
- **Advantages:** captures patterns humans might miss, works well with rich interaction data.
- **Limitations:** cold-start problem for new users/items, can be computationally heavy at scale.
- **Alternatives:** Content-Based Filtering (handles cold start better), hybrid approaches.
- **How to implement it:** `surprise` Python package, or matrix factorization libraries (`implicit`).

##### 2.8.2 Content-Based Filtering
- **Problem it solves:** recommending items similar to what a user already likes, based on item attributes.
- **Why use it:** works even without much user interaction history.
- **When to use it:** item metadata is rich and user history is limited.
- **When to avoid it:** item attributes don't capture what actually drives user preference (Collaborative Filtering may work better).
- **Advantages:** no cold-start problem for new users with some stated preferences, interpretable recommendations.
- **Limitations:** tends to over-recommend similar items (limited diversity/serendipity).
- **Alternatives:** Collaborative Filtering, hybrid recommender systems.
- **How to implement it:** custom similarity-based approach (TF-IDF + cosine similarity), or `sklearn` feature-based similarity.

##### 2.8.3 Matrix Factorization
- **Problem it solves:** large-scale recommendation by learning latent factors for users and items.
- **Why use it:** scales well and often outperforms naive collaborative filtering on sparse interaction matrices.
- **When to use it:** large-scale recommender systems with a sparse user-item interaction matrix.
- **When to avoid it:** very small datasets where simpler similarity-based methods are sufficient.
- **Advantages:** scalable, captures latent structure, strong baseline for large-scale recommendation.
- **Limitations:** latent factors are hard to interpret; still faces cold-start issues.
- **Alternatives:** Deep learning-based recommenders, Collaborative Filtering.
- **How to implement it:** `implicit` Python package (ALS), or `surprise` (SVD).

---

### 2.9 Explainable AI

##### 2.9.1 SHAP
- **Problem it solves:** quantifying how much each feature contributed to a specific model prediction.
- **Why use it:** consistent, theoretically grounded (game-theoretic) feature attribution.
- **When to use it:** explaining predictions from complex models (XGBoost, LightGBM, CatBoost, Random Forest) to stakeholders or for compliance.
- **When to avoid it:** extremely large datasets/models where exact SHAP computation is too slow (use approximate/TreeSHAP variants).
- **Advantages:** theoretically consistent, works well with tree-based models, provides both global and local explanations.
- **Limitations:** can be computationally expensive for some model types; results can be misinterpreted without proper context.
- **Alternatives:** LIME (simpler, model-agnostic), Feature Importance (simpler, less rigorous).
- **How to implement it:** `shap` Python package.

##### 2.9.2 LIME
- **Problem it solves:** explaining an individual prediction from any model, regardless of its internal structure.
- **Why use it:** fully model-agnostic — works even on black-box models SHAP doesn't natively support.
- **When to use it:** you need a quick, local explanation for one prediction and don't need theoretical guarantees.
- **When to avoid it:** you need globally consistent explanations across many predictions (SHAP is more consistent).
- **Advantages:** model-agnostic, intuitive, works on any model.
- **Limitations:** explanations can be unstable (vary between runs), only locally valid.
- **Alternatives:** SHAP.
- **How to implement it:** `lime` Python package.

---

## 3. Deep Learning

##### 3.1 MLP (MultiLayer Perceptron) 
- **Problem it solves:** general nonlinear function approximation for tabular classification/regression.
- **Why use it:** flexible, can model complex nonlinear relationships given enough data.
- **When to use it:** tabular data with complex nonlinear patterns, or as a component within larger architectures.
- **When to avoid it:** most tabular problems, where Gradient Boosting typically performs as well or better with less tuning.
- **Advantages:** flexible, can approximate complex functions.
- **Limitations:** needs more data and tuning than tree-based models for tabular tasks; less interpretable.
- **Alternatives:** XGBoost / LightGBM / CatBoost for tabular data.
- **How to implement it:** `torch.nn` / `tensorflow.keras` (`Dense` layers), or `sklearn.neural_network.MLPClassifier`.
- **Repository**: https://github.com/mrderiu/Neuronal_Network_MLP

##### 3.2 CNN (Convolutional Neural Network)
- **Problem it solves:** detecting spatial patterns in images (or grid-like data).
- **Why use it:** convolutional filters efficiently capture local spatial structure (edges, shapes, objects).
- **When to use it:** image classification, object detection, segmentation, medical imaging.
- **When to avoid it:** tabular or non-spatial data, where CNNs offer no advantage.
- **Advantages:** excellent at capturing spatial hierarchies, parameter-efficient compared to fully connected networks on images.
- **Limitations:** needs substantial labeled image data (or transfer learning) and compute (ideally GPU).
- **Alternatives:** Vision Transformers (for large-scale image tasks).
- **How to implement it:** `torchvision.models` (ResNet, EfficientNet) or `tensorflow.keras.applications`.
- **Repository**: https://github.com/mrderiu/Convolutional_Neural_Networks_CNN

##### 3.3 RNN (Recurrent Neural Network)
- **Problem it solves:** modeling sequential data where order matters.
- **Why use it:** processes sequences of variable length, maintaining a memory of past inputs.
- **When to use it:** short-to-medium sequences (text, sensor signals, simple time series).
- **When to avoid it:** long sequences with long-term dependencies (use LSTM/GRU/Transformer instead).
- **Advantages:** naturally handles sequential/variable-length input.
- **Limitations:** struggles with long-term dependencies (vanishing gradients).
- **Alternatives:** LSTM, GRU, Transformers.
- **How to implement it:** `torch.nn.RNN` / `tensorflow.keras.layers.SimpleRNN`.

##### 3.4 LSTM
- **Problem it solves:** modeling sequences with long-term dependencies.
- **Why use it:** memory gates let it retain relevant information over long sequences, unlike plain RNNs.
- **When to use it:** time-series forecasting, text generation, sequential anomaly detection with long-range dependencies.
- **When to avoid it:** very long sequences or tasks where Transformers now consistently outperform it (most large-scale NLP).
- **Advantages:** handles long-term dependencies much better than plain RNNs.
- **Limitations:** slower to train than GRU, largely superseded by Transformers for large-scale NLP.
- **Alternatives:** GRU (lighter), Transformers (state of the art for most sequence tasks).
- **How to implement it:** `torch.nn.LSTM` / `tensorflow.keras.layers.LSTM`.

##### 3.5 GRU
- **Problem it solves:** the same long-term dependency problem as LSTM, with a lighter architecture.
- **Why use it:** fewer parameters, faster training, often comparable accuracy to LSTM.
- **When to use it:** sequential modeling where training speed/resource constraints matter.
- **When to avoid it:** tasks where Transformers are clearly state of the art and compute isn't a constraint.
- **Advantages:** faster to train than LSTM, fewer parameters, often similar performance.
- **Limitations:** still shares RNNs' sequential (non-parallelizable) computation, unlike Transformers.
- **Alternatives:** LSTM, Transformers.
- **How to implement it:** `torch.nn.GRU` / `tensorflow.keras.layers.GRU`.

##### 3.6 Transformers
- **Problem it solves:** modeling sequences (and other structured data) by learning which elements should attend to each other.
- **Why use it:** parallelizable training, captures long-range dependencies far better than RNNs/LSTMs.
- **When to use it:** as the default architecture for most modern NLP and increasingly vision tasks.
- **When to avoid it:** very small datasets or constrained compute, where a simpler model (or a pretrained Transformer via transfer learning) is more practical.
- **Advantages:** captures long-range dependencies, highly parallelizable, state of the art across many domains.
- **Limitations:** requires large amounts of data and compute to train from scratch; quadratic attention cost with sequence length.
- **Alternatives:** LSTM/GRU (for smaller-scale sequential tasks), CNNs (for vision, though Vision Transformers now compete).
- **How to implement it:** `transformers` (Hugging Face) for pretrained models; `torch.nn.Transformer` for custom architectures.

##### 3.7 Large Language Models (LLMs)
- **Problem it solves:** complex natural language understanding and generation at scale.
- **Why use it:** captures broad world knowledge and language patterns from massive pretraining.
- **When to use it:** summarization, translation, code generation, conversational systems, complex reasoning over text.
- **When to avoid it:** simple, narrow tasks where a much smaller, cheaper model (or classical NLP) would do the job just as well.
- **Advantages:** extremely versatile, strong few-shot/zero-shot capabilities.
- **Limitations:** expensive to run, can hallucinate, requires careful prompting/fine-tuning for reliability.
- **Alternatives:** smaller fine-tuned Transformers, classical NLP methods for narrow, well-defined tasks.
- **How to implement it:** APIs (Anthropic, OpenAI, etc.) or open-weight models via `transformers` / `vLLM`.

##### 3.8 Embeddings
- **Problem it solves:** representing complex objects (words, images, users) as numerical vectors that capture similarity.
- **Why use it:** enables similarity search, clustering, and input to downstream ML models from unstructured data.
- **When to use it:** semantic search, recommendation, clustering, as input to RAG pipelines.
- **When to avoid it:** when a simpler, sparse representation (like TF-IDF) already performs well and interpretability matters more than nuance.
- **Advantages:** captures semantic similarity, reusable across many downstream tasks.
- **Limitations:** not directly interpretable; quality depends heavily on the model that generated them.
- **Alternatives:** TF-IDF (sparse, interpretable), one-hot encoding (for small categorical spaces).
- **How to implement it:** `sentence-transformers`, OpenAI/Anthropic/other embedding APIs, or `word2vec`/`fastText` for word-level embeddings.

##### 3.9 RAG (Retrieval-Augmented Generation)
- **Problem it solves:** giving an LLM access to information it wasn't trained on (private, recent, or domain-specific).
- **Why use it:** avoids costly retraining/fine-tuning while keeping answers grounded in real, up-to-date documents.
- **When to use it:** the LLM needs private documents, frequently updated information, or specialized domain knowledge.
- **When to avoid it:** the required knowledge is already well captured by the base model, or the task doesn't depend on external documents.
- **Advantages:** keeps answers grounded and current without retraining the model.
- **Limitations:** quality depends heavily on retrieval quality; adds system complexity (vector DB, chunking, indexing).
- **Alternatives:** Fine-tuning (bakes knowledge into weights instead of retrieving it), long-context prompting (for smaller document sets).
- **How to implement it:** vector databases (Pinecone, Weaviate, Chroma) + `sentence-transformers` embeddings + an LLM API.

##### 3.10 Autoencoders
- **Problem it solves:** learning a compressed representation of data by reconstructing it.
- **Why use it:** useful for dimensionality reduction, denoising, and anomaly detection without labels.
- **When to use it:** unsupervised representation learning, anomaly detection, or denoising tasks.
- **When to avoid it:** you need to generate genuinely new samples (use a VAE or GAN instead), or PCA already does the job well enough linearly.
- **Advantages:** learns nonlinear compressed representations, works without labels.
- **Limitations:** the latent space isn't inherently structured for generation (unlike VAEs); requires careful architecture/tuning.
- **Alternatives:** PCA (linear, simpler), VAE (if generation is needed).
- **How to implement it:** `torch.nn` / `tensorflow.keras` encoder-decoder architecture.

##### 3.11 Variational Autoencoders (VAE)
- **Problem it solves:** generating new samples similar to training data, with a structured latent space.
- **Why use it:** unlike plain autoencoders, the latent space is a proper probability distribution you can sample from.
- **When to use it:** generative tasks where a smooth, structured latent space is valuable (e.g., interpolating between samples).
- **When to avoid it:** you need the sharpest possible generated images (GANs/diffusion models typically produce higher fidelity).
- **Advantages:** structured latent space, stable training compared to GANs.
- **Limitations:** generated samples tend to be blurrier than GAN/diffusion outputs.
- **Alternatives:** GANs, Diffusion Models.
- **How to implement it:** custom encoder-decoder with a reparameterization trick in PyTorch/TensorFlow.

##### 3.12 GANs
- **Problem it solves:** generating realistic synthetic data (especially images).
- **Why use it:** adversarial training can produce very sharp, realistic samples.
- **When to use it:** image generation, style transfer, synthetic data augmentation.
- **When to avoid it:** you need stable, easy-to-train generative modeling (training GANs is notoriously tricky) — diffusion models are often preferred today.
- **Advantages:** can produce very high-fidelity, realistic samples.
- **Limitations:** notoriously unstable training (mode collapse), hard to tune.
- **Alternatives:** Diffusion Models (now often preferred for image generation), VAEs (more stable, lower fidelity).
- **How to implement it:** custom generator/discriminator in PyTorch, or pretrained GAN libraries.

##### 3.13 Diffusion Models
- **Problem it solves:** high-quality generation of images, audio, or video.
- **Why use it:** currently produces state-of-the-art image/video generation quality with more stable training than GANs.
- **When to use it:** image/video/audio generation and editing tasks.
- **When to avoid it:** you need very fast (real-time) generation — diffusion sampling is slower than a single GAN forward pass.
- **Advantages:** state-of-the-art generation quality, more stable training than GANs.
- **Limitations:** slow inference (many denoising steps), computationally expensive.
- **Alternatives:** GANs (faster inference, less stable training), VAEs.
- **How to implement it:** `diffusers` (Hugging Face) for pretrained diffusion pipelines (Stable Diffusion, etc.).

##### 3.14 Graph Neural Networks
- **Problem it solves:** learning from data structured as graphs (nodes + edges).
- **Why use it:** captures relational structure that standard neural networks (CNN, MLP) can't represent directly.
- **When to use it:** social networks, fraud detection, molecular property prediction, transport networks, knowledge graphs.
- **When to avoid it:** data isn't naturally graph-structured, or a simpler tabular representation captures what matters.
- **Advantages:** naturally models relational/graph-structured data.
- **Limitations:** more complex to implement and tune than standard architectures; can be harder to scale to very large graphs.
- **Alternatives:** feature engineering + standard ML models on graph-derived features (e.g., node degree, centrality).
- **How to implement it:** `PyTorch Geometric` or `DGL` (Deep Graph Library).

---

## 4. Natural Language Processing

##### 4.1 TF-IDF
- **Problem it solves:** representing text numerically based on term importance across a document collection.
- **Why use it:** simple, fast, and surprisingly strong baseline for many text tasks.
- **When to use it:** text classification baselines, keyword extraction, classical search/retrieval.
- **When to avoid it:** you need to capture semantic similarity beyond exact word overlap (use embeddings instead).
- **Advantages:** fast, interpretable, requires no training data beyond the corpus itself.
- **Limitations:** ignores word order and semantic meaning; sparse, high-dimensional representation.
- **Alternatives:** Word2Vec, sentence embeddings, Transformer-based representations.
- **How to implement it:** `sklearn.feature_extraction.text.TfidfVectorizer`.

##### 4.2 Word2Vec
- **Problem it solves:** learning dense vector representations of words that capture semantic relationships.
- **Why use it:** captures analogy-like semantic relationships ("king − man + woman ≈ queen") better than sparse methods.
- **When to use it:** as a stepping stone before modern contextual embeddings, or when a lightweight, offline word-level embedding is sufficient.
- **When to avoid it:** you need context-dependent meaning (Word2Vec gives one fixed vector per word regardless of context) — use Transformer embeddings instead.
- **Advantages:** captures semantic relationships, relatively lightweight and fast to train.
- **Limitations:** one vector per word regardless of context; largely superseded by contextual embeddings.
- **Alternatives:** GloVe, FastText, Transformer-based contextual embeddings (BERT, sentence-transformers).
- **How to implement it:** `gensim.models.Word2Vec`.

##### 4.3 Topic Modeling (LDA)
- **Problem it solves:** discovering latent topics within a collection of documents.
- **Why use it:** unsupervised way to explore and organize large text collections without labels.
- **When to use it:** exploratory analysis of large, unlabeled document collections.
- **When to avoid it:** you need precise, high-quality topic labels for downstream decisions — modern embedding + clustering approaches often work better.
- **Advantages:** unsupervised, interpretable topic-word distributions.
- **Limitations:** requires choosing the number of topics in advance; topics can be noisy or hard to interpret.
- **Alternatives:** embedding-based clustering (e.g., BERTopic), NMF (Non-negative Matrix Factorization).
- **How to implement it:** `gensim.models.LdaModel` or `sklearn.decomposition.LatentDirichletAllocation`.

---

## 5. Computer Vision

- **Problem it solves:** enabling machines to interpret and reason about images and video.
- **Why use it:** automates visual tasks that would otherwise require manual human inspection.
- **When to use it:** image classification, object detection, segmentation, facial recognition, medical imaging.
- **When to avoid it:** the task doesn't actually involve visual/image data.
- **Advantages:** mature tooling, strong pretrained models available via transfer learning.
- **Limitations:** needs substantial labeled data (unless leveraging pretrained models) and often GPU compute.
- **Alternatives:** classical image processing (edge detection, thresholding) for simple, well-defined tasks.
- **How to implement it:** CNNs / Vision Transformers via `torchvision`, `timm`, or `tensorflow.keras.applications`.

---

## 6. Generative AI

- **Problem it solves:** creating new content (text, images, audio, video) rather than just analyzing existing data.
- **Why use it:** automates content creation, ideation, and prototyping at scale.
- **When to use it:** content generation, assisted design, synthetic data creation, conversational assistants.
- **When to avoid it:** the task requires strictly deterministic, verifiable outputs where hallucination/variability is unacceptable.
- **Advantages:** highly versatile, rapidly improving quality across modalities.
- **Limitations:** can produce inconsistent or incorrect ("hallucinated") outputs; compute-intensive.
- **Alternatives:** rule-based or template-based content generation for narrow, controlled use cases.
- **How to implement it:** LLM APIs (text), `diffusers` (images), or specialized audio/video generation APIs.

---

## 7. Robotics & Autonomous Systems

- **Problem it solves:** enabling a physical agent to perceive, decide, and act in the real world.
- **Why use it:** necessary whenever software needs to control something physical (vehicles, drones, manipulators).
- **When to use it:** autonomous navigation, robotic manipulation, self-driving vehicles, drones.
- **When to avoid it:** the problem is purely digital/software with no physical actuation involved.
- **Advantages:** integrates perception (Computer Vision), estimation (Kalman Filter), and decision-making (Reinforcement Learning) into one system.
- **Limitations:** safety-critical, requires extensive real-world testing, hard to simulate perfectly.
- **Alternatives:** simulation-only environments for early-stage development before real-world deployment.
- **How to implement it:** ROS (Robot Operating System), simulators like Gazebo or Isaac Sim, combined with RL/CV frameworks.

---

## 8. Model Evaluation

**Regression:** MAE, MSE, RMSE, R², MAPE.
**Classification:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC, Log Loss.

On imbalanced datasets, accuracy can be misleading — Precision, Recall, F1, and PR-AUC are usually more informative.

- **How to implement it:** `sklearn.metrics` (`mean_absolute_error`, `mean_squared_error`, `r2_score`, `accuracy_score`, `f1_score`, `roc_auc_score`, etc.).

---

## 9. Algorithm Selection Cheat Sheet

| Problem | Recommended Algorithms |
|---|---|
| Simple regression | Linear Regression |
| Nonlinear tabular regression | Random Forest / XGBoost |
| Simple binary classification | Logistic Regression |
| Complex tabular classification | XGBoost / LightGBM / CatBoost |
| Small high-dimensional dataset | SVM |
| Text classification baseline | TF-IDF + Logistic Regression |
| Customer segmentation | K-Means |
| Geographic clustering | DBSCAN |
| Probabilistic clustering | GMM |
| Dimensionality reduction | PCA |
| Data visualization | PCA / UMAP / t-SNE |
| Anomaly detection | Isolation Forest |
| Seasonal time series | SARIMA / Exponential Smoothing |
| Complex time series | XGBoost / LSTM / Transformer |
| State transitions | Markov Chains |
| Future probability scenarios | Monte Carlo |
| Images | CNN / Vision Transformer |
| Sequential data | LSTM / GRU / Transformer |
| Language understanding | Transformer |
| Text generation | LLM |
| Semantic search | Embeddings |
| Private knowledge + LLM | RAG |
| Image generation | Diffusion Models |
| Synthetic data | GAN / VAE / Diffusion |
| Graph relationships | GNN |
| Recommendation | Collaborative Filtering / Matrix Factorization |
| Autonomous decision making | Reinforcement Learning |
| Complex optimization | Genetic Algorithms / PSO |
| Transparent deterministic rules | Expert Systems |
| Uncertain relationships | Bayesian Networks |

---

## 10. Practical Workflow for Tabular Data

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

This progression helps determine whether the added complexity actually improves performance. On many real-world tabular datasets, **Gradient Boosting algorithms outperform deep neural networks** as a starting point. Deep Learning becomes more attractive with large volumes of images, text, audio, video, or multimodal data.

---

## 11. The Most Important Rule

There is no universally best algorithm. The right choice depends on:

```text
Problem Type + Dataset Size + Number of Features + Data Quality
+ Interpretability Requirements + Latency Requirements
+ Computational Resources + Business Constraints
```

A more sophisticated model isn't automatically a better model. A good Machine Learning workflow usually starts with a simple baseline and increases complexity only when necessary.
# 12. Algorithm Selection Cheat Sheet

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

# 13. Tabular Data: Practical Model Selection

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

>  **Pending:** Please add the actual links to each of your repos here as you upload them (one per algorithm/project, or grouped by thematic block).

| Bloque | Código | Estado |
|---|--- |---|
| Regresión Lineal / Logística | [repo](#) | 🔲 pendiente |
| Árboles / Random Forest / XGBoost | [repo](#) | 🔲 pendiente |
| Clustering (K-Means, DBSCAN, GMM) | [repo](#) | 🔲 pendiente |
| Series temporales | [repo](#) | 🔲 pendiente |
| Deep Learning (MLP) | https://github.com/mrderiu/Neuronal_Network_MLP | DONE |
| Deep Learning (CNN) | https://github.com/mrderiu/Convolutional_Neural_Networks_CNN | DONE |
| Transformers / LLMs / RAG | [repo](#) | 🔲 pendiente |
| Generative AI (GAN/VAE/Diffusion) | [repo](#) | 🔲 pendiente |
| Reinforcement Learning | [repo](#) | 🔲 pendiente |

---

# Final perspective

Artificial Intelligence should not be understood as a collection of isolated algorithms, but as a **toolbox**.

```text
Business Problem / Research
         ↓
 Understand the Data
         ↓
 Define the ML Problem
         ↓
  Select a Baseline
         ↓
       Train
         ↓
      Evaluate
         ↓
      Explain
         ↓
      Improve
         ↓
      Deploy
         ↓
      Monitor
```

**True skill lies not in knowing how to execute each algorithm, but in knowing which approach to use, why it is appropriate, what assumptions it makes, what its limitations are, and how to evaluate whether it actually solves the problem.**

---

<div align="center">

Notes created during the Master's in Artificial Intelligence and Programming

</div>

