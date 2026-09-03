# Machine Learning Systems Lab

> **A first-principles, anti-black-box curriculum for engineering reliable Machine Learning pipelines.**

In an era where AI tools can generate dozens of lines of code in seconds, syntax memorization has lost its value. The real bottleneck in modern software and data engineering is **system comprehension, data flow architecture, and debugging logic.**

This repository is built to train developers to stop relying on blind code generation and start understanding the mechanical plumbing behind machine learning algorithms.

---

## 🧭 The 4-Stage Learning Loop

Every algorithm module in this laboratory is built around an uncompromising 4-stage pedagogical workflow:

1. **Stage 1: The Architectural Blueprint**  
   Before writing code, students analyze data shape transformations, identify library "black boxes," and map the end-to-end data flow using a structured pre-flight checklist.

2. **Stage 2: First-Principles Math Engine**  
   We demystify the algorithm by calculating its core mechanics using raw Python loops and basic math, completely bypassing high-level abstractions.

3. **Stage 3: Leak-Proof Pipeline Guardrails**  
   We integrate production frameworks alongside strict automated testing assertions that immediately halt execution if data leakage occurs between training and testing splits.

4. **Stage 4: Adversarial Debugging**  
   Students run intentionally broken code to encounter, analyze, and resolve real-world failure states (such as 1D array traps, collinearity clashes, and dimensionality mismatches).

---

## 📂 Laboratory Roadmap

### Phase 1: Foundations
- [ ] **01-environment-and-plumbing** — Virtual environments, command-line navigation, data ingestion, and Pandas shape manipulation.

### Phase 2: Regression Systems
- [ ] **02-simple-and-multiple-linear-regression** — Manual hyperplane math, leak-proof scaling, and collinearity debugging.
- [ ] **03-polynomial-and-non-linear-regression** — Degree curve fitting, feature mapping, and understanding non-linear equations.

### Phase 3: Classification Architecture
- [ ] **04-k-nearest-neighbors-knn** — Distance metrics math, K-value optimization, and high-dimensional space mapping.
- [ ] **05-decision-trees** — Manual entropy/Gini impurity calculations, node splitting logic, and tree pruning.
- [ ] **06-logistic-regression** — Binary classification mechanics, sigmoid functions, and cross-entropy loss calculation from scratch.
- [ ] **07-support-vector-machines-svm** — Margin maximization, hyperplanes, and the kernel trick under the hood.

### Phase 4: Unsupervised Learning & Clustering
- [ ] **08-k-means-clustering** — Centroid initialization, distance iteration math, and WCSS (Within-Cluster Sum of Square) logic.
- [ ] **09-hierarchical-clustering** — Agglomerative vs. divisive logic, distance matrices, and dendrogram data flow.
- [ ] **10-dbscan** — Density-based spatial clustering, core-point identification, and noise/outlier handling.

### Phase 5: Recommender Systems
- [ ] **11-content-based-filtering** — Profile building, TF-IDF vectors, and similarity matrix architecture.
- [ ] **12-collaborative-filtering** — User-item matrices, similarity scores, and mitigating the cold-start problem.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Microsoft VS Code
- Git installed on your local machine

### Local Setup
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR-GITHUB-USERNAME/ml-systems-lab.git](https://github.com/YOUR-GITHUB-USERNAME/ml-systems-lab.git)
   cd ml-systems-lab
