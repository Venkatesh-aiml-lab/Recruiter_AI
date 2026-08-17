# RecruiterAI — AI-Powered Recruitment Decision-Support Platform

RecruiterAI is a production-oriented AI recruitment prototype for **semantic Job Description (JD)–resume matching, candidate ranking, skill-gap analysis, experience-gap analysis, and evidence-based recruiter support**.

The system is designed as a **decision-support tool**, not an autonomous hiring system. Final hiring decisions remain with human recruiters.

---

## 1. Business Problem

Recruitment teams may receive hundreds or thousands of resumes for a single position. Traditional keyword-based screening has several limitations:

- A keyword may appear in a resume without demonstrating actual competency.
- Equivalent skills may be written differently.
- Manual comparison of experience, skills, education, and job requirements is time-consuming.
- Candidate ranking can become inconsistent across large applicant pools.
- Keyword-only systems can miss semantically relevant candidates.
- Rankings without explanations are difficult for recruiters to audit or trust.

RecruiterAI addresses these limitations by combining classical NLP, deep learning, Transformer-based semantic matching, ranking metrics, and recruiter-facing explanations.

---

## 2. Project Objectives

The project aims to:

1. Prepare structured resume and job-description text.
2. Detect and remove data leakage.
3. Exclude protected/sensitive attributes and strong proxy features from ranking.
4. Build baseline and deep-learning ranking models.
5. Perform semantic JD–resume matching.
6. Rank candidates for each job.
7. Improve recovery of the true Top-10 candidates using Recall@10-focused training.
8. Identify skill and experience gaps.
9. Generate evidence-based candidate explanations.
10. Keep a human recruiter in the final decision loop.

---

## 3. High-Level Architecture

```text
Job Description
      |
      v
Text / Requirement Preparation
      |
      v
Sentence-Transformer Bi-Encoder
      |
      v
Top-K Semantic Retrieval
      |
      v
BERT / MiniLM Cross-Encoder
      |
      v
Candidate Re-ranking
      |
      +--> Skill-Gap Analysis
      |
      +--> Experience-Gap Analysis
      |
      +--> Evidence / Explanation
      |
      v
Recruiter Decision Support
      |
      v
Human Decision
```

For a large candidate repository, the intended architecture is a **multi-stage retrieval and reranking pipeline** rather than running an expensive Cross-Encoder against every resume.

---

## 4. Models Implemented

The notebook benchmarks multiple approaches so that more complex Transformer models can be compared with simpler baselines.

### Classical / ML Baselines

- TF-IDF + cosine similarity
- TF-IDF + Logistic Regression
- MLP Regressor

### Deep Learning

- Bi-LSTM

### Transformer / Semantic Models

- Sentence Transformer Bi-Encoder
- BERT Pair Regressor
- BERT Recall Ranker
- Validation-tuned BERT Hybrid
- Fine-tuned Cross-Encoder
- Two-stage Bi-Encoder → Cross-Encoder retrieval and reranking

---

## 5. Recall@10-Focused BERT Update

The latest notebook contains additional logic specifically designed to address low **Recall@10**.

### BERT Pair Regressor

The BERT Pair Regressor receives:

```text
Job Description + Resume
          |
          v
         BERT
          |
          v
   Regression Head
          |
          v
Predicted matched_score
```

It is fine-tuned as a regression model using the continuous `matched_score`.

Unlike the earlier version, the best BERT checkpoint is selected using **validation Recall@10** rather than only the lowest MAE.

### BERT Recall Ranker

A second BERT model is trained specifically to identify whether a candidate belongs to the ground-truth Top-10 candidates for a JD.

```text
Ground-truth Top-10 candidate -> 1
Other candidate               -> 0
```

Because Top-10 candidates are relatively rare compared with non-Top-10 candidates, the model uses **weighted binary cross-entropy** so positive examples receive more influence during training.

### BERT Hybrid

The notebook combines:

```text
BERT Pair Regressor
        +
BERT Recall Ranker
        |
        v
Validation-tuned blend
        |
        v
Final Candidate Ranking
```

The blend weight is selected using the **validation set only**.

Selection priority:

1. Recall@10
2. NDCG@10
3. MRR@10

The chosen weight is frozen before evaluation on the held-out test set to avoid test-set leakage.

---

## 6. Evaluation Metrics

Because the main problem is candidate ranking, ranking-specific metrics are prioritized.

### Ranking Metrics

- **NDCG@10** — evaluates whether highly relevant candidates are placed near the top.
- **Recall@10** — measures how many of the ground-truth Top-10 candidates are recovered in the predicted Top-10.
- **MRR@10** — measures how early the first relevant candidate appears.
- **HitRate@10** — checks whether at least one ground-truth Top-10 candidate appears in the predicted Top-10.

### Regression Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

### Recall-Ranker Metric

- **PR-AUC / Average Precision** — useful for evaluating the imbalanced Top-10 relevance classification task.

Model selection should not be based on training loss or accuracy alone.

---

## 7. Data Leakage and Responsible Feature Selection

The dataset contains an important leakage issue:

```text
responsibilities == responsibilities.1
```

The notebook therefore excludes resume-side `responsibilities` from the ranking features and retains `responsibilities.1` only as a JD-side field.

The following types of attributes are also excluded from ranking where applicable:

- Address
- Age requirements
- Passing years
- Educational institution names
- Professional company names
- Company URLs
- Locations
- Languages / proficiency fields when treated as sensitive or proxy attributes
- Resume-side leaking responsibilities
- `matched_score` as an input feature because it is the target

The objective is to reduce leakage and avoid inappropriate use of protected, sensitive, or strong proxy characteristics.

---

## 8. Responsible AI Principles

RecruiterAI follows these principles:

- Avoid protected/sensitive characteristics in ranking.
- Provide auditability.
- Support human review.
- Expose model limitations.
- Avoid autonomous hiring or rejection decisions.
- Treat generated rankings as decision support rather than final employment decisions.

A missing skill in the generated explanation means that the skill was **not clearly demonstrated in the available structured data**; it does not prove that the candidate lacks that skill.

---

## 9. Project Structure

A recommended GitHub structure is:

```text
RecruiterAI/
|
├── RecruitAI_End_to_End_Colab_Recall10_Improved.ipynb
├── README.md
├── requirements.txt
|
├── data/
│   └── resume_data_for_ranking.csv
|
└── artifacts/
    ├── model_comparison.csv
    ├── tfidf_retriever.joblib
    ├── tfidf_pair_vectorizer.joblib
    ├── logistic_regression.joblib
    ├── mlp_pipeline.joblib
    ├── bilstm_model.keras
    ├── sentence_transformer/
    ├── bert_pair_regressor/
    ├── bert_recall_ranker/
    ├── bert_hybrid_config.json
    └── cross_encoder/
```

> If the dataset contains real or sensitive candidate information, do not commit it to a public GitHub repository.

---

## 10. Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- FAISS
- Matplotlib
- Google Colab

---

## 11. Environment Setup

The project is designed to run in **Google Colab**.

For Transformer experiments:

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select a **GPU** runtime.
4. Install the project dependencies.

```bash
pip install -r requirements.txt
```

The notebook itself also contains a dependency-installation cell for a fresh Colab runtime.

---

## 12. Dataset Setup

The notebook supports two approaches.

### Option A — Google Drive

Store the CSV in Google Drive and configure:

```python
USE_DRIVE = True
DRIVE_CSV_PATH = "/content/drive/MyDrive/RecruiterAI/resume_data_for_ranking.csv"
```

### Option B — Manual Colab Upload

Set:

```python
USE_DRIVE = False
```

and upload the CSV when prompted.

---

## 13. Running the Project

Run the notebook sequentially from top to bottom.

The major stages are:

```text
1. Runtime and dependency setup
2. Data loading
3. Data-quality audit
4. Leakage audit
5. Text preprocessing
6. Grouped train / validation / test split
7. Ranking metric implementation
8. TF-IDF baseline
9. Logistic Regression baseline
10. MLP benchmark
11. Bi-LSTM benchmark
12. Sentence Transformer retrieval
13. BERT Pair Regressor fine-tuning
14. BERT Recall Ranker fine-tuning
15. Validation-tuned BERT Hybrid
16. Cross-Encoder fine-tuning
17. Two-stage retrieval + reranking
18. Skill-gap analysis
19. Experience-gap analysis
20. Candidate explanations
21. Model comparison
22. Artifact saving
```

---

## 14. Important BERT Configuration

The current notebook uses:

```python
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
BERT_MAX_LENGTH = 384
```

The Cross-Encoder uses:

```python
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
```

BERT and Cross-Encoder training should preferably be run with a GPU.

---

## 15. Train / Validation / Test Strategy

The notebook uses grouped splitting based on job position so that job titles do not leak across train, validation, and test partitions.

Conceptually:

```text
Training JDs
     |
     v
Model training

Validation JDs
     |
     +--> checkpoint selection
     +--> Recall@10 optimization
     +--> hybrid-weight selection

Held-out Test JDs
     |
     v
Final unbiased evaluation
```

The validation set is used for model selection and tuning. The test set should only be used for final evaluation.

---

## 16. Generated Artifacts

The notebook can save trained models and experiment results to:

```text
/content/recruitai_artifacts
```

Generated artifacts include:

- TF-IDF models
- Logistic Regression model
- MLP pipeline
- Bi-LSTM model
- Sentence Transformer
- BERT Pair Regressor
- BERT Recall Ranker
- BERT hybrid configuration
- Cross-Encoder
- Final model comparison CSV

The artifacts can also be copied to Google Drive.

---

## 17. Model Selection

The latest Recall-focused notebook prioritizes:

```text
Recall@10
   |
   v
NDCG@10
   |
   v
MRR@10
```

Other considerations include:

- MAE / RMSE
- Latency
- Memory consumption
- Scalability
- Explainability
- Failure cases
- Responsible AI constraints

A final architecture should be selected from reproducible held-out evaluation results rather than assumed performance.

---

## 18. Current Project Status

This repository represents a **production-oriented reference implementation / AI recruitment decision-support prototype**.

It demonstrates:

- Resume and JD text preparation
- Leakage detection
- Responsible feature selection
- Classical ranking baselines
- Neural-network benchmarking
- Semantic embedding retrieval
- BERT fine-tuning
- Recall-focused ranking
- Cross-Encoder reranking
- Multi-stage retrieval architecture
- Ranking-specific evaluation
- Skill-gap analysis
- Experience evidence
- Recruiter-facing explanations
- Human-in-the-loop decision support

It should not be described as an autonomous hiring system or as a production deployment unless it has actually been deployed and validated in such an environment.

---

## 19. Future Improvements

Potential future work includes:

- Hyperparameter optimization
- Hard-negative mining
- Pairwise or listwise learning-to-rank objectives
- Better skill normalization and ontology mapping
- More robust experience extraction
- Calibration of ranking scores
- Fairness and subgroup evaluation
- Latency and throughput benchmarking
- Larger-scale FAISS retrieval
- Model compression / distillation
- API-based inference
- Recruiter-facing UI
- Monitoring for drift and ranking-quality degradation

---

## 20. Disclaimer

RecruiterAI is intended for **research, learning, and human-assisted recruitment decision support**.

Candidate ranking models can contain errors and may reflect limitations or biases in the underlying data. Predictions and explanations must be reviewed by qualified human decision-makers before being used in any employment-related decision.
