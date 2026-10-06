# Project TODO

## Completed

- [x] Create isolated Python 3.13 environment
- [x] Install and validate project dependencies
- [x] Implement CSV and Parquet transaction loading
- [x] Implement customer RFM feature engineering
- [x] Implement standardized K-Means++ baseline
- [x] Add silhouette, Davies-Bouldin, Calinski-Harabasz, and inertia metrics
- [x] Add behavioral autoencoder and LDA review-topic representations
- [x] Add embedding fusion
- [x] Add K-Means and BIRCH clustering options
- [x] Add PSI drift monitoring and incremental MiniBatchKMeans updates
- [x] Add adaptive cluster-count selection
- [x] Add segment profiles
- [x] Add SHAP cluster-membership explanations
- [x] Add Streamlit dashboard
- [x] Add report generation to JSON, CSV, and Markdown
- [x] Add FastAPI health endpoint
- [x] Add focused tests and dependency validation
- [x] Add the supplied transaction CSV as a raw-data demo
- [x] Run the baseline and report generation on real transaction data
- [x] Record initial measured baseline metrics
- [x] Add DBSCAN noise filtering followed by K-Means
- [x] Compare baseline and hybrid metrics on the supplied data
- [x] Add SDG 8, 9, and 12 alignment to the generated report
- [x] Add DBSCAN epsilon/min_samples sensitivity sweep
- [x] Add retention-first configuration selection and business-viability discussion
- [x] Add automatic dataset profiling and algorithm selection
- [x] Add smart-selector CLI and report integration

## Next Priorities

- [ ] Connect `/predict` to a persisted scaler and clustering model
- [ ] Add model save/load commands for repeatable inference
- [x] Add a small sample transaction dataset and an end-to-end demo command
- [ ] Add dashboard download buttons for assignments, profiles, and reports
- [ ] Add API tests for valid and invalid prediction requests
- [ ] Add benchmark comparison for raw RFM versus autoencoder embeddings
- [x] Document measured baseline results using the supplied transaction dataset
- [ ] Add CI to run tests and dependency checks

## Future Research Integrations

- [ ] Replace LDA/lightweight text encoding with BERT embeddings
- [ ] Implement heterogeneous hypergraph neural network clustering
- [ ] Implement a true reinforcement-learning hyperparameter agent
- [ ] Add Evidently drift reports and automated retraining triggers
- [ ] Add MLflow experiment tracking and model registry
- [ ] Add Kafka/Flink streaming ingestion
- [ ] Evaluate scalability on millions of customers
