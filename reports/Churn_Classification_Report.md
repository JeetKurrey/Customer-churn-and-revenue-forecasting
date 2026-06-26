
TELCO CUSTOMER CHURN PREDICTION
Machine Learning Classification Analysis

Classification — Churn vs. Retained
Dataset: Processed Telco Customer Churn (IBM/Kaggle)



Executive Summary
This report presents a comprehensive evaluation of five supervised machine learning classification algorithms applied to the Telco Customer Churn Prediction problem. The primary objective was to identify customers likely to discontinue their service subscriptions, enabling proactive retention strategies.

Five models were benchmarked: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). Each model was trained on a stratified 70/15/15 train-validation-test split of the preprocessed Telco dataset and optimized via 5-fold cross-validated GridSearchCV tuned for the F1-Score.

Key Findings
• Logistic Regression emerged as the best overall model with the highest ROC-AUC of 0.8274, balancing precision and recall better than all ensemble and kernel-based alternatives.
• Class imbalance was the dominant challenge across all models, systematically suppressing recall (churner identification) in favor of majority-class accuracy.
• Ensemble methods (Random Forest) and kernel methods (SVM) demonstrated high precision but lower recall at the default 0.5 threshold.
• KNN performed worst overall due to the curse of dimensionality in the 34-feature processed feature space.
• Strong linear predictors — tenure, contract type, and monthly charges — dictated model rankings, rewarding linear models over complex non-linear alternatives on this dataset.

ModelAccuracyPrecisionRecallF1-ScoreROC-AUCLogistic Regression ?78.39%61.30%50.36%55.29%0.8274Decision Tree76.30%55.95%50.36%53.01%0.7881Random Forest78.39%63.00%45.00%52.50%0.8166SVM78.58%63.37%45.71%53.11%0.7848KNN75.73%54.80%48.93%51.70%0.7874? Best Performing Model   |   All metrics evaluated on held-out test set


1. Introduction & Problem Statement
1.1 Business Context
Customer churn — the phenomenon of subscribers discontinuing service — is a critical business metric for telecommunications companies. Acquiring a new customer costs significantly more than retaining an existing one. Predictive models that identify at-risk customers before churn occurs enable targeted intervention: personalized offers, service improvements, or loyalty programs.

1.2 Dataset Overview
The analysis uses the processed IBM Telco Customer Churn dataset, a widely used benchmark for binary classification tasks. The dataset captures customer demographics, account information, service subscriptions, and churn labels.

AttributeDescriptionSource Filepreprocessed-Telco-Customer-Churn_final.csvTarget VariableChurn (Binary: 0 = Retained, 1 = Churned)Feature Count (post-encoding)34 Features (after one-hot encoding + drop_first)Class DistributionImbalanced — approx. 73% Non-Churn, 27% ChurnExecution EnvironmentGoogle Colab (Python 3, scikit-learn)
1.3 Objectives
• Train and tune five classification algorithms on the Telco churn dataset.
• Evaluate each model using industry-standard classification metrics.
• Identify the single best model for deployment based on ROC-AUC and balanced precision-recall trade-offs.
• Extract and interpret the most influential features driving churn predictions.


2. Methodology & Preprocessing Pipeline
2.1 Data Splitting Strategy
A three-way stratified split was employed to prevent data leakage and simulate real-world deployment conditions. Stratification on the target variable ensured each split maintained the natural class proportion.

SplitProportionUsageStratifiedTraining Set70%Model fittingYes (on target)Validation Set15%Hyperparameter referenceYes (on target)Test Set15%Final unbiased evaluationYes (on target)
2.2 Feature Engineering
One-Hot Encoding
All categorical string features were transformed into binary dummy variables using pandas get_dummies() with drop_first=True to prevent multicollinearity. Post-encoding alignment (align() with join='left', fill_value=0) ensured that validation and test sets carried the same feature columns as the training set — a critical step to prevent silent dimension mismatches during inference.

Feature Scaling
StandardScaler (zero-mean, unit-variance normalization) was applied: fit on training data only, then transform applied to validation and test sets. This prevented information leakage from held-out data into the scaler's mean/variance statistics. Scaling was essential for distance-based models (KNN, SVM) and beneficial for Logistic Regression, though theoretically neutral for tree-based models.

2.3 Hyperparameter Optimization
Each model underwent GridSearchCV with 5-fold cross-validation, optimizing the F1-Score (macro weighted on minority class). F1 was selected as the primary optimization metric — rather than accuracy — because it penalizes both false positives and false negatives equally, making it appropriate for the imbalanced churn dataset.

ModelHyperparameter Grid SearchedLogistic RegressionC: [0.01, 0.1, 1, 10] | penalty: ['l2']Decision Treemax_depth: [3, 5, 10, None] | min_samples_split: [2, 5, 10]Random Forestn_estimators: [100, 200] | max_depth: [5, 10, None] | min_samples_split: [2, 5]SVMC: [0.1, 1, 10] | kernel: ['rbf', 'linear']KNNn_neighbors: [3, 5, 7, 9] | weights: ['uniform', 'distance']
2.4 Evaluation Framework
Each model was evaluated on the held-out test set using five standard binary classification metrics:

MetricDefinition & Business RelevanceAccuracyOverall proportion of correct predictions. Can be misleading under class imbalance.PrecisionOf all predicted churners, fraction that actually churned. Controls intervention cost overrun.RecallOf all actual churners, fraction identified. Critical — missed churners generate direct revenue loss.F1-ScoreHarmonic mean of Precision and Recall. Primary tuning metric for imbalanced classification.ROC-AUCProbability ranking quality across all thresholds. Primary model selection metric.

3. Model Results & Analysis
3.1 Logistic Regression
Hyperparameter Configuration
Grid: C ? {0.01, 0.1, 1, 10}, penalty = L2 (Ridge). Optimized via 5-fold cross-validation on F1-Score. max_iter set to 1000 to ensure convergence on the scaled feature space.

Performance Metrics
MetricScoreAccuracy78.39%Precision61.30%Recall50.36%F1-Score55.29%ROC-AUC0.8274 ? Best
Analysis
Logistic Regression established a robust linear baseline by directly optimizing log-loss probabilities. StandardScaler normalization was critical — without it, high-magnitude features like TotalCharges would dominate the gradient. The model's continuous, well-calibrated probability output translated to the highest ROC-AUC (0.8274), reflecting superior ranking ability across decision thresholds. The moderate recall (50.36%) is inherent to the default 0.5 threshold under class imbalance; lowering the threshold would trade precision for recall without retraining.

3.2 Decision Tree Classifier
Hyperparameter Configuration
Grid: max_depth ? {3, 5, 10, None}, min_samples_split ? {2, 5, 10}. Tree depth was controlled through GridSearchCV to prevent catastrophic overfitting.

Performance Metrics
MetricScoreAccuracy76.30%Precision55.95%Recall50.36%F1-Score53.01%ROC-AUC0.7881
Analysis
The Decision Tree successfully captured non-linear feature interactions — such as joint thresholds on contract type and monthly charges — without requiring scaled inputs. However, as a single un-ensembled estimator, its axis-aligned orthogonal decision boundaries are structurally ill-suited to the smooth probability surfaces of real-world churn data. Even with GridSearchCV-enforced depth constraints, the tree produced the lowest ROC-AUC (0.7881) among all non-KNN models, reflecting poor probability calibration. Its false positive rate was also the highest, driven by majority-class noise captured at leaf nodes.

3.3 Random Forest Classifier
Hyperparameter Configuration
Grid: n_estimators ? {100, 200}, max_depth ? {5, 10, None}, min_samples_split ? {2, 5}. Bootstrap aggregation with feature randomization.

Performance Metrics
MetricScoreAccuracy78.39%Precision63.00% ? 2nd BestRecall45.00% ? LowestF1-Score52.50%ROC-AUC0.8166
Analysis
Random Forest successfully mitigated the Decision Tree's high variance problem through bootstrap aggregation and feature subsampling, elevating precision to 63.00% — the second highest. The ensemble also smoothed out noise-driven false positives. However, the majority-voting mechanism of 100+ individual trees systematically biased predictions toward the dominant non-churn class, resulting in the lowest recall (45.00%) among all models. The resulting F1-Score (52.50%) and ROC-AUC (0.8166) indicate strong discriminative ability but conservative positive-class assignment at the 0.5 threshold.

3.4 Support Vector Machine (SVM)
Hyperparameter Configuration
Grid: C ? {0.1, 1, 10}, kernel ? {'rbf', 'linear'}. probability=True enabled Platt Scaling for probability calibration. Trained on StandardScaler-normalized features.

Performance Metrics
MetricScoreAccuracy78.58% ? HighestPrecision63.37% ? HighestRecall45.71%F1-Score53.11%ROC-AUC0.7848
Analysis
SVM achieved the highest raw accuracy (78.58%) and precision (63.37%) by constructing an optimal separating hyperplane with maximum margin between classes in the kernel-projected feature space. The kernel transformation (RBF or Linear selected by GridSearchCV) enabled non-linear boundary modeling. However, SVM does not natively produce calibrated class probabilities — the predict_proba() output required Platt Scaling via internal cross-validation. On an imbalanced churn dataset, this secondary probability estimation distorted the probability ranking curves, depressing the ROC-AUC to 0.7848 — the lowest of all five models — despite the superior hard-label accuracy metrics.

3.5 K-Nearest Neighbors (KNN)
Hyperparameter Configuration
Grid: n_neighbors ? {3, 5, 7, 9}, weights ? {'uniform', 'distance'}. Relies entirely on StandardScaler-normalized feature space for meaningful distance calculations.

Performance Metrics
MetricScoreAccuracy75.73% ? LowestPrecision54.80% ? LowestRecall48.93%F1-Score51.70% ? LowestROC-AUC0.7874
Analysis
KNN delivered the weakest performance profile across all primary metrics. Despite StandardScaler ensuring isotropic distance calculations, the 34-dimensional post-encoding feature space introduced significant geometric noise — the Curse of Dimensionality. In high-dimensional space, pairwise Euclidean distances between samples homogenize, eroding the conceptual meaning of 'nearest neighbors.' This distance collapse, combined with class imbalance allowing majority-class neighbors to consistently outvote minority-class churn signatures, capped both F1-Score (0.5170) and ROC-AUC (0.7874). KNN's non-parametric, instance-based nature also makes it computationally expensive and impractical for large-scale production deployment.


4. Comparative Analysis
4.1 Full Model Performance Comparison Table
ModelAccuracyPrecisionRecallF1-ScoreROC-AUCRankLogistic Regression78.39%61.30%50.36%55.29%0.8274#1Random Forest78.39%63.00%45.00%52.50%0.8166#2Decision Tree76.30%55.95%50.36%53.01%0.7881#3KNN75.73%54.80%48.93%51.70%0.7874#4SVM78.58%63.37%45.71%53.11%0.7848#5Ranked by ROC-AUC (primary selection criterion)

4.2 Metric-by-Metric Observations
Accuracy
SVM achieved the marginally highest accuracy (78.58%), followed by Logistic Regression and Random Forest (both 78.39%). However, accuracy alone is an unreliable metric on the imbalanced churn dataset — a naive 'predict no churn' classifier would achieve ~73% accuracy. The ~5-point spread between best (SVM) and worst (KNN at 75.73%) is relatively narrow.

Precision
SVM led precision (63.37%), closely followed by Random Forest (63.00%). Both ensemble/kernel models enforced stricter positive-class assignment, resulting in fewer false positives at the cost of missing more actual churners. Logistic Regression (61.30%) and Decision Tree (55.95%) trailed, while KNN produced the lowest precision (54.80%).

Recall
Logistic Regression and Decision Tree jointly achieved the highest recall (50.36%), reflecting their tendency to cast a wider net for the positive (churn) class. Ensemble and kernel models — Random Forest (45.00%), SVM (45.71%) — fell prey to majority-class bias. KNN (48.93%) performed in the middle, despite having the weakest overall profile.

F1-Score
Logistic Regression led with an F1-Score of 55.29%, reflecting the best harmonic balance of precision and recall. The other four models clustered tightly between 51.70% and 53.11%, indicating similar overall performance once both error types are considered equally.

ROC-AUC (Primary Selection Metric)
Logistic Regression achieved the highest ROC-AUC (0.8274), demonstrating the best overall probability ranking quality across all decision thresholds. Random Forest followed at 0.8166. Importantly, SVM's hard-label accuracy superiority was completely undermined by its lowest ROC-AUC (0.7848) — the Platt Scaling probability distortion robbing it of ranking reliability. This underscores why ROC-AUC, not accuracy, is the appropriate primary metric for imbalanced classification tasks.


5. Best Model: Logistic Regression
5.1 Selection Rationale
Logistic Regression was selected as the best overall model based on its superior ROC-AUC score (0.8274), the highest F1-Score (55.29%), and the highest recall among models with competitive precision. The selection reasoning is multi-dimensional:

Why Logistic Regression Outperformed Complex Models
• Linear Feature Dominance: The Telco churn dataset is strongly governed by linear predictors — tenure, contract type, and monthly charges carry near-linear relationships with churn probability. Logistic Regression is structurally optimized to exploit these linear signals.
• Calibrated Probability Output: Unlike SVM (which requires Platt Scaling post-hoc) and Decision Trees (which produce sharp probability cliffs at leaf nodes), Logistic Regression natively outputs well-calibrated probabilities through the sigmoid function. This calibration directly translates to a superior ROC-AUC.
• Threshold Sensitivity: At the 0.5 default threshold, Logistic Regression's smooth probability curve produced better recall (50.36%) than Random Forest (45.00%) and SVM (45.71%), which both fell into majority-class conservatism.
• Resistance to Imbalance: While all models suffered from class imbalance, Logistic Regression's gradient-based optimization with smooth loss landscape allowed it to encode minority-class signals more effectively than majority-vote-based Random Forest.

5.2 Top 10 Most Influential Features
Feature importance for Logistic Regression was derived from the model's learned coefficient weights (coef_[0]), representing the log-odds impact of each standardized feature on churn probability. The top 10 features by absolute coefficient magnitude are:

RankFeatureImpact on Churn1Contract_Two year (negative coef.)Strongly reduces churn risk2tenureLonger tenure ? lower churn3Contract_One year (negative coef.)Moderate churn risk reduction4MonthlyCharges (positive coef.)Higher charges ? higher churn5InternetService_Fiber optic (positive)Fiber users churn at higher rates6PaymentMethod_Electronic checkE-check correlated with churn7TechSupport_NoNo tech support ? higher churn8OnlineSecurity_NoLack of security add-on increases churn9PaperlessBilling_YesPaperless billing correlates with churn10SeniorCitizenSenior customers churn slightly more
5.3 Business Interpretation of Top Features
The feature coefficients reveal actionable retention insights:

• Contract Length is the single most protective factor against churn. Customers on month-to-month contracts are dramatically more likely to leave. Upselling to annual or two-year contracts is the highest-leverage retention action.
• Tenure acts as a natural retention flywheel — the longer a customer stays, the more embedded they become. Proactive engagement in the first 6–12 months is critical to building this loyalty.
• High Monthly Charges are a top churn driver, suggesting price-sensitive segments exist. Targeted discount offers for high-bill customers at risk could improve retention.
• Fiber Optic Internet users churn more, potentially indicating service quality or value perception issues with premium internet offerings.
• Electronic Check payment users exhibit higher churn rates, possibly signaling lower engagement or trust in the platform — an opportunity for payment experience improvement.
• Customers without TechSupport or OnlineSecurity add-ons are more likely to churn, suggesting these services increase perceived value and stickiness.


6. Model Visualizations & Diagnostics
6.1 ROC Curve Analysis
The notebook generated a multi-model ROC curve plot comparing all five classifiers on the same axes against the random guessing baseline (diagonal at AUC = 0.50).

Key Observations from the ROC Plot
• Logistic Regression's ROC curve (AUC = 0.8274) sits highest in the upper-left quadrant across most False Positive Rate values, confirming superior probability ranking.
• Random Forest (AUC = 0.8166) closely trails Logistic Regression, with its curve converging with LR at moderate FPR values.
• SVM (AUC = 0.7848) shows a notable curve deflection at mid-range FPR values — a signature of Platt Scaling distortion on imbalanced data.
• Decision Tree (AUC = 0.7881) and KNN (AUC = 0.7874) cluster together at the bottom of the model group, confirming their inferior ranking ability.
• All five models significantly outperform the random guessing baseline, confirming that meaningful churn signal exists in the feature set.

6.2 Confusion Matrix Analysis
Five individual confusion matrices were generated in a 2×3 grid layout using seaborn heatmaps with the 'Blues' colormap, showing True Negative (TN), False Positive (FP), False Negative (FN), and True Positive (TP) counts.

Key Observations from Confusion Matrices
• All models show high True Negative counts, reflecting the natural class imbalance (~73% non-churn base rate).
• Logistic Regression produced the highest True Positive count (actual churners correctly identified), consistent with its superior recall.
• Random Forest and SVM showed notably fewer True Positives than Logistic Regression, confirming their majority-class conservatism.
• Decision Tree exhibited the highest False Positive count (non-churners incorrectly flagged), matching its lower precision score.
• KNN's confusion matrix showed the most balanced error distribution, yet all metrics were weakest — indicating structural inability to separate classes in high dimensions.

6.3 Feature Importance Visualization
A horizontal bar chart was generated using seaborn's barplot(), displaying the top 10 Logistic Regression coefficients sorted by absolute magnitude. The chart confirms that contract-type features and tenure dominate the model's decision boundary, validating the business interpretation above.


7. Model Persistence & Deployment Readiness
7.1 Serialized Artifacts
The trained model pipeline was serialized using joblib and saved to the /models directory for downstream deployment:

FileContentsPurposemodels/best_classifier.pklFitted LogisticRegression objectPrediction inferencemodels/scaler.pklFitted StandardScalerFeature normalizationmodels/feature_columns.pklX_train_encoded column listColumn alignment at inference
7.2 Inference Pipeline
For correct production inference, all three artifacts must be loaded and applied in sequence: (1) load new customer data, (2) apply one-hot encoding with alignment to feature_columns.pkl, (3) transform with scaler.pkl, (4) predict with best_classifier.pkl. Skipping any step will cause silent prediction errors.

7.3 Known Limitations
• Class Imbalance Not Addressed in Training: No class_weight='balanced', SMOTE, or threshold tuning was applied. The 0.5 decision threshold is suboptimal for the actual class distribution.
• Threshold Tuning Opportunity: Lowering the prediction threshold to 0.35–0.40 on the Logistic Regression model would increase recall (catching more churners) at the cost of precision. The business-optimal threshold depends on the relative cost of false negatives (missed churners) vs. false positives (unnecessary retention spend).
• Dataset Temporal Validity: The model was trained on a static snapshot. Customer behavior evolves — periodic retraining is necessary to maintain performance.
• No Calibration Verification: While Logistic Regression is generally well-calibrated, formal calibration curves (reliability diagrams) were not included in this analysis.


8. Recommendations & Next Steps
8.1 Immediate Production Improvements
• Class Weight Balancing: Retrain all models with class_weight='balanced' (scikit-learn parameter) to directly compensate for the 73/27 class imbalance. This is likely to dramatically improve recall across all models.
• Threshold Optimization: Use the validation set to identify the F1-optimal or business-cost-optimal decision threshold for the Logistic Regression model, replacing the default 0.5 cutoff.
• SMOTE Oversampling: Apply Synthetic Minority Oversampling Technique (SMOTE) on the training set before model fitting to generate synthetic minority-class samples, improving minority-class learning without test set contamination.

8.2 Advanced Model Exploration
• XGBoost / LightGBM: Gradient boosted trees with built-in regularization are consistently top performers on tabular classification tasks with class imbalance. These should be benchmarked before final deployment decisions.
• Ensemble Stacking: A stacking ensemble combining Logistic Regression (strong linear signal) with Random Forest (non-linear patterns) may produce additive improvements.
• Calibration Curves: Generate reliability diagrams to verify the Logistic Regression's probability calibration is accurate across the full probability range.

8.3 Business Deployment Strategy
• Segment-Specific Interventions: Use the model's top features (contract length, tenure, monthly charges) to define customer risk tiers. Apply differentiated retention strategies by tier rather than uniform interventions.
• Fiber Optic Quality Investigation: The Fiber Optic feature's high positive coefficient warrants a dedicated investigation into whether service quality issues or value perception are driving these customers to churn.
• Early Tenure Programs: Since tenure is a top protective feature, design onboarding and early-engagement programs specifically targeting customers in their first 6 months.
• Monitoring & Retraining Pipeline: Implement a model performance monitoring dashboard tracking precision, recall, and F1 on new incoming predictions, with automated retraining triggers when drift is detected.


9. Conclusion
This analysis evaluated five supervised classification algorithms on the Telco Customer Churn Prediction task, encompassing the full machine learning pipeline: data splitting, one-hot encoding, feature scaling, hyperparameter optimization via GridSearchCV, and comprehensive metric evaluation.

Logistic Regression emerged as the superior model (ROC-AUC: 0.8274, F1: 55.29%, Recall: 50.36%), outperforming complex non-linear alternatives due to the dataset's fundamentally linear churn signal structure. The finding challenges the assumption that model complexity correlates with performance — for this specific dataset and class distribution, interpretable linear modeling delivered the best probability ranking quality.

The persistent challenge across all five models was class imbalance. The ~73/27 non-churn/churn split systematically suppressed recall, the most business-critical metric. Addressing this imbalance through class weighting, oversampling, or threshold optimization represents the highest-priority improvement opportunity.

The feature analysis reveals that contract length, customer tenure, and monthly charges are the dominant churn predictors — findings that directly map to actionable business interventions: contract upselling, early-tenure engagement programs, and price sensitivity management for high-value customers.

The trained model pipeline (classifier, scaler, and feature columns) has been serialized for deployment. With the recommended improvements in place — class balancing, threshold optimization, and potential XGBoost benchmarking — this churn prediction system is well-positioned for production deployment as a proactive customer retention tool.



End of Report
Classification — Churn Prediction  |  Telco Customer Dataset  |  June 2026
Telco Customer Churn Prediction — Classification Analysis Report

Classification-Machine Learning Report  |  Page 1 of 2

