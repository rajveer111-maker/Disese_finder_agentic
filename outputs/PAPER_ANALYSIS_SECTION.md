# Agentic Disease Finder: Experimental Results & Analysis

## Motor Imagery Classification (BCI2A)

The non-quantum agentic system achieved a **test accuracy of 89.00%** on the BCI2A dataset, with a weighted F1-score of 0.8899. The system demonstrated strongest performance in detecting **Foot** (accuracy: 93.3%), while **Right Hand** showed the most room for improvement (85.3%).

Confusion matrix analysis reveals specific misclassification patterns. The most significant confusion occurred between **Right Hand** being misclassified as **Tongue**, suggesting shared feature characteristics in the EEG frequency bands for these motor tasks.

## Parkinson's Disease Detection

In the binary classification task for Parkinson's Disease detection, the system achieved an accuracy of **84.67%**. This high level of performance indicates the robustness of the agentic selection mechanism in identifying pathological EEG signatures.

## Overall System Effectiveness

The agentic framework successfully integrated multi-modal decision making. By automatically selecting the appropriate model based on input characteristics, the system maintains high performance across diverse medical tasks without manual intervention. The consistent accuracy across both motor imagery and disease detection validates the scalability of the architecture.