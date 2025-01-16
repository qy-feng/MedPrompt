# Experimental Design for Validating MedPrompt with Llama3.2-1B

## 1. Data Splitting and Selection:
- Training Split: Use 2,000 samples from the MedQA training set to generate chain-of-thought (CoT) exemplars.
- Test Split: Use the MedQA test set for final evaluation.
- Few-shot Example Selection: Implement dynamic kNN-based retrieval using a semantic embedding model (sentence-transformers/all-MiniLM-L6-v2).

## 2. Prompting Techniques Implementation:
- Dynamic Few-shot Learning: Retrieve semantically similar questions using kNN and embeddings.
- Self-Generated Chain of Thought (CoT): Generate CoT reasoning using the model itself for the selected few-shot exemplars.
- Choice Shuffling Ensemble: Shuffle answer choices multiple times and aggregate results via majority voting.

## 3. Evaluation Results and Analysis:

Primary Metrics:
- Test Accuracy: Measure accuracy across the full test set.
- Few-shot Effectiveness: Compare dynamic vs. static few-shot examples.
- Component Contribution: Ablation study to evaluate the impact of:
- Dynamic Few-shot Learning
- Self-Generated CoT
- Choice Shuffling Ensemble

Data:
- Train samples: 2000
- Test samples: 250

Ablation Study:

| Technique	| Accuracy (%)|
| -------------- | -------------- |
|Baseline (Zero-shot)|	50.4|
|Dynamic Few-shot Only|	52|
|CoT Only|	60.8|
|Choice Shuffling Only|	12|
|Full MedPrompt |	26.4|

Error Analysis:
- Common Failure Modes: Identify patterns in incorrect predictions (e.g., hallucinations, positional bias).
- Edge Cases: Examine performance on complex medical questions requiring deeper reasoning.

Potential Limitations:
- Limitation of context length and reasoning capability of Llama3.2-1b
- Compute overhead from repeated sampling
- Possible CoT hallucinations

## 4. Future Work to Improve Accuracy:

Algorithmic Enhancements:
- Expanded Few-shot Pool: Increase the number of few-shot exemplars beyond 5 to explore the impact of a larger CoT set.
- Multi-stage Reasoning: Use hierarchical CoT where intermediate reasoning steps are further refined.
- Confidence Calibration: Add a confidence scoring mechanism to flag low-certainty predictions.

Embedding Model Improvements:
- Richer Embeddings: Experiment with more powerful embedding models like text-embedding-ada-002.
- Task-specific Fine-tuning: Fine-tune the embedding model on medical QA datasets.

Data Augmentation:
- Synthetic Data: Generate synthetic CoT examples using the model to expand the training pool.
- Data Balancing: Ensure a balanced distribution of question complexity and subject areas.

Inference Optimization:
- Reducing API Calls: Limit the number of ensemble iterations while maintaining accuracy.
- Temperature Adjustment: Experiment with different temperature values for increased reasoning diversity.