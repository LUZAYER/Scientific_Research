# Benchmark Comparison Results

| dataset          | model                      |   accuracy |   precision |   recall |   f1_score |   roc_auc |   false_positive_rate |   false_negative_rate |   avg_latency_sec |
|:-----------------|:---------------------------|-----------:|------------:|---------:|-----------:|----------:|----------------------:|----------------------:|------------------:|
| HaluEval-QA      | Confidence-Only (B1)       |        0.5 |      0.5    |   1      |     0.6667 |    0.5    |                     1 |                0      |            0.0003 |
| HaluEval-QA      | Retrieval-Only (B2)        |        0.5 |      0.5    |   1      |     0.6667 |    1      |                     1 |                0      |            0.0037 |
| HaluEval-QA      | Semantic-Only (B3)         |        0.5 |      0.5    |   1      |     0.6667 |    0.6    |                     1 |                0      |            0.0036 |
| HaluEval-QA      | SelfVerification-Only (B4) |        0.5 |      0.5    |   1      |     0.6667 |    0.5    |                     1 |                0      |            0.0001 |
| HaluEval-QA      | Proposed Hybrid Framework  |        0.5 |      0.5    |   1      |     0.6667 |    0.88   |                     1 |                0      |            0.0061 |
| TruthfulQA       | Confidence-Only (B1)       |        0.5 |      0.5    |   1      |     0.6667 |    0.5    |                     1 |                0      |            0.0002 |
| TruthfulQA       | Retrieval-Only (B2)        |        0.5 |      0.5    |   1      |     0.6667 |    0.32   |                     1 |                0      |            0.0028 |
| TruthfulQA       | Semantic-Only (B3)         |        0.5 |      0.5    |   1      |     0.6667 |    0.5    |                     1 |                0      |            0.003  |
| TruthfulQA       | SelfVerification-Only (B4) |        0.5 |      0.5    |   1      |     0.6667 |    0.5    |                     1 |                0      |            0.0001 |
| TruthfulQA       | Proposed Hybrid Framework  |        0.5 |      0.5    |   1      |     0.6667 |    0.32   |                     1 |                0      |            0.0058 |
| WikiBio-GPT3     | Confidence-Only (B1)       |        1   |      1      |   1      |     1      |    0.5    |                     0 |                0      |            0.0014 |
| WikiBio-GPT3     | Retrieval-Only (B2)        |        1   |      1      |   1      |     1      |    0.5    |                     0 |                0      |            0.0372 |
| WikiBio-GPT3     | Semantic-Only (B3)         |        1   |      1      |   1      |     1      |    0.5    |                     0 |                0      |            0.0411 |
| WikiBio-GPT3     | SelfVerification-Only (B4) |        1   |      1      |   1      |     1      |    0.5    |                     0 |                0      |            0.0994 |
| WikiBio-GPT3     | Proposed Hybrid Framework  |        1   |      1      |   1      |     1      |    0.5    |                     0 |                0      |            0.1774 |
| FavaMultiSamples | Confidence-Only (B1)       |        0.7 |      0.7    |   1      |     0.8235 |    0.5    |                     1 |                0      |            0.0012 |
| FavaMultiSamples | Retrieval-Only (B2)        |        0.7 |      0.7    |   1      |     0.8235 |    0.5    |                     1 |                0      |            0.0006 |
| FavaMultiSamples | Semantic-Only (B3)         |        0.7 |      0.7    |   1      |     0.8235 |    0.5    |                     1 |                0      |            0.0005 |
| FavaMultiSamples | SelfVerification-Only (B4) |        0.6 |      0.6667 |   0.8571 |     0.75   |    0.4286 |                     1 |                0.1429 |            0.0622 |
| FavaMultiSamples | Proposed Hybrid Framework  |        0.7 |      0.7    |   1      |     0.8235 |    0.4286 |                     1 |                0      |            0.0523 |