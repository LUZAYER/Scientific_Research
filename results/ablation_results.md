# Systematic Ablation Study Results

| configuration               |   f1_score |   precision |   recall |   accuracy |   roc_auc |   avg_latency_sec |
|:----------------------------|-----------:|------------:|---------:|-----------:|----------:|------------------:|
| Full Hybrid (C + R + S + V) |     0.6667 |         0.5 |        1 |        0.5 |     0.84  |            0.4208 |
| Hybrid – Confidence         |     0.6667 |         0.5 |        1 |        0.5 |     0.79  |            0.4591 |
| Hybrid – Retrieval          |     0.6667 |         0.5 |        1 |        0.5 |     0.535 |            0.0044 |
| Hybrid – Semantic           |     0.6667 |         0.5 |        1 |        0.5 |     0.96  |            0.0319 |
| Hybrid – Self-Verification  |     0.6667 |         0.5 |        1 |        0.5 |     0.84  |            0.0532 |