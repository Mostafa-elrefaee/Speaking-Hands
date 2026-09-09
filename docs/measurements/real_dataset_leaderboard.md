### Training / evaluation metrics (held-out test split; batch=1 model latency: tflite_p50_ms = TFLite interpreter as run by app.py, latency_p50_ms = Keras tf.function)

| config | runs | test_accuracy | test_f1_macro | test_f1_weighted | tflite_p50_ms | latency_p50_ms | param_count | train_time_s |
|---|---|---|---|---|---|---|---|---|
| mlp | mlp ★ | 2 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.020 ± 0.000 | 0.59 ± 0.01 | 333090 | 9.6 ± 0.1 |
| gru | gru | 2 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.277 ± 0.013 | 2.50 ± 0.02 | 55906 | 12.9 ± 0.5 |
| lstm | lstm | 2 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.301 ± 0.000 | 2.19 ± 0.03 | 73314 | 11.9 ± 0.3 |
| gru_lstm | gru_lstm | 2 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.278 ± 0.010 | 2.31 ± 0.07 | 63970 | 12.6 ± 1.0 |

★ = Pareto-optimal on (test accuracy ↑, TFLite model latency ↓)
