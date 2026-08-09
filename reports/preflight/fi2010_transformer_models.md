# FI-2010 Transformer Models — Preflight Evidence

**Run kind**: smoke/preflight — NOT confirmatory

All timing and memory measurements are diagnostic only.

## TRANSLOB

### Architecture
- Parameter count: 101895
- Input shape: [2, 1, 100, 40]
- Output shape: [2, 3]
- Forward passed: True
- Backward passed: True
- Nonzero gradient: True

### CUDA
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- Forward/backward passed: True
- Peak GPU memory: 19464704 bytes

### Tiny-Overfit (Diagnostic Only)
- LR: 0.001 (diagnostic-only)
- Initial loss: 1.186334252357483
- Final loss: 1.3038341421633959e-05
- Initial accuracy: 0.625
- Final accuracy: 1.0
- Pass/fail: pass

### Resume Determinism
- Max parameter delta: 0.0
- Prediction match: True
- Pass/fail: pass

### Throughput (Diagnostic Only)
- Batch size: 32
- Samples/sec: 6094
- Inference latency: 0.164 ms
- Note: diagnostic only — hardware/run dependent — not a scientific acceptance metric

## TLOB

### Architecture
- Parameter count: 734478
- Input shape: [2, 1, 100, 40]
- Output shape: [2, 3]
- Forward passed: True
- Backward passed: True
- Nonzero gradient: True

### CUDA
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- Forward/backward passed: True
- Peak GPU memory: 26408960 bytes

### Tiny-Overfit (Diagnostic Only)
- LR: 0.001 (diagnostic-only)
- Initial loss: 1.0691826343536377
- Final loss: 0.0004636024823412299
- Initial accuracy: 0.625
- Final accuracy: 1.0
- Pass/fail: pass

### Resume Determinism
- Max parameter delta: 0.0
- Prediction match: True
- Pass/fail: pass

### Throughput (Diagnostic Only)
- Batch size: 32
- Samples/sec: 1363
- Inference latency: 0.734 ms
- Note: diagnostic only — hardware/run dependent — not a scientific acceptance metric
