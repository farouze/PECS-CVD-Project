# PECS-CVD-Project
# pyCFD 3.0 Lab Edition

## Complementary Feature Domain Discovery for Biomedical and Multimodal Learning

**pyCFD** is a student-ready Python library for testing whether multiple data domains are truly complementary before performing fusion.

The central rule is:

> Same encoder first, then CFD.

This prevents model-architecture differences from being confused with domain complementarity.

---

## 1. What pyCFD Answers

pyCFD answers:

> Should these domains be fused?

It does **not** assume that more domains are always better.

For example, pyCFD can test:

```text
ECG Time vs ECG Frequency
ECG Time vs ECG Time-Frequency
Frequency vs Time-Frequency
ECG + PPG
ECG + PPG + BP
Text + Image + Audio
```

---

## 2. Key CFD Concepts

### Fusion Gain

How much the fusion model improves over the best individual domain.

### Redundancy

How similar the domain predictions are.

### CFD Score

```text
CFD Score = Fusion Gain × (1 - Redundancy)
```

### CFD Index

```text
CFD Index = 100 × CFD Score
```

### Decision Rule

A domain combination is complementary when:

```text
Fusion Gain > 0.01
p-value < 0.05
CI lower bound > 0
```

---

## 3. Installation in Google Colab

Upload:

```text
pycfd_3_lab_edition.zip
```

Then run:

```python
!unzip -q pycfd_3_lab_edition.zip
%cd pycfd_3_lab_edition
!pip install -e . --no-deps
```

Verify:

```python
import pycfd
print(pycfd.__version__)
```

Expected:

```text
3.0.0
```

---

## 4. Install Optional Dependencies

For PTB-XL:

```python
!pip install wfdb
```

For same-encoder deep learning:

```python
!pip install tensorflow
```

Most Colab runtimes already include TensorFlow.

---

## 5. Quickstart: PTB-XL ECG

```python
from pycfd.biosignal import (
    load_ptbxl_signals,
    make_ptbxl_norm_labels,
    cfd_biosignal_same_encoder_discover
)

PTBXL_PATH = "/content/drive/MyDrive/datasets/ptbxl_full"

signals, df = load_ptbxl_signals(
    PTBXL_PATH,
    max_records=1000
)

y = make_ptbxl_norm_labels(df)

report = cfd_biosignal_same_encoder_discover(
    signals=signals,
    y=y,
    signal_name="PTBXL_ECG",
    domains=("time", "frequency", "time_frequency"),
    epochs=50,
    batch_size=32,
    patience=8,
    n_bootstrap=200,
    n_permutations=200
)

print(report.summary())
```

---

## 6. Quickstart: Already Extracted Domains

Use this when you already have:

```python
X_time
X_frequency
X_timefrequency
y
```

Run:

```python
from pycfd.deep import cfd_same_encoder_discover

domains = {
    "Time": X_time,
    "Frequency": X_frequency,
    "TimeFrequency": X_timefrequency
}

report = cfd_same_encoder_discover(
    domains=domains,
    y=y,
    epochs=50
)

print(report.summary())
```

---

## 7. Quickstart: Prediction-Based CFD

Use this if your models are already trained.

```python
from pycfd import cfd_test_predictions

result = cfd_test_predictions(
    y_true=y_test,
    y_a=time_pred,
    y_b=freq_pred,
    y_fusion=fusion_pred,
    name_a="Time",
    name_b="Frequency"
)

print(result.summary())
```

---

## 8. Plotting

```python
from pycfd import plot_cfd_matrix, plot_cfd_ranking

plot_cfd_matrix(report)
plot_cfd_ranking(report)
```

---

## 9. Student Assignment

Each student should:

1. Select a dataset.
2. Define candidate domains.
3. Run same-encoder pyCFD.
4. Identify complementary and redundant domains.
5. Build a final fusion model using only complementary domains.
6. Report Fusion Gain, CI, p-value, Redundancy, and CFD Index.

---

## 10. Recommended Manuscript Statement

> To isolate domain complementarity from architectural bias, all domains were encoded using the same neural architecture and identical training configuration. Complementarity was evaluated using pyCFD with bootstrap confidence intervals and paired permutation testing. A domain pair was considered complementary when fusion gain exceeded the minimum practical threshold, the p-value was below 0.05, and the lower confidence bound was positive.

---

## 11. Package Structure

```text
pycfd_3_lab_edition/
│
├── pyproject.toml
├── README.md
├── QUICKSTART.md
├── STUDENT_LAB_MANUAL.md
│
├── pycfd/
│   ├── __init__.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── redundancy.py
│   ├── report.py
│   ├── plotting.py
│   ├── test.py
│   │
│   ├── deep/
│   │   └── same_encoder.py
│   │
│   └── biosignal/
│       ├── domains.py
│       ├── discover.py
│       └── ptbxl.py
│
├── examples/
├── notebooks/
├── tests/
└── docs/
```
