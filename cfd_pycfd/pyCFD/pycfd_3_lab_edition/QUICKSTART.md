# pyCFD 3.0 Quickstart

## Install

```python
!unzip -q pycfd_3_lab_edition.zip
%cd pycfd_3_lab_edition
!pip install -e . --no-deps
```

## Verify

```python
import pycfd
print(pycfd.__version__)
```

Expected:

```text
3.0.0
```

## PTB-XL Example

```python
from pycfd.biosignal import load_ptbxl_signals, make_ptbxl_norm_labels
from pycfd.biosignal import cfd_biosignal_same_encoder_discover

PTBXL_PATH = "/content/drive/MyDrive/datasets/ptbxl_full"

signals, df = load_ptbxl_signals(PTBXL_PATH, max_records=1000)
y = make_ptbxl_norm_labels(df)

report = cfd_biosignal_same_encoder_discover(signals, y)
print(report.summary())
```

## Custom Domain Example

```python
from pycfd.deep import cfd_same_encoder_discover

domains = {
    "DomainA": X_a,
    "DomainB": X_b,
    "DomainC": X_c
}

report = cfd_same_encoder_discover(domains, y)
print(report.summary())
```
