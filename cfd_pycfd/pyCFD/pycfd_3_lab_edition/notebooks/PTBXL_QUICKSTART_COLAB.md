# PTB-XL Colab Quickstart

This is a notebook-style guide.

```python
!unzip -q pycfd_3_lab_edition.zip
%cd pycfd_3_lab_edition
!pip install -e . --no-deps
!pip install wfdb
```

```python
from pycfd.biosignal import load_ptbxl_signals, make_ptbxl_norm_labels
from pycfd.biosignal import cfd_biosignal_same_encoder_discover

PTBXL_PATH = "/content/drive/MyDrive/datasets/ptbxl_full"

signals, df = load_ptbxl_signals(PTBXL_PATH, max_records=1000)
y = make_ptbxl_norm_labels(df)

report = cfd_biosignal_same_encoder_discover(signals, y)
print(report.summary())
```
