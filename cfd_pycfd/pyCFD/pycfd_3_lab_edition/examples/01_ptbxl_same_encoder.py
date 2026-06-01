from pycfd.biosignal import load_ptbxl_signals, make_ptbxl_norm_labels
from pycfd.biosignal import cfd_biosignal_same_encoder_discover

PTBXL_PATH = "/content/drive/MyDrive/datasets/ptbxl_full"

signals, df = load_ptbxl_signals(PTBXL_PATH, max_records=1000)
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
    n_permutations=200,
)

print(report.summary())
