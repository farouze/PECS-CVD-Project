from pathlib import Path
import ast
import numpy as np
import pandas as pd

def load_ptbxl_signals(ptbxl_path, max_records=1000, sampling_rate=100):
    try:
        import wfdb
    except ImportError as e:
        raise ImportError("Install wfdb first: pip install wfdb") from e

    ptbxl_path = Path(ptbxl_path)
    metadata = ptbxl_path / "ptbxl_database.csv"
    if not metadata.exists():
        raise FileNotFoundError(f"Could not find {metadata}")

    df = pd.read_csv(metadata)
    if max_records is not None:
        df = df.iloc[:max_records].copy()

    filename_col = "filename_lr" if sampling_rate == 100 and "filename_lr" in df.columns else "filename_hr"
    if filename_col not in df.columns:
        raise ValueError("PTB-XL metadata must contain filename_lr or filename_hr")

    signals = []
    kept = []
    for i, rec in enumerate(df[filename_col]):
        try:
            sig, _ = wfdb.rdsamp(str(ptbxl_path / rec))
            signals.append(sig)
            kept.append(i)
        except Exception:
            continue

    if len(signals) == 0:
        raise RuntimeError("No PTB-XL signals loaded. Check PTBXL_PATH and records folders.")

    min_len = min(s.shape[0] for s in signals)
    signals = np.asarray([s[:min_len] for s in signals])
    df = df.iloc[kept].reset_index(drop=True)
    return signals, df

def make_ptbxl_norm_labels(df):
    df = df.copy()
    df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)
    return np.asarray([1 if "NORM" in d else 0 for d in df["scp_dict"]])
