import numpy as np
from scipy.signal import resample, stft
from scipy.stats import entropy

def ensure_3d(signals):
    x = np.asarray(signals)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("signals must be (samples, timesteps) or (samples, timesteps, channels)")
    return x

def _select_channels(x, channels=None):
    if channels is None:
        return x
    return x[:, :, channels]

def extract_time_domain(signals, n_time_points=None, channels=None):
    x = ensure_3d(signals)
    x = _select_channels(x, channels)
    if n_time_points is not None:
        return np.asarray([resample(sample, n_time_points, axis=0).reshape(-1) for sample in x])
    return x.reshape(x.shape[0], -1)

def extract_frequency_domain(signals, n_fft=256, channels=None, keep_dc=True):
    x = ensure_3d(signals)
    x = _select_channels(x, channels)
    mag = np.abs(np.fft.rfft(x, n=n_fft, axis=1))
    if not keep_dc:
        mag = mag[:, 1:, :]
    return mag.reshape(mag.shape[0], -1)

def extract_time_frequency_domain(signals, nperseg=64, features_per_channel=100, channels=None):
    x = ensure_3d(signals)
    x = _select_channels(x, channels)
    feats = []
    for sample in x:
        sample_feats = []
        for c in range(sample.shape[1]):
            _, _, Z = stft(sample[:, c], nperseg=nperseg)
            mag = np.abs(Z).reshape(-1)
            if features_per_channel is not None and len(mag) != features_per_channel:
                mag = resample(mag, features_per_channel)
            sample_feats.extend(mag)
        feats.append(sample_feats)
    return np.asarray(feats)

def extract_statistical_domain(signals, channels=None, bins=30):
    x = ensure_3d(signals)
    x = _select_channels(x, channels)
    feats = []
    for sample in x:
        sample_feats = []
        for c in range(sample.shape[1]):
            s = sample[:, c]
            hist, _ = np.histogram(s, bins=bins, density=True)
            hist = hist + 1e-12
            sample_feats.extend([
                np.mean(s), np.std(s), np.min(s), np.max(s),
                np.median(s), np.percentile(s, 25), np.percentile(s, 75),
                np.mean(np.abs(s)), np.sum(s ** 2) / len(s), entropy(hist),
            ])
        feats.append(sample_feats)
    return np.asarray(feats)

def extract_biosignal_domains(
    signals,
    domains=("time", "frequency", "time_frequency"),
    n_time_points=None,
    n_fft=256,
    nperseg=64,
    tf_features_per_channel=100,
    channels=None,
):
    extracted = {}
    for d in domains:
        key = d.lower().replace("-", "_")
        if key in {"time", "t"}:
            extracted["Time"] = extract_time_domain(signals, n_time_points=n_time_points, channels=channels)
        elif key in {"frequency", "freq", "f"}:
            extracted["Frequency"] = extract_frequency_domain(signals, n_fft=n_fft, channels=channels)
        elif key in {"time_frequency", "tf", "timefrequency"}:
            extracted["TimeFrequency"] = extract_time_frequency_domain(signals, nperseg=nperseg, features_per_channel=tf_features_per_channel, channels=channels)
        elif key in {"statistical", "stats"}:
            extracted["Statistical"] = extract_statistical_domain(signals, channels=channels)
        else:
            raise ValueError(f"Unknown biosignal domain: {d}")
    return extracted
