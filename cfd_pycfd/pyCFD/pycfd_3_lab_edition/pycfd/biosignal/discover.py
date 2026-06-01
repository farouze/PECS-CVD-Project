from .domains import extract_biosignal_domains
from ..deep import cfd_same_encoder_discover

def cfd_biosignal_same_encoder_discover(
    signals,
    y,
    signal_name="BioSignal",
    domains=("time", "frequency", "time_frequency"),
    n_time_points=None,
    n_fft=256,
    nperseg=64,
    tf_features_per_channel=100,
    channels=None,
    **kwargs,
):
    domain_data = extract_biosignal_domains(
        signals=signals,
        domains=domains,
        n_time_points=n_time_points,
        n_fft=n_fft,
        nperseg=nperseg,
        tf_features_per_channel=tf_features_per_channel,
        channels=channels,
    )

    report = cfd_same_encoder_discover(domain_data, y, **kwargs)

    if report.metadata is None:
        report.metadata = {}
    report.metadata.update({
        "signal_name": signal_name,
        "biosignal_domains": list(domain_data.keys()),
        "domain_shapes": {k: v.shape for k, v in domain_data.items()},
    })
    return report
