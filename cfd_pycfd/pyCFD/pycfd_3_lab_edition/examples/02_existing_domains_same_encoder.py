from pycfd.deep import cfd_same_encoder_discover

domains = {
    "Time": X_time,
    "Frequency": X_freq,
    "TimeFrequency": X_tf,
}

report = cfd_same_encoder_discover(domains, y)
print(report.summary())
