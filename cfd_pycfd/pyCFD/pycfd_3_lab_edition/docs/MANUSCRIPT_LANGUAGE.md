# Manuscript Language

## Methods

Complementarity was evaluated using pyCFD 3.0. To isolate domain complementarity from architectural bias, each domain was encoded using the same neural architecture and identical training configuration. Domain-specific encoders were trained independently, and fusion models were trained on concatenated latent embeddings. Complementarity was assessed using fusion gain, bootstrap confidence intervals, paired permutation testing, and redundancy-adjusted CFD Index.

## Results

A domain combination was considered complementary if fusion gain exceeded 0.01, the p-value was below 0.05, and the lower confidence bound was positive.

## Interpretation

pyCFD identifies whether a domain combination is statistically justified for fusion. A non-complementary result does not imply a domain is useless; it indicates that the fusion model did not add reliable information beyond the strongest individual domain under the tested configuration.
