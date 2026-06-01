# pyCFD 3.0 Student Lab Manual

## Lab Objective

The objective of this lab is to determine whether multiple domains are complementary before building a fusion model.

## Core Rule

Same encoder first, then CFD.

## Deliverables

Each student must submit:

1. Dataset description
2. Domain definitions
3. CFD report
4. Complementarity matrix
5. Ranking table
6. Recommended fusion strategy
7. Short interpretation

## Required Output Table

| Combination | Fusion Gain | p-value | CI Lower | CI Upper | Redundancy | CFD Index | Complementary |
|---|---:|---:|---:|---:|---:|---:|---|

## Interpretation Guide

If Complementary = True, fusion is statistically justified.

If Complementary = False, do not fuse unless there is a strong scientific reason.

## Example Conclusion

The pyCFD analysis showed that Time and Time-Frequency representations were complementary, while Frequency and Time-Frequency were redundant. Therefore, the recommended fusion strategy is Time + Time-Frequency.
