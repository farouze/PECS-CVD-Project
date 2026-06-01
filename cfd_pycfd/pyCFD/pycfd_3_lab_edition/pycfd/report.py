from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd

@dataclass
class CFDResult:
    name: str
    domains: Tuple[str, ...]
    type: str
    is_complementary: bool
    hypothesis_decision: str
    fusion_gain: float
    p_value: float
    confidence_interval: Tuple[float, float]
    redundancy: float
    cfd_score: float
    cfd_index: float
    strength: str
    scores: Dict[str, float]
    metadata: Optional[Dict] = None

    def summary(self):
        return "\n".join([
            f"Combination: {self.name}",
            f"Type: {self.type}",
            f"Domains: {self.domains}",
            f"CFD Result: {'Complementary' if self.is_complementary else 'Not complementary'}",
            f"Hypothesis Decision: {self.hypothesis_decision}",
            f"Fusion Gain: {self.fusion_gain:.4f}",
            f"95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]",
            f"p-value: {self.p_value:.6f}",
            f"Redundancy: {self.redundancy:.4f}",
            f"CFD Score: {self.cfd_score:.4f}",
            f"CFD Index: {self.cfd_index:.2f}",
            f"Strength: {self.strength}",
            f"Scores: {self.scores}",
            f"Metadata: {self.metadata}",
        ])

@dataclass
class CFDDiscoveryReport:
    pairwise_results: Dict[str, CFDResult]
    full_result: Optional[CFDResult]
    matrix: pd.DataFrame
    ranking: pd.DataFrame
    recommended_strategy: str
    metadata: Optional[Dict] = None

    def summary(self):
        lines = []
        lines.append("=" * 78)
        lines.append("pyCFD 3.0 SAME-ENCODER COMPLEMENTARY FEATURE DOMAIN REPORT")
        lines.append("=" * 78)

        lines.append("\nPAIRWISE CFD TESTS")
        for name, res in self.pairwise_results.items():
            lines.append("-" * 78)
            lines.append(name)
            lines.append(f"Complementary: {res.is_complementary}")
            lines.append(f"Fusion Gain: {res.fusion_gain:.4f}")
            lines.append(f"p-value: {res.p_value:.6f}")
            lines.append(f"CI: [{res.confidence_interval[0]:.4f}, {res.confidence_interval[1]:.4f}]")
            lines.append(f"Redundancy: {res.redundancy:.4f}")
            lines.append(f"CFD Index: {res.cfd_index:.2f}")
            lines.append(f"Strength: {res.strength}")

        if self.full_result is not None:
            lines.append("\nFULL-DOMAIN FUSION TEST")
            lines.append("-" * 78)
            res = self.full_result
            lines.append(f"Domains: {res.domains}")
            lines.append(f"Complementary: {res.is_complementary}")
            lines.append(f"Fusion Gain: {res.fusion_gain:.4f}")
            lines.append(f"p-value: {res.p_value:.6f}")
            lines.append(f"CI: [{res.confidence_interval[0]:.4f}, {res.confidence_interval[1]:.4f}]")
            lines.append(f"Redundancy: {res.redundancy:.4f}")
            lines.append(f"CFD Index: {res.cfd_index:.2f}")
            lines.append(f"Strength: {res.strength}")

        lines.append("\nCFD COMPLEMENTARITY MATRIX")
        lines.append(str(self.matrix.round(3)))

        lines.append("\nRANKING")
        lines.append(str(self.ranking.round(4)))

        lines.append("\nRECOMMENDED STRATEGY")
        lines.append(self.recommended_strategy)

        if self.metadata:
            lines.append("\nMETADATA")
            lines.append(str(self.metadata))

        return "\n".join(lines)

    def to_csv(self, path):
        self.ranking.to_csv(path, index=False)
        return path
