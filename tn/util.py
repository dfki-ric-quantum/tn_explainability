from dataclasses import dataclass


@dataclass
class AUCROC:
    """Data class to store AUC ROC statistics"""
    train: tuple[float, float]
    test: tuple[float, float]
