from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import ot  # Python Optimal Transport library


# =========================
# Transfer Sampling Class
# =========================
class TransferSampling:
    """
    Implements the Transfer Sampling pipeline:
      1. Fit SVM on source features & labels with CV tuning
      2. Compute OT alignment between source and target
      3. Compute transferred scores for target samples
      4. Select top-k target indices
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[float]]] = None,
        cv: int = 5,
        ot_reg: float = 1e-1,
        scoring: str = "f1",
        n_jobs: int = -1,
    ):
        """Initialize the TransferSampling pipeline.
        Args:
            param_grid (Optional[Dict[str, List[float]]]): Grid for SVM hyperparameters.
            cv (int): Number of cross-validation folds.
            ot_reg (float): Regularization for optimal transport.
            scoring (str): Scoring metric for SVM.
            n_jobs (int): Number of parallel jobs for grid search.
        """
        self.param_grid = param_grid or {"svm__C": [0.01, 0.1, 1, 10]}
        self.cv = cv
        self.ot_reg = ot_reg
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.svm_pipeline = None
        self.source_feats = None

    def fit_source(self, feats: np.ndarray, labels: np.ndarray) -> None:
        """Fit the SVM on source features and labels with cross-validation.
        Args:
            feats (np.ndarray): Source features.
            labels (np.ndarray): Source labels.
        """
        self.source_feats = feats
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("svm", LinearSVC(max_iter=10000))]
        )
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        grid.fit(feats, labels)
        print(f"Best SVM params: {grid.best_params_}")
        self.svm_pipeline = grid.best_estimator_

    def compute_transport(self, target_feats: np.ndarray) -> np.ndarray:
        """Compute the optimal transport plan between source and target features.
        Args:
            target_feats (np.ndarray): Target features.
        Returns:
            np.ndarray: Optimal transport matrix.
        """
        if self.source_feats is None:
            raise ValueError("Source not fitted")
        a = np.ones((self.source_feats.shape[0],)) / self.source_feats.shape[0]
        b = np.ones((target_feats.shape[0],)) / target_feats.shape[0]
        M = ot.dist(self.source_feats, target_feats, metric="euclidean")
        return ot.sinkhorn(a, b, M, self.ot_reg)

    def score_target(self, target_feats: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Score target samples using the transferred SVM decision function.
        Args:
            target_feats (np.ndarray): Target features.
            T (np.ndarray): Optimal transport matrix.
        Returns:
            np.ndarray: Scores for target samples.
        """
        if self.svm_pipeline is None:
            raise ValueError("SVM not fitted")
        s = self.svm_pipeline.decision_function(self.source_feats)
        mask = (T > 0).astype(float)
        raw = mask.T.dot(s)
        counts = mask.sum(axis=0) + 1e-8
        return raw / counts

    def select_top_k(self, scores: np.ndarray, k: int) -> List[int]:
        """Select indices of the top-k scoring target samples.
        Args:
            scores (np.ndarray): Scores for target samples.
            k (int): Number of top samples to select.
        Returns:
            List[int]: Indices of top-k samples.
        """
        return list(np.argsort(scores)[-k:][::-1])

    def run(
        self,
        source_feats: np.ndarray,
        source_labels: np.ndarray,
        target_feats: np.ndarray,
        k: int,
    ) -> List[int]:
        """Run the full transfer sampling pipeline and return top-k indices.
        Args:
            source_feats (np.ndarray): Source features.
            source_labels (np.ndarray): Source labels.
            target_feats (np.ndarray): Target features.
            k (int): Number of top samples to select.
        Returns:
            List[int]: Indices of top-k samples.
        """
        self.fit_source(source_feats, source_labels)
        T = self.compute_transport(target_feats)
        scores = self.score_target(target_feats, T)
        return self.select_top_k(scores, k)
