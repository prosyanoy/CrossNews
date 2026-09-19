import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from attribution_models.attribution_model import AttributionModel


def _l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = max(float(np.linalg.norm(x)), eps)
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


class CrossID(AttributionModel):
    """
    Phase-1 CROSS-ID model for CROSSNEWS.

    Main difference from SELMA:
      * SELMA represents each author by a single mean embedding.
      * CROSS-ID keeps a small set of author prototypes and the original
        reference bundle, then combines prototype-, centroid-, and local
        top-reference similarity.

    This is intentionally a drop-in, training-free first milestone. It reuses
    the SELMA embedding files produced by generate_selma_embeddings.py.
    """

    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)

        self.train_embedding_loc = parameter_set["train_embedding_loc"]
        self.test_embedding_loc = parameter_set["test_embedding_loc"]

        self.num_prototypes = int(parameter_set.get("num_prototypes", 4))
        self.prototype_top_k = int(parameter_set.get("prototype_top_k", 2))
        self.reference_top_k = int(parameter_set.get("reference_top_k", 4))

        # Mixture weights. They are normalized at scoring time, so users can
        # set any non-negative magnitudes.
        self.prototype_weight = float(parameter_set.get("prototype_weight", 0.55))
        self.centroid_weight = float(parameter_set.get("centroid_weight", 0.20))
        self.reference_weight = float(parameter_set.get("reference_weight", 0.25))

        self.random_state = int(parameter_set.get("random_state", 1234))

        self.id_to_embedding = self._load_embeddings(
            self.train_embedding_loc,
            self.test_embedding_loc,
        )

        self._author_profiles = None
        self._profile_signature = None

    def _load_embeddings(self, *paths):
        out = {}
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                out.update(json.load(f))
        return out

    def get_model_name(self):
        return self.parameter_set.get("name", "crossid")

    def train_internal(self, params):
        # Phase 1 is training-free. Later phases will replace this with
        # topic/genre-invariant contrastive training on the silver split.
        return None

    def save_model(self, folder):
        os.makedirs(folder, exist_ok=True)
        cfg_path = Path(folder) / "crossid_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.parameter_set, f, indent=2)

    def load_model(self, folder):
        # Embedding locations and hyperparameters are read from the model
        # parameter file, as in SELMA. Nothing trainable is persisted yet.
        return None

    def _embedding_for_id(self, doc_id):
        key = str(doc_id)
        if key not in self.id_to_embedding:
            raise KeyError(
                f"Missing embedding for document id={key}. "
                "Generate SELMA train/test embeddings first."
            )
        return np.asarray(self.id_to_embedding[key], dtype=np.float32)

    def _make_profile(self, matrix):
        matrix = _l2_normalize(matrix)
        n = matrix.shape[0]

        centroid = _l2_normalize(matrix.mean(axis=0))

        k = min(max(self.num_prototypes, 1), n)
        if k == 1:
            prototypes = centroid[None, :]
        elif k == n:
            # No need to fit KMeans when every reference can be a prototype.
            prototypes = matrix.copy()
        else:
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            km.fit(matrix)
            prototypes = _l2_normalize(km.cluster_centers_)

        return {
            "centroid": centroid,
            "prototypes": prototypes,
            "references": matrix,
        }

    def _build_author_profiles(self, query_df):
        # Cache profiles for a given query dataframe shape/content signature.
        signature = (
            len(query_df),
            tuple(sorted(map(int, set(query_df["author"])))),
            tuple(map(str, query_df["id"].tolist()[:16])),
        )
        if self._author_profiles is not None and signature == self._profile_signature:
            return self._author_profiles

        profiles = {}
        for author in sorted(set(query_df["author"])):
            author_rows = query_df[query_df["author"] == author]
            matrix = np.stack(
                [self._embedding_for_id(doc_id) for doc_id in author_rows["id"]],
                axis=0,
            )
            profiles[int(author)] = self._make_profile(matrix)

        self._author_profiles = profiles
        self._profile_signature = signature
        return profiles

    @staticmethod
    def _topk_mean(values, k):
        values = np.asarray(values, dtype=np.float32)
        k = max(1, min(int(k), values.shape[0]))
        if k == values.shape[0]:
            return float(values.mean())
        idx = np.argpartition(values, -k)[-k:]
        return float(values[idx].mean())

    def _score_profile(self, target, profile):
        target = _l2_normalize(target)

        proto_sims = profile["prototypes"] @ target
        proto_score = self._topk_mean(proto_sims, self.prototype_top_k)

        centroid_score = float(profile["centroid"] @ target)

        ref_sims = profile["references"] @ target
        ref_score = self._topk_mean(ref_sims, self.reference_top_k)

        weights = np.asarray(
            [
                self.prototype_weight,
                self.centroid_weight,
                self.reference_weight,
            ],
            dtype=np.float32,
        )
        if np.any(weights < 0):
            raise ValueError("CROSS-ID mixture weights must be non-negative.")
        if float(weights.sum()) == 0.0:
            raise ValueError("At least one CROSS-ID mixture weight must be > 0.")
        weights = weights / weights.sum()

        return float(
            weights[0] * proto_score
            + weights[1] * centroid_score
            + weights[2] * ref_score
        )

    def evaluate_internal(self, query_df, target_df, df_name=None):
        profiles = self._build_author_profiles(query_df)
        author_ids = sorted(profiles.keys())

        all_scores = []
        for _, row in target_df.iterrows():
            target = self._embedding_for_id(row["id"])
            scores = [
                self._score_profile(target, profiles[author])
                for author in author_ids
            ]
            all_scores.append(scores)

        return all_scores
