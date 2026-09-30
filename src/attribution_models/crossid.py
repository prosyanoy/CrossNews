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
        if args.load:
            with open(Path(args.load_folder) / "crossid_config.json", encoding="utf-8") as f:
                parameter_set = json.load(f)
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

        weights = np.array([self.prototype_weight, self.centroid_weight,
                            self.reference_weight], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("CROSS-ID weights must be finite, non-negative, and sum to > 0.")
        if min(self.num_prototypes, self.prototype_top_k, self.reference_top_k) < 1:
            raise ValueError("CROSS-ID prototype counts and top-k values must be positive.")
        self._weights = weights / weights.sum()

        self.random_state = int(parameter_set.get("random_state", 1234))
        self.profile_mode = parameter_set.get("profile_mode", "pooled")
        if self.profile_mode not in {"pooled", "genre_prototypes", "genre_balanced", "genre_matched"}:
            raise ValueError("Unknown CROSS-ID profile_mode.")

        self.id_to_embedding = self._load_embeddings(
            self.train_embedding_loc,
            self.test_embedding_loc,
        )

        self._author_profiles = None
        self._profile_signature = None

    @staticmethod
    def _load_embeddings(*paths):
        out = {}
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
            if not isinstance(embeddings, dict):
                raise ValueError(f"Expected a document-id to embedding object in {path}.")
            for key, value in embeddings.items():
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or not np.isfinite(np.linalg.norm(vector)) or np.linalg.norm(vector) == 0:
                    raise ValueError(f"Invalid embedding for document {key} in {path}.")
                if out and len(vector) != len(next(iter(out.values()))):
                    raise ValueError(f"Inconsistent embedding dimension for document {key}.")
                if key in out and not np.array_equal(out[key], vector):
                    raise ValueError(f"Conflicting embeddings for document {key}; reference and target IDs must not overlap.")
                out[key] = vector
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
        # __init__ restores the saved configuration before initializing the base class.
        # Profiles are rebuilt from the supplied reference documents.
        return None

    def _embedding_for_id(self, doc_id):
        key = str(doc_id)
        if key not in self.id_to_embedding:
            raise KeyError(
                f"Missing embedding for document id={key}. "
                "Generate SELMA train/test embeddings first."
            )
        return np.asarray(self.id_to_embedding[key], dtype=np.float32)

    def _make_profile(self, matrix, num_prototypes=None):
        matrix = _l2_normalize(matrix)
        n = matrix.shape[0]

        centroid = _l2_normalize(matrix.mean(axis=0))

        k = min(self.num_prototypes if num_prototypes is None else num_prototypes, n)
        # Centroid/reference ablations do not need an unused KMeans fit.
        if self.prototype_weight == 0:
            k = 1
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
        signature = tuple(zip(query_df["author"].map(int), query_df["id"].map(str),
                              query_df["genre"].map(str)))
        if self._author_profiles is not None and signature == self._profile_signature:
            return self._author_profiles

        profiles = {}
        for author in sorted(set(query_df["author"])):
            author_rows = query_df[query_df["author"] == author]
            if self.profile_mode == "pooled":
                matrix = np.stack([self._embedding_for_id(doc_id) for doc_id in author_rows["id"]])
                profiles[int(author)] = self._make_profile(matrix)
            else:
                genres = sorted(set(author_rows["genre"]))
                if not set(genres).issubset({"Article", "Tweet"}):
                    raise ValueError("Genre-aware CROSS-ID requires Article/Tweet metadata.")
                if self.num_prototypes < len(genres):
                    raise ValueError("Prototype budget must allow at least one per reference genre.")
                # Same total budget as the pooled model. Odd remainders go to
                # genres in sorted order; the shipped budgets (4/6) divide evenly.
                budgets = [self.num_prototypes // len(genres) + (i < self.num_prototypes % len(genres))
                           for i in range(len(genres))]
                groups = {
                    genre: self._make_profile(np.stack([
                        self._embedding_for_id(doc_id)
                        for doc_id in author_rows.loc[author_rows["genre"] == genre, "id"]]), budget)
                    for genre, budget in zip(genres, budgets)}
                if self.profile_mode == "genre_prototypes":
                    # Isolate prototype allocation: preserve pooled centroid,
                    # references, and top-k scoring, changing only KMeans groups.
                    matrix = np.stack([self._embedding_for_id(doc_id) for doc_id in author_rows["id"]])
                    profile = self._make_profile(matrix, num_prototypes=1)
                    profile["prototypes"] = np.concatenate([g["prototypes"] for g in groups.values()])
                    profiles[int(author)] = profile
                else:
                    profiles[int(author)] = {"genres": groups}

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

    def _score_profile(self, target, profile, target_genre=None):
        if "genres" in profile:
            groups = profile["genres"]
            if self.profile_mode == "genre_matched" and target_genre in groups:
                return self._score_profile(target, groups[target_genre])
            # Equal genre weights, independent of reference-document counts.
            # Single-genre cross-genre trials naturally fall back to that genre.
            return float(np.mean([self._score_profile(target, group) for group in groups.values()]))
        target = _l2_normalize(target)

        proto_sims = profile["prototypes"] @ target
        proto_score = self._topk_mean(proto_sims, self.prototype_top_k)

        centroid_score = float(profile["centroid"] @ target)

        ref_sims = profile["references"] @ target
        ref_score = self._topk_mean(ref_sims, self.reference_top_k)

        weights = self._weights

        return float(
            weights[0] * proto_score
            + weights[1] * centroid_score
            + weights[2] * ref_score
        )

    def evaluate_internal(self, query_df, target_df, df_name=None):
        if query_df.empty:
            raise ValueError("CROSS-ID requires at least one reference document.")
        if self.profile_mode != "pooled" and not set(target_df["genre"]).issubset({"Article", "Tweet"}):
            raise ValueError("Genre-aware CROSS-ID requires Article/Tweet target metadata.")
        overlap = set(query_df["id"].map(str)) & set(target_df["id"].map(str))
        if overlap:
            raise ValueError("Reference and target document IDs overlap; benchmark would leak data.")
        profiles = self._build_author_profiles(query_df)
        author_ids = sorted(profiles.keys())

        all_scores = []
        for _, row in target_df.iterrows():
            target = self._embedding_for_id(row["id"])
            scores = [
                self._score_profile(target, profiles[author], row["genre"])
                for author in author_ids
            ]
            all_scores.append(scores)

        return all_scores
