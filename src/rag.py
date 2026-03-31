import pandas as pd
from typing import Optional, Tuple, List, Dict

class PollingRAG:
    """
    TECHNIQUE 1: RAG
    Retrieval-Augmented Generation for Polls
    """

    def __init__(self, csv_path: str, top_k: int = 5):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load polling CSV at {csv_path}: {e}")

        self.top_k = top_k

        if "snapshot_date" not in self.df.columns:
            raise ValueError("Polling CSV missing 'snapshot_date' column.")
        if "race_id" not in self.df.columns:
            raise ValueError("Polling CSV missing 'race_id' column.")

        self.df["snapshot_date"] = pd.to_datetime(self.df["snapshot_date"])

    def retrieve_polls(self, race_id: str, as_of_date: Optional[str] = None) -> pd.DataFrame:
        sub = self.df[self.df["race_id"] == race_id].copy()

        if sub.empty:
            return sub

        if as_of_date is not None:
            cutoff = pd.to_datetime(as_of_date)
            sub = sub[sub["snapshot_date"] <= cutoff]

        sub = sub.sort_values("snapshot_date", ascending=False).head(self.top_k)
        return sub

    def compute_poll_implied_prob(self, race_id: str, as_of_date: Optional[str] = None, candidate_support_col: str = "candidate1_support_pct") -> Tuple[Optional[float], List[Dict]]:
        polls = self.retrieve_polls(race_id, as_of_date)

        if polls.empty or candidate_support_col not in polls.columns:
            return None, []

        probs = polls[candidate_support_col].astype(float) / 100.0
        poll_prob = float(probs.mean())

        cols_to_keep = [
            "race_id", "snapshot_date", "pollster", "sample_size",
            "candidate1_name", "candidate1_support_pct",
            "candidate2_name", "candidate2_support_pct",
        ]
        cols_to_keep = [c for c in cols_to_keep if c in polls.columns]
        evidence = polls[cols_to_keep].to_dict(orient="records")

        # Convert timestamps to string to make JSON serializable
        for item in evidence:
            item["snapshot_date"] = str(item["snapshot_date"])

        return poll_prob, evidence
