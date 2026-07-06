"""
GapFinder Pipeline
------------------
Accepts paths to two Excel files (exam responses + answer key),
runs the full diagnostic pipeline, and returns JSON-serialisable results.
Uses the trained QNN if model weights are supplied; falls back to
proportion-correct mastery estimation otherwise.
"""

import re
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional PyTorch import ────────────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    class QNN(nn.Module):
        def __init__(self, n_questions, n_outcomes, q_matrix_tensor,
                     hidden_dim=64, dropout_rate=0.3):
            super().__init__()
            self.register_buffer("q_mask", q_matrix_tensor)
            self.hidden = nn.Sequential(
                nn.Linear(n_questions, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            )
            self.output_layer = nn.Linear(hidden_dim, n_outcomes)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.output_layer(self.hidden(x)))

    TORCH_AVAILABLE = True
except ImportError:
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_col(df, *keywords):
    """Return first column whose lowercase name contains ALL given keywords."""
    for col in df.columns:
        cl = col.lower()
        if all(k in cl for k in keywords):
            return col
    return None


def _extract_q_number(text):
    """Pull the first Q-number from e.g. 'CIP-01-May-2026-Q77 Answer'."""
    m = re.search(r'[Qq](\d+)', str(text))
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', str(text))
    return int(m.group(1)) if m else None


def _norm_answer(text):
    """Normalise an answer option for robust comparison.

    Handles: 'A', 'a', 'A.', '(A)', 'A)', ' A ', or full answer text.
    Returns uppercase stripped string.
    """
    s = str(text).strip().upper()
    # Strip surrounding punctuation/brackets to extract bare option letter
    m = re.match(r'^\(?([A-D])\)?\.?$', s)
    if m:
        return m.group(1)
    return s


def _clean_outcome(text):
    """Strip chapter/KUA prefix, returning just the learning outcome description.

    Handles formats like:
      'CIP121\\Chapter 1\\K - Explain the structure...'
      'K/ Explain the structure...'
      'K - Explain the structure...'
    K/U/A variants of the same topic collapse to the same string, giving 24 unique outcomes.
    """
    s = str(text).strip().strip('"')
    # Match any [KUA] followed by a separator then the outcome text
    m = re.search(r'[KUAkua]\s*[-/\\–]\s*(.+)', s)
    if m:
        return m.group(1).strip().strip('"')
    # Fallback: strip a leading path-like prefix (anything up to the last \ or /)
    s = re.sub(r'^.+[/\\]', '', s)
    return s.strip().strip('"')


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(responses_path: str, answers_path: str,
                 model_path=None) -> dict:
    """
    Run the GapFinder pipeline.

    Parameters
    ----------
    responses_path : path to the full-results Excel file
    answers_path   : path to the answer-key Excel file
    model_path     : optional path to gapfinder_qnn_best.pt

    Returns
    -------
    JSON-serialisable dict with gap profiles for all failing candidates.
    """

    # ── 1. Load answer key ─────────────────────────────────────────────────────
    ans = pd.read_excel(answers_path)
    ans.columns = [str(c).strip() for c in ans.columns]

    q_num_col    = _find_col(ans, "question", "number") or ans.columns[1]
    category_col = _find_col(ans, "category") or ans.columns[2]
    correct_col  = (_find_col(ans, "correct", "response") or
                    _find_col(ans, "answer") or ans.columns[-1])
    chapter_col  = _find_col(ans, "chapter") or _find_col(ans, "section")

    ans["_q_num"] = pd.to_numeric(ans[q_num_col], errors="coerce")
    ans["_lo"]    = ans[category_col].apply(_clean_outcome)
    ans = ans.dropna(subset=["_q_num"]).copy()
    ans["_q_num"] = ans["_q_num"].astype(int)

    answer_key = dict(zip(ans["_q_num"], ans[correct_col].astype(str).str.strip()))
    lo_key     = dict(zip(ans["_q_num"], ans["_lo"]))
    chapter_key = {}
    if chapter_col:
        for _, row in ans.iterrows():
            m = re.search(r"\d+", str(row[chapter_col]))
            chapter_key[int(row["_q_num"])] = int(m.group()) if m else 0

    n_questions = max(answer_key.keys())
    outcomes    = sorted(set(lo_key.values()))
    n_outcomes  = len(outcomes)
    out_idx     = {o: i for i, o in enumerate(outcomes)}

    # ── 2. Build Q-matrix ──────────────────────────────────────────────────────
    q_matrix = np.zeros((n_questions, n_outcomes), dtype=np.float32)
    for qnum, lo in lo_key.items():
        if 1 <= qnum <= n_questions and lo in out_idx:
            q_matrix[qnum - 1, out_idx[lo]] = 1.0

    # ── 3. Load response file ──────────────────────────────────────────────────
    resp = pd.read_excel(responses_path)
    resp.columns = [str(c).strip() for c in resp.columns]

    student_col = (
        _find_col(resp, "learner") or
        _find_col(resp, "student") or
        _find_col(resp, "candidate") or
        resp.columns[0]
    )
    result_col  = (_find_col(resp, "overall", "result") or
                   _find_col(resp, "result"))
    score_col   = (_find_col(resp, "overall", "score") or
                   _find_col(resp, "score"))

    answer_cols = [
        c for c in resp.columns
        if (re.search(r'\bans\b', c, re.IGNORECASE) or "answer" in c.lower())
        and _extract_q_number(c) is not None
        and "overall" not in c.lower()
        and "score" not in c.lower()
    ]

    # ── 4. Build candidate x question matrix ──────────────────────────────────
    records = []
    for _, row in resp.iterrows():
        sid = str(row[student_col])

        overall_result = "Unknown"
        if result_col:
            overall_result = str(row.get(result_col, "Unknown")).strip()

        overall_score = 0.0
        if score_col:
            try:
                overall_score = float(row.get(score_col, 0) or 0)
            except (ValueError, TypeError):
                overall_score = 0.0

        q_vec = np.zeros(n_questions, dtype=np.float32)
        for col in answer_cols:
            qnum = _extract_q_number(col)
            if qnum and 1 <= qnum <= n_questions and qnum in answer_key:
                given = str(row[col]).strip() if pd.notna(row[col]) else ""
                q_vec[qnum - 1] = 1.0 if _norm_answer(given) == _norm_answer(answer_key[qnum]) else 0.0

        records.append({
            "student_id":     sid,
            "overall_result": overall_result,
            "overall_score":  overall_score,
            "q_vec":          q_vec,
        })

    X = np.stack([r["q_vec"] for r in records]).astype(np.float32)

    # ── 5. Determine pass/fail ─────────────────────────────────────────────────
    def is_fail(rec):
        r = rec["overall_result"].lower()
        return "fail" in r or r in {"f", "no", "n"}

    failing_mask = np.array([is_fail(r) for r in records])

    # Fall back to score-based split if result column unreliable
    if failing_mask.sum() == 0:
        scores = np.array([r["overall_score"] for r in records])
        if scores.max() > 0:
            thr = np.percentile(scores, 27)
            failing_mask = scores <= thr

    # ── 6. Mastery probabilities ───────────────────────────────────────────────
    model_used = "threshold"
    mastery_probs = None

    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        try:
            q_tensor = torch.tensor(q_matrix)
            model = QNN(n_questions, n_outcomes, q_tensor)
            saved = torch.load(model_path, map_location="cpu",
                               weights_only=False)
            state = (saved.get("model_state_dict", saved)
                     if isinstance(saved, dict) else saved)
            current = model.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in current and current[k].shape == v.shape}
            current.update(compatible)
            model.load_state_dict(current, strict=False)
            model.eval()
            with torch.no_grad():
                mastery_probs = model(torch.tensor(X)).numpy()
            model_used = "qnn"
        except Exception as exc:
            print(f"[GapFinder] Model load failed ({exc}); using threshold method.")
            mastery_probs = None

    if mastery_probs is None:
        mastery_probs = np.zeros((len(X), n_outcomes), dtype=np.float32)
        for j in range(n_outcomes):
            idx = np.where(q_matrix[:, j] == 1)[0]
            if len(idx):
                mastery_probs[:, j] = X[:, idx].mean(axis=1)

    # ── 7. Build gap profiles ──────────────────────────────────────────────────
    failing_profiles = []
    passing_profiles = []
    all_profiles     = []
    profile_dist = {"Concentrated": 0, "Moderate": 0, "Broad": 0}
    outcome_gap_counts: dict = {}

    for i, rec in enumerate(records):
        probs = mastery_probs[i]

        gaps = [
            {
                "outcome":      outcomes[j],
                "mastery_prob": round(float(probs[j]) * 100, 1),
                "severity":     "high" if probs[j] < 0.30 else "moderate",
            }
            for j in range(n_outcomes) if probs[j] < 0.50
        ]
        gaps.sort(key=lambda g: g["mastery_prob"])

        strengths = [
            {
                "outcome":      outcomes[j],
                "mastery_prob": round(float(probs[j]) * 100, 1),
                "level":        "strong" if probs[j] >= 0.80 else "good",
            }
            for j in range(n_outcomes) if probs[j] >= 0.50
        ]
        strengths.sort(key=lambda s: -s["mastery_prob"])

        n_gaps = len(gaps)
        profile_type = (
            "Concentrated" if n_gaps <= 7 else
            "Moderate"     if n_gaps <= 15 else
            "Broad"
        )

        profile = {
            "student_id":     rec["student_id"],
            "overall_score":  round(rec["overall_score"], 1),
            "overall_result": rec["overall_result"],
            "n_gaps":         n_gaps,
            "profile_type":   profile_type,
            "gaps":           gaps,
            "strengths":      strengths,
            "is_failing":     bool(failing_mask[i]),
        }

        all_profiles.append(profile)

        if failing_mask[i]:
            failing_profiles.append(profile)
            profile_dist[profile_type] += 1
            for g in gaps:
                outcome_gap_counts[g["outcome"]] = (
                    outcome_gap_counts.get(g["outcome"], 0) + 1
                )
        else:
            passing_profiles.append(profile)

    top_weak = sorted(outcome_gap_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total_candidates":     len(records),
        "failing_count":        len(failing_profiles),
        "passing_count":        len(passing_profiles),
        "n_outcomes":           n_outcomes,
        "n_questions":          n_questions,
        "model_used":           model_used,
        "profile_distribution": profile_dist,
        "top_weak_outcomes":    [{"outcome": o, "count": c} for o, c in top_weak],
        "failing_candidates":   failing_profiles,
        "all_candidates":       all_profiles,
        "outcomes":             outcomes,
    }
