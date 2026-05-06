import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_dim_score(dim_result: Any) -> float:
    """
    VBench dimension output schema varies slightly by dimension.
    We try common keys and fall back to numeric coercion.
    """
    if isinstance(dim_result, (int, float)):
        return float(dim_result)
    # VBench-I2V often returns: [mean_score, per-sample-details]
    if isinstance(dim_result, (list, tuple)) and len(dim_result) > 0:
        if isinstance(dim_result[0], (int, float)):
            return float(dim_result[0])
    if isinstance(dim_result, dict):
        for k in ["score", "final_score", "avg_score", "mean_score"]:
            if k in dim_result and isinstance(dim_result[k], (int, float)):
                return float(dim_result[k])
        # sometimes values are nested
        for k, v in dim_result.items():
            if isinstance(v, (int, float)) and k.lower().endswith("score"):
                return float(v)
    raise ValueError(f"Cannot extract score from: {type(dim_result)}")


def _summarize(eval_json_path: str) -> Tuple[Dict[str, float], float]:
    data = _load_json(eval_json_path)
    dim_scores: Dict[str, float] = {}
    for dim, dim_result in data.items():
        try:
            dim_scores[dim] = _extract_dim_score(dim_result)
        except Exception:
            # keep NaN-like sentinel
            dim_scores[dim] = float("nan")
    valid = [v for v in dim_scores.values() if v == v]  # NaN check
    overall = sum(valid) / max(1, len(valid))
    return dim_scores, overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_json",
        nargs="+",
        required=True,
        help="one or more VBench *_eval_results.json paths (globs supported)",
    )
    args = ap.parse_args()

    paths: List[str] = []
    for p in args.eval_json:
        matches = glob.glob(p)
        paths.extend(matches if matches else [p])
    paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("No existing eval json provided.")

    rows = []
    for p in paths:
        dim_scores, overall = _summarize(p)
        rows.append((p, overall, dim_scores))

    rows.sort(key=lambda x: x[1], reverse=True)

    print("=== Ranking (higher is better; simple mean over available dimensions) ===")
    for rank, (p, overall, dim_scores) in enumerate(rows, start=1):
        dims = ", ".join([f"{k}={dim_scores[k]:.4f}" for k in sorted(dim_scores.keys()) if dim_scores[k] == dim_scores[k]])
        print(f"{rank:02d}. overall={overall:.4f}  file={p}")
        if dims:
            print(f"    {dims}")


if __name__ == "__main__":
    main()

