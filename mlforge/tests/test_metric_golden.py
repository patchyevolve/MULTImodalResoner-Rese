"""Golden-value metric tests — every operator-facing metric pinned by a
number a human can derive on paper, so a refactor that silently changes
the definition (gain function, bin edges, tie-breaks, threshold
convention) breaks a test instead of a benchmark claim.

Each test follows the same three-part pattern where the material exists:

  1. spec quote    — the file:line in 10_training_plan/ that defines or
                     targets the metric (03:493 NDCG@3 ≥ 0.85,
                     06 §5 ECE/coverage tables, 06:251 mAP ≥ 53.0,
                     06:301 Rank-1 > 95% / mAP > 85%);
  2. golden value  — a hand-worked example whose arithmetic is written
                     out in the comment next to the assertion (the
                     literal is frozen — drift breaks the test);
  3. external oracle — where a third-party implementation exists it must
                     agree: LightGBM's own ndcg@3/@5 eval for the
                     ranker, numpy.percentile for the conformal
                     quantile, pycocotools COCOeval for detection AP.

No new dependencies: numpy/lightgbm/pycocotools are importorskip'd
(scalars-only hand arithmetic uses `math`). Deterministic, CPU-only,
pure functions — no trainer, no workspace, no harness lift needed for
behavior (conftest sets the gate anyway).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mlforge.ops.engines.calibrator import compute_metrics, metric_names
from mlforge.ops.engines.detection import _run_cocoeval
from mlforge.ops.engines.reid import _rank_metrics
from mlforge.trainers.calibrator import (
    _brier,
    _ece,
    _fit_conformal,
    _mce,
    _nll,
    _probs_at,
    _quantile,
)
from mlforge.trainers.gbdt import _mean_ndcg

# ---------------------------------------------------------------------------
# NDCG@k — 03_training_pipeline.md:493 "| NDCG@3 | ≥ 0.85 | Top-3
# hypotheses contain correct one" and :498 "Target: NDCG@3 ≥ 0.85,
# inference < 1ms". The trainer optimizes this exact function
# (trainers/gbdt.py `_mean_ndcg` — exponential gain 2^rel - 1,
# discount log2(rank + 2), group-mean).
# ---------------------------------------------------------------------------


def test_ndcg_golden_hand_worked():
    # Group A: labels [3, 2, 0, 1], model ranks them in index order
    # (scores 0.9 > 0.8 > 0.7 > 0.6), k = 3:
    #   DCG@3  = 7/log2(2) + 3/log2(3) + 0/log2(4)
    #          = 7 + 1.8927892607143718 + 0           = 8.892789260714372
    #   IDCG@3 from ideal order [3, 2, 1]
    #          = 7 + 1.8927892607143718 + 1/log2(4)   = 9.392789260714372
    #   NDCG@3 = 8.892789260714372 / 9.392789260714372
    # Group B: labels [1, 0], scores rank index 1 first:
    #   DCG@3  = 0/log2(2) + 1/log2(3) = 0.6309297535714574
    #   IDCG@3 from [1, 0]             = 1.0
    y = [3, 2, 0, 1, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.2, 0.9]
    groups = [4, 2]

    a3 = _mean_ndcg(y, scores, groups[:1], 3)
    b3 = _mean_ndcg(y[4:], scores[4:], groups[1:], 3)
    both = _mean_ndcg(y, scores, groups, 3)

    assert a3 == pytest.approx(0.9467676761267002, rel=1e-12)
    assert b3 == pytest.approx(0.6309297535714575, rel=1e-12)
    # group-mean = arithmetic mean over groups (not row-weighted)
    assert both == pytest.approx(0.7888487148490788, rel=1e-12)

    # k larger than the group is a no-op over the whole group:
    # DCG = 7 + 1.8927892607143718 + 0 + 1/log2(5) = 9.323471581914251
    # IDCG = 7 + 1.8927892607143718 + 0.5 + 0      = 9.392789260714372
    a5 = _mean_ndcg(y, scores, groups[:1], 5)
    assert a5 == pytest.approx(0.9926195041747018, rel=1e-12)

    # zero ideal gain (all labels 0) is 0.0, never a division crash
    assert _mean_ndcg([0, 0], [0.4, 0.6], [2], 3) == 0.0
    # no groups at all -> 0.0
    assert _mean_ndcg([], [], [], 3) == 0.0


def test_ndcg_oracle_lightgbm_own_metric_agrees():
    """External oracle: LightGBM computes NDCG@k itself while training
    (its default label_gain table IS 2^rel - 1). Train one tiny
    lambdarank round, let `record_evaluation` capture the framework's
    ndcg@3/@5, and require `_mean_ndcg` over the very same scores to
    match to 1e-9 — the trainer's metric and the objective's metric are
    one and the same (03:493)."""
    np = pytest.importorskip("numpy")
    lgb = pytest.importorskip("lightgbm")

    y = [3, 2, 0, 1, 1, 0]
    groups = [4, 2]
    X = np.random.RandomState(0).rand(6, 3)
    ds = lgb.Dataset(X, label=np.asarray(y, dtype=float), group=groups,
                     free_raw_data=False)
    rec: dict = {}
    booster = lgb.train(
        {"objective": "lambdarank", "metric": "ndcg", "eval_at": [3, 5],
         "verbose": -1, "min_data_in_leaf": 1, "learning_rate": 1.0,
         "num_leaves": 3, "seed": 7, "deterministic": True},
        ds,
        num_boost_round=1,
        valid_sets=[ds],
        valid_names=["tr"],
        callbacks=[lgb.record_evaluation(rec)],
    )
    lgb3 = float(rec["tr"]["ndcg@3"][0])
    lgb5 = float(rec["tr"]["ndcg@5"][0])
    scores = [float(v) for v in booster.predict(X)]

    ours3 = _mean_ndcg(y, scores, groups, 3)
    ours5 = _mean_ndcg(y, scores, groups, 5)
    assert ours3 == pytest.approx(lgb3, abs=1e-9)
    assert ours5 == pytest.approx(lgb5, abs=1e-9)


# ---------------------------------------------------------------------------
# ECE / NLL / conformal coverage — 06_benchmarking_plan.md §5:
# "compute_ece(confidences, labels, n_bins=15)" with the target table
# :348 "| ECE | < 0.05 |", :350 "| Brier Score | < 0.15 |",
# :351-352 "| Coverage (α=0.05) | ≥95% |" and "| Coverage (α=0.10) |
# ≥90% |". The trainer's `_ece` docstring pins the semantics further:
# equal confidence bins (lo, hi], weighted |accuracy - confidence|
# (13 §Model 5). No external ECE implementation ships with the repo's
# optional deps (sklearn absent) — the hand-worked arithmetic IS the
# oracle here.
# ---------------------------------------------------------------------------


def test_ece_golden_15_bins_hand_worked():
    # Four binary rows at t = 1:
    #   r0 logits [2, 0]  -> p = [0.88079707795, 0.11920292205], pred 0, label 0 OK
    #   r1 logits [0, 2]  -> p = [0.11920292205, 0.88079707795], pred 1, label 1 OK
    #   r2 logits [0, 0]  -> p = [0.5, 0.5], argmax picks index 0, label 1 WRONG
    #   r3 logits [0, 0]  -> same, label 0 OK
    # 15 equal bins (i/15, (i+1)/15]:
    #   bin 13 (0.8667, 0.9333] holds r0+r1: acc = 1.0,
    #     conf = 0.88079707795 -> |1 - conf| = 0.11920292205
    #   bin 7  (0.4667, 0.5333] holds r2+r3: acc = 0.5,
    #     conf = 0.5          -> |0.5 - 0.5| = 0
    #   ECE = (2/4) * 0.11920292205 + (2/4) * 0 = 0.05960146101
    rows = [
        {"logits": [2.0, 0.0], "label": 0},
        {"logits": [0.0, 2.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 0},
    ]
    ece = _ece(rows, 1.0, n_bins=15)
    assert ece == pytest.approx(0.05960146101105884, rel=1e-9)


def test_ece_bin_boundary_is_lo_exclusive_hi_inclusive():
    # The (lo, hi] contract in the docstring — with n_bins=2 the first
    # bin is (0, 0.5]. A row with conf exactly 0.5 (equal logits) must
    # land in bin 0, not be dropped:
    #   r0 logits [0, 0] -> conf 0.5, pred 0, label 0 -> acc 1.0
    #     bin (0, 0.5]: |1 - 0.5| = 0.5, weight 1/2  -> 0.25
    #   r1 logits [4, 0] -> p = [0.9820137900379085, ...], pred 0, label 0
    #     bin (0.5, 1]: |1 - 0.9820137900379085| = 0.0179862099620915,
    #     weight 1/2      -> 0.0089931049810458
    #   ECE = 0.2589931049810458
    # If the boundary row were dropped (wrong side of the edge), the
    # result would be only 0.0089931049810458 — this assertion
    # discriminates the two conventions.
    rows = [
        {"logits": [0.0, 0.0], "label": 0},
        {"logits": [4.0, 0.0], "label": 0},
    ]
    ece = _ece(rows, 1.0, n_bins=2)
    assert ece == pytest.approx(0.2589931049810458, rel=1e-9)


def test_nll_golden_temperature_actually_applies():
    # Same four rows as the ECE golden. NLL = -mean(log p(true)):
    #   t = 1: -log(0.88079707795) x 2, -log(0.5) x 2 over n=4
    #        = (2 x 0.1269280110044587 + 2 x 0.6931471805599453) / 4
    #        = 0.41003759580145893
    #   t = 2: logits halved first -> p([1, 0]) = e/(e+1)
    #        = 0.7310585786300049; rows 2/3 stay at 0.5
    #        = (2 x (-log 0.7310585786300049) + 2 x 0.6931471805599453) / 4
    #        = 0.5032044340390841  — worse, as expected for T = 2
    rows = [
        {"logits": [2.0, 0.0], "label": 0},
        {"logits": [0.0, 2.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 0},
    ]
    assert _nll(rows, 1.0) == pytest.approx(0.41003759580145893, rel=1e-9)
    assert _nll(rows, 2.0) == pytest.approx(0.5032044340390841, rel=1e-9)

    # `_probs_at` output is a distribution (the NLL above depends on it)
    p = _probs_at([[2.0, 0.0]], 1.0)[0]
    assert sum(p) == pytest.approx(1.0, abs=1e-12)


def test_mce_golden_shares_ece_binning():
    # 06 §5 MCE < 0.10 (06:349), formula 06:317: worst-bin gap over the
    # SAME 15 (lo, hi] bins ECE averages. Four rows from the ECE golden:
    #   bin 13 (0.8667, 0.9333] holds r0+r1: acc 1.0,
    #     conf 0.88079707795 -> gap |1 - conf| = 0.11920292205
    #   bin 7  (0.4667, 0.5333] holds r2+r3: gap 0
    #   MCE = max(0.11920292205, 0) = 0.11920292205  (= 1 - e^2/(e^2+1))
    # ECE weights the same gaps by bin mass -> MCE >= ECE, always.
    rows = [
        {"logits": [2.0, 0.0], "label": 0},
        {"logits": [0.0, 2.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 0},
    ]
    mce = _mce(rows, 1.0, n_bins=15)
    ece = _ece(rows, 1.0, n_bins=15)
    assert mce == pytest.approx(0.11920292202211769, rel=1e-9)
    assert mce >= ece

    # n_bins=2 boundary rows (from the (lo, hi] ECE test): the conf=0.5
    # row's gap 0.5 is the worst bin; the conf=0.982 row's gap
    # 0.0179862099620915 is not. MCE = 0.5 exactly — if the boundary
    # row were dropped the answer would be 0.0179862099620915.
    rows2 = [
        {"logits": [0.0, 0.0], "label": 0},
        {"logits": [4.0, 0.0], "label": 0},
    ]
    mce2 = _mce(rows2, 1.0, n_bins=2)
    assert mce2 == pytest.approx(0.5, abs=1e-15)
    assert mce2 >= _ece(rows2, 1.0, n_bins=2)

    # no rows -> 0.0, never a max() crash
    assert _mce([], 1.0) == 0.0


def test_brier_golden_hand_worked_and_numpy_oracle():
    np = pytest.importorskip("numpy")

    # 06 §5 Brier < 0.15 (06:350), multiclass form 06:320:
    #   Brier = mean_i SUM_c (p_ic - 1[y_i = c])^2   at the fitted T
    # Same four rows at t = 1 (binary => two squared terms per row):
    #   r0 [2, 0] label 0: (0.88079707795 - 1)^2 + 0.11920292205^2
    #                    = 2 x 0.11920292205^2 = 0.02841867143
    #   r1 [0, 2] label 1: symmetric, same 0.02841867143
    #   r2 [0, 0] label 1: (0.5 - 0)^2 + (0.5 - 1)^2 = 0.5
    #   r3 [0, 0] label 0: same 0.5
    #   total 1.05683734286 / 4 = 0.26420933661861107
    rows = [
        {"logits": [2.0, 0.0], "label": 0},
        {"logits": [0.0, 2.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 0},
    ]
    brier = _brier(rows, 1.0)
    assert brier == pytest.approx(0.26420933661861107, rel=1e-9)

    # external oracle: numpy one-hot form of the same expression
    probs = np.asarray(_probs_at([r["logits"] for r in rows], 1.0))
    onehot = np.eye(len(probs[0]))[np.asarray([r["label"] for r in rows])]
    assert float(np.mean(((probs - onehot) ** 2).sum(axis=1))) == \
        pytest.approx(brier, rel=1e-12)

    # a perfectly certain, always-right predictor scores 0; a
    # confidently-wrong one scores the full two terms
    assert _brier([{"logits": [50.0, 0.0], "label": 0}], 1.0) == \
        pytest.approx(0.0, abs=1e-12)
    assert _brier([{"logits": [0.0, 50.0], "label": 0}], 1.0) == \
        pytest.approx(2.0, abs=1e-6)


def test_calibrator_eval_golden_per_alpha_coverage_and_avg_set_size():
    # 06:351-353: coverage(alpha) = fraction of rows whose TRUE class
    # falls inside the prediction set {c : 1 - p_c <= thr_alpha}, and
    # avg set size = mean number of classes in that set (empty sets
    # count 0, honestly). Five rows, T = 1, thresholds 0.05 -> 0.1,
    # 0.10 -> 0.4 (fitted state, not the spec defaults' meaning):
    #   r0 [2,0] l0: 1-p_true = 0.11920292205; sets: sizes (0, 1)
    #   r1 [0,2] l1: 1-p_true = 0.11920292205; sizes (0, 1)
    #   r2 [0,0] l1: 1-p_true = 0.5;           sizes (0, 0)
    #   r3 [0,0] l0: 1-p_true = 0.5;           sizes (0, 0)
    #   r4 [4,0] l0: 1-p_true = 0.017986209962; sizes (1, 1)
    # coverage_05 = 1/5 (only r4), coverage_10 = 3/5 (r0, r1, r4)
    # avg_set_size_05 = (0+0+0+0+1)/5 = 0.2, _10 = (1+1+0+0+1)/5 = 0.6
    weights = json.dumps({
        "temperature": 1.0, "classes": 2,
        "thresholds": {"0.05": 0.1, "0.10": 0.4},
    }).encode()
    records = [
        {"logits": [2.0, 0.0], "label": 0},
        {"logits": [0.0, 2.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 1},
        {"logits": [0.0, 0.0], "label": 0},
        {"logits": [4.0, 0.0], "label": 0},
    ]
    out = compute_metrics(
        entry={"name": "golden", "version": "v1"},
        weights=weights,
        dataset={"transform": "calibration", "records": records,
                 "ref": "golden:v1"},
        protocol={},
    )
    assert out["coverage_alpha_05"] == pytest.approx(0.2, abs=1e-9)
    assert out["coverage_alpha_10"] == pytest.approx(0.6, abs=1e-9)
    assert out["avg_set_size_alpha_05"] == pytest.approx(0.2, abs=1e-9)
    assert out["avg_set_size_alpha_10"] == pytest.approx(0.6, abs=1e-9)
    # the protocol declares exactly what the metrics deliver, in order
    # (12 §15.4)
    assert tuple(out) == metric_names(weights)
    assert set(out) == {
        "ece", "nll", "mce", "brier",
        "coverage_alpha_05", "coverage_alpha_10",
        "avg_set_size_alpha_05", "avg_set_size_alpha_10",
    }


def test_conformal_coverage_golden_fractions_and_numpy_oracle():
    np = pytest.importorskip("numpy")

    # 99 calibration rows, one per nonconformity value: logits [a, 0],
    # label 0 -> p_true = sigmoid(a) strictly increasing in a, so the
    # nonconformity scores 1 - p_true are 99 DISTINCT values (the
    # quantile below is never a tie).
    n = 99
    rows = [{"logits": [i * 0.1, 0.0], "label": 0} for i in range(n)]
    # Independent recomputation of the sorted scores (pure math, no
    # mlforge import): score = 1 - e^a / (1 + e^a).
    scores = sorted(
        1.0 - math.exp(i * 0.1) / (1.0 + math.exp(i * 0.1))
        for i in range(n)
    )

    thresholds, coverages = _fit_conformal(rows, 1.0, [0.05, 0.10])

    # Spec rule (06 §5 conformal block, `_fit_conformal` docstring):
    # level = ceil((n + 1)(1 - alpha)) / n
    #   a=0.05: ceil(100 x 0.95) / 99 = 95/99 = 0.9595959595959596
    #     -> linear quantile index 94.040404..., sits strictly between
    #        sorted[94] and sorted[95] -> exactly 95 of 99 scores <= thr
    #   a=0.10: ceil(100 x 0.90) / 99 = 90/99 = 0.9090909090909091
    #     -> index 89.090909... -> exactly 90 of 99 scores <= thr
    # Those are the fractions the target table (:351-352) anchors
    # (>= 95% / >= 90% coverage at the fitted thresholds).
    assert coverages[0.05] == pytest.approx(95 / 99, rel=1e-12)
    assert coverages[0.10] == pytest.approx(90 / 99, rel=1e-12)

    # External oracle: numpy's linear-interpolation percentile must
    # produce the same threshold as `_quantile`'s contract.
    for alpha, level in ((0.05, 95 / 99), (0.10, 90 / 99)):
        oracle = float(np.percentile(scores, 100.0 * level,
                                     method="linear"))
        assert thresholds[alpha] == pytest.approx(oracle, abs=1e-12)

    # Too-few-rows clamp: with n = 9, alpha = 0.05 needs level =
    # ceil(10 x 0.95) / 9 = 10/9 > 1 -> clamped to 1.0 -> threshold =
    # max score, coverage = 1.0 (never a crash, never > 1).
    small = [{"logits": [i * 0.1, 0.0], "label": 0} for i in range(9)]
    thr_small, cov_small = _fit_conformal(small, 1.0, [0.05])
    assert cov_small[0.05] == pytest.approx(1.0, rel=1e-12)
    assert thr_small[0.05] == max(
        1.0 - math.exp(i * 0.1) / (1.0 + math.exp(i * 0.1))
        for i in range(9)
    )


def test_quantile_matches_numpy_linear_interpolation():
    np = pytest.importorskip("numpy")

    vals = sorted([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    for q in (0.0, 0.25, 0.5, 0.777, 1.0):
        oracle = float(np.percentile(vals, 100.0 * q, method="linear"))
        assert _quantile(vals, q) == pytest.approx(oracle, abs=1e-12)
    # degenerate n = 1
    assert _quantile([7.0], 0.4) == 7.0


# ---------------------------------------------------------------------------
# Rank-1 / Rank-5 / mAP — 06_benchmarking_plan.md:301 "Target: Rank-1 >
# 95%, mAP > 85% (Market1501)" over the query->gallery protocol
# documented on `_rank_metrics` (junk `0000`/`-1` dropped, same identity
# + same camera dropped as the trivial match, ties broken by record
# index). No external Market1501 scorer ships with the optional deps —
# the hand-worked AP arithmetic below is the oracle.
# ---------------------------------------------------------------------------


class _Sims:
    """Minimal duck-type of numpy's matrix: `_rank_metrics` only calls
    `.tolist()` — keeps this golden dependency-free."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, i):
        return _Row(self._rows[i])

    def tolist(self):
        return [list(r) for r in self._rows]


class _Row:
    def __init__(self, row):
        self._row = row

    def tolist(self):
        return list(self._row)


def test_rank_metrics_golden_hand_worked_map():
    # Gallery: g0 pid1/cam1, g1 pid2/cam1, g2 pid1/cam2,
    #          g3 "0000"/cam1 (junk), g4 pid1/cam1, g5 pid3/cam2
    gallery = [
        {"pid": "1", "cam": "1"}, {"pid": "2", "cam": "1"},
        {"pid": "1", "cam": "2"}, {"pid": "0000", "cam": "1"},
        {"pid": "1", "cam": "1"}, {"pid": "3", "cam": "2"},
    ]
    # q0 pid1/cam1: usable = g1, g2, g5 (g0/g4 same identity+camera,
    # g3 junk). Ground truth total = 1 (g2). Ranking by similarity:
    #   g1 0.90 (miss), g2 0.80 (hit @2), g5 0.10 (miss)
    #   rank1 miss, rank5 hit; AP = (1/2) / 1 = 0.5
    # q1 pid2/cam1: usable = g0, g2, g4, g5 — none has pid 2
    #   (g1 is same identity+camera) -> NO ground truth -> skipped.
    # q2 pid3/cam1: g5 0.99 ranks first -> hit @1, AP = 1.0
    # used = 2 -> rank1 = 1/2, rank5 = 2/2, mAP = (0.5 + 1.0) / 2
    queries = [
        {"pid": "1", "cam": "1"},
        {"pid": "2", "cam": "1"},
        {"pid": "3", "cam": "1"},
    ]
    sims = _Sims([
        [0.95, 0.90, 0.80, 0.99, 0.85, 0.10],   # q0 (g0/g3/g4 excluded)
        [0.60, 0.55, 0.50, 0.99, 0.40, 0.30],   # q1 (skipped: no GT)
        [0.10, 0.20, 0.30, 0.99, 0.40, 0.99],   # g5 first for q2
    ])
    out = _rank_metrics(queries, gallery, sims)
    assert out == {
        "rank1": pytest.approx(0.5, abs=1e-9),
        "rank5": pytest.approx(1.0, abs=1e-9),
        "mAP": pytest.approx(0.75, abs=1e-9),
    }


def test_rank_metrics_tie_break_is_record_index():
    # Two usable candidates with IDENTICAL similarity: the lower gallery
    # index must win. gallery-first-pid1 -> hit at rank 1 (AP = 1.0);
    # reversed order -> miss at rank 1 (AP = 0.5). Pins the documented
    # "(ties broken by record index)" rule.
    q = [{"pid": "1", "cam": "1"}]
    g_a = [{"pid": "1", "cam": "2"}, {"pid": "2", "cam": "1"}]
    g_b = list(reversed(g_a))
    hit = _rank_metrics(q, g_a, _Sims([[0.5, 0.5]]))
    miss = _rank_metrics(q, g_b, _Sims([[0.5, 0.5]]))
    assert hit["mAP"] == pytest.approx(1.0, abs=1e-9)
    assert miss["mAP"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# COCO mAP / AP50 — 06_benchmarking_plan.md:251 "Target: mAP ≥ 53.0
# (RF-DETR-S baseline on COCO)" measured by "COCOeval.stats[0:2]"
# (06 §4, engine `metric_names()`). pycocotools IS the oracle; these
# goldens pin the wrapper's interpretation with hand-derived AP over the
# 10 IoU thresholds {0.50, 0.55, ..., 0.95}:
#   single GT box [0, 0, 100, 100] (area 10000 > 96^2 -> "large"),
#   single detection at IoU 0.57 -> matches only t < 0.57 (2 of 10
#   thresholds, AP = 1.0 each) -> mAP = 0.2, while AP50 (t = 0.50)
#   = 1.0. A perfect box (IoU 1.0) hits all 10 -> 1.0 / 1.0 — the
#   engine seam proves that case too (test_engines_detection).
# ---------------------------------------------------------------------------


def _golden_ann(tmp_path: Path) -> Path:
    ann = {
        "info": {"description": "golden single-box COCO"},
        "images": [{"id": 1, "width": 200, "height": 200}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0,
        }],
        "categories": [{"id": 1, "name": "obj", "supercategory": "x"}],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(ann), encoding="utf-8")
    return path


def test_coco_ap_golden_partial_iou_and_subset_honesty(tmp_path):
    pytest.importorskip("pycocotools")

    ann = _golden_ann(tmp_path)

    # IoU = inter / union = (100 x 57) / (100 x 100) = 0.57 exactly.
    partial = [{
        "image_id": 1, "category_id": 1,
        "bbox": [0, 0, 100, 57], "score": 0.9,
    }]
    map_all, ap50_all = _run_cocoeval(ann, partial)
    assert map_all == pytest.approx(0.2, abs=1e-6)   # 2 of 10 thresholds
    assert ap50_all == pytest.approx(1.0, abs=1e-6)  # 0.57 > 0.50

    # sample_subset honesty: restricting the eval index to exactly the
    # images the detector ran over must not change the number here.
    map_sub, ap50_sub = _run_cocoeval(ann, partial, img_ids=[1])
    assert map_sub == map_all
    assert ap50_sub == ap50_all

    # Perfect box: all 10 thresholds match.
    perfect = [{
        "image_id": 1, "category_id": 1,
        "bbox": [0, 0, 100, 100], "score": 0.9,
    }]
    map_p, ap50_p = _run_cocoeval(ann, perfect)
    assert map_p == pytest.approx(1.0, abs=1e-6)
    assert ap50_p == pytest.approx(1.0, abs=1e-6)
