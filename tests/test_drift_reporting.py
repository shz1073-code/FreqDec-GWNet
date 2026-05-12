"""Tests for drift reporting (DriftReport / CSV / PNG / assemble)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from freqdec_gwnet.utils.drift_reporting import (
    DriftReport,
    assemble_drift_report,
    render_drift_png,
    write_drift_csv,
)


def _make_report(T: int = 4) -> DriftReport:
    nan = float("nan")
    landmark = torch.tensor([
        [10.0, 10.0],
        [13.0, 14.0],
        [nan, nan],
        [10.0, 14.0],
    ])
    U = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
        [3.0, 4.0],
        [0.0, 4.0],
    ])
    delta_u = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
        [0.0, 0.0],
        [-3.0, 0.0],
    ])
    return assemble_drift_report(
        seq_name="seq_demo", dataset="ds_test", split="train",
        frame_names=[f"frame_{i:03d}.png" for i in range(T)],
        reference_index=0, reference_strategy="warmup_score",
        delta_u=delta_u, U=U, landmark_xy=landmark,
        extra={"state_source": "sinusoidal"},
    )


# ---------------------------------------------------------------------------
# DriftReport invariants
# ---------------------------------------------------------------------------


def test_assemble_report_matches_metric_outputs():
    rep = _make_report()
    assert rep.seq_name == "seq_demo"
    assert rep.reference_index == 0
    # drift_curve == ||U||
    assert torch.allclose(rep.drift_curve, rep.U.norm(dim=-1))
    # frame 2 没标注 → before/after NaN
    assert torch.isnan(rep.before_residual[2])
    assert torch.isnan(rep.after_residual[2])
    assert rep.landmark_present.tolist() == [True, True, False, True]


def test_drift_report_rejects_size_mismatch():
    """T 不一致时 __post_init__ 必须报错。"""
    with pytest.raises(ValueError, match="DriftReport"):
        DriftReport(
            seq_name="x", dataset="d", split="train",
            frame_names=["a", "b"],
            reference_index=0, reference_strategy="first_frame",
            delta_u=torch.zeros(3, 2),    # T=3 但 frame_names 只有 2
            U=torch.zeros(3, 2),
            drift_curve=torch.zeros(3),
            before_residual=torch.zeros(3),
            after_residual=torch.zeros(3),
            landmark_present=torch.zeros(3, dtype=torch.bool),
            improvement_mean=torch.tensor(0.0),
        )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def test_write_drift_csv_creates_expected_columns(tmp_path):
    rep = _make_report()
    out = write_drift_csv(rep, tmp_path / "out.csv")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    # header lines
    assert "# seq_name=seq_demo" in text
    assert "# reference_index=0" in text
    assert "# reference_strategy=warmup_score" in text
    assert "# state_source=sinusoidal" in text
    # column header
    assert ("frame_index,frame_name,delta_u_x,delta_u_y,U_x,U_y,"
            "drift_norm,landmark_present,before_residual,after_residual"
            in text)


def test_write_drift_csv_renders_nan_as_empty(tmp_path):
    rep = _make_report()
    out = write_drift_csv(rep, tmp_path / "out.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    # 找到对应 frame 2 的数据行（去掉 header 行 + 列名行）
    data_rows = [ln for ln in lines if not ln.startswith("#")
                 and not ln.startswith("frame_index")]
    assert len(data_rows) == rep.delta_u.shape[0]
    row2 = data_rows[2].split(",")
    # before_residual / after_residual 应该是空字符串（因为 NaN）
    # 列顺序见 _CSV_COLUMNS：最后两列正是 before / after
    assert row2[-2] == "" and row2[-1] == ""


def test_write_drift_csv_rounds_floats(tmp_path):
    rep = _make_report()
    out = write_drift_csv(rep, tmp_path / "out.csv")
    text = out.read_text(encoding="utf-8")
    # frame 1 的 delta_u_x = 3.0 -> 应该被格式化为 "3"（%g 去尾零）
    assert ",3," in text or ",3.0," in text or ",3.00000," not in text


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------


def test_render_drift_png_creates_file(tmp_path):
    rep = _make_report()
    out = render_drift_png(rep, tmp_path / "out.png")
    assert out.is_file()
    # 看起来不像空文件
    assert out.stat().st_size > 1000


def test_render_drift_png_handles_no_landmark(tmp_path):
    """没有任何 landmark 时也要能生成图，不抛错。"""
    T = 5
    U = torch.zeros(T, 2)
    delta_u = torch.zeros(T, 2)
    rep = assemble_drift_report(
        seq_name="no_tip", dataset="ds", split="train",
        frame_names=[f"f_{i}.png" for i in range(T)],
        reference_index=0, reference_strategy="first_frame",
        delta_u=delta_u, U=U, landmark_xy=None,
    )
    out = render_drift_png(rep, tmp_path / "no_tip.png")
    assert out.is_file()


# ---------------------------------------------------------------------------
# scripts/evaluate_drift integration smoke test (real data, auto-skip)
# ---------------------------------------------------------------------------


def test_evaluate_drift_pipeline_real_data(tmp_path):
    """End-to-end run on dense_v1/cmu_seq_007 using public API."""
    pytest.importorskip("matplotlib")
    sys_path = __import__("sys").path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "scripts") not in sys_path:
        sys_path.insert(0, str(repo_root / "scripts"))
    try:
        evaluate_drift = __import__("evaluate_drift")
    except ImportError as exc:
        pytest.skip(f"cannot import scripts/evaluate_drift: {exc}")

    from freqdec_gwnet.data import DataPaths
    try:
        paths = DataPaths.from_default_config()
    except FileNotFoundError as exc:
        pytest.skip(f"real data not mounted: {exc}")
    if "dense_v1" not in paths.datasets:
        pytest.skip("dense_v1 not configured")

    result = evaluate_drift.evaluate_sequence(
        paths,
        dataset="dense_v1",
        split="train",
        seq_name="cmu_seq_007",
        output_dir=tmp_path,
        max_frames=20,
        state_source="sinusoidal",
    )
    assert result["csv_path"].is_file()
    assert result["png_path"].is_file()
    assert result["n_frames"] == 20
    # ReferenceSelector 必须报告策略
    assert result["reference_strategy"] in (
        "clinical", "warmup_score", "first_frame",
    )
    # 未训练模型 -> Δu 全 0 -> after 应等于 before -> improvement_mean ≈ 0
    assert abs(result["improvement_mean"]) < 1e-5
