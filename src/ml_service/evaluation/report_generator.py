"""Generate ranked HTML evaluation reports for all model checkpoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import ServiceConfig, get_settings
from ..core.runtime import get_logger

_rlog = get_logger("evaluation.report_generator")


_VIEW_CODES = {"ax", "co", "sa"}

_NAME_MAP: dict[str, str] = {
    "classification.vit.finetune": "ViT Finetune",
    "classification.efficientnet.finetune": "EfficientNet Finetune",
    "classification.efficientnet.scratch": "EfficientNet Scratch",
    "classification.resnet.finetune": "ResNet Finetune",
    "classification.resnet.scratch": "ResNet Scratch",
    "classification.view_classifier": "View Classifier",
    "segmentation.unet.finetune": "U-Net Finetune",
    "segmentation.unet.scratch": "U-Net Scratch",
    "segmentation.resnet_hybrid.finetune": "ResNet-UNet Finetune",
    "segmentation.resnet_hybrid.scratch": "ResNet-UNet Scratch",
    "segmentation.swin_hafunet.finetune": "Swin-HaFUNet Finetune",
    "hybrid.adpt_net": "ADPT-Net",
    "hybrid.tgd_adpt_net": "TGD-ADPT-Net",
    "hybrid.multitask_joint": "Multitask Joint",
}
_VIEW_LABELS = {"ax": "Axial", "co": "Coronal", "sa": "Sagittal"}


def _display_name(model_name: str) -> str:
    if model_name in _NAME_MAP:
        return _NAME_MAP[model_name]
    parts = model_name.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in _VIEW_CODES and parts[0] in _NAME_MAP:
        return f"{_NAME_MAP[parts[0]]} ({_VIEW_LABELS[parts[1]]})"
    return model_name.replace("_", " ").replace(".", " · ")


def _view_suffix(model_name: str) -> str | None:
    parts = model_name.rsplit(".", 1)
    return parts[1] if len(parts) == 2 and parts[1] in _VIEW_CODES else None


def _task(payload: dict[str, Any]) -> str:
    return str(payload.get("task", "classification")).lower()


def _primary_score(payload: dict[str, Any]) -> float:
    m = payload.get("metrics", {})
    t = _task(payload)
    if t == "joint":
        return ((m.get("classification_f1_score") or 0.0) + (m.get("segmentation_dice_score") or 0.0)) / 2.0
    if t == "segmentation":
        return float(m.get("dice_score") or 0.0)
    return float(m.get("f1_score") or 0.0)


def _pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%" if v is not None else "—"
    except Exception:
        return "—"


def _num(v: Any) -> str:
    try:
        return f"{float(v):.2f}" if v is not None else "—"
    except Exception:
        return "—"


def _metric_cells(payload: dict[str, Any]) -> str:
    m = payload.get("metrics", {})
    t = _task(payload)
    if t == "joint":
        return (
            f"<td>{_pct(m.get('classification_accuracy'))}</td>"
            f"<td>{_pct(m.get('classification_f1_score'))}</td>"
            f"<td>{_pct(m.get('classification_auc'))}</td>"
            f"<td>{_pct(m.get('segmentation_dice_score'))}</td>"
            f"<td>{_pct(m.get('segmentation_iou'))}</td>"
            f"<td>{_num(m.get('segmentation_hausdorff_distance'))}</td>"
        )
    if t == "segmentation":
        return (
            f"<td colspan='3' style='color:#bbb'>—</td>"
            f"<td>{_pct(m.get('dice_score'))}</td>"
            f"<td>{_pct(m.get('iou'))}</td>"
            f"<td>{_num(m.get('hausdorff_distance'))}</td>"
        )
    return (
        f"<td>{_pct(m.get('accuracy'))}</td>"
        f"<td>{_pct(m.get('f1_score'))}</td>"
        f"<td>{_pct(m.get('auc'))}</td>"
        f"<td colspan='3' style='color:#bbb'>—</td>"
    )


def _badge(score: float) -> str:
    if score >= 0.90: return "excellent"
    if score >= 0.75: return "good"
    if score >= 0.55: return "fair"
    return "poor"


_TASK_TAG = {
    "joint": ('<span class="tag tag-joint">JOINT</span>', "tag-joint"),
    "segmentation": ('<span class="tag tag-seg">SEG</span>', "tag-seg"),
    "classification": ('<span class="tag tag-cls">CLS</span>', "tag-cls"),
    "view_classification": ('<span class="tag tag-view">VIEW</span>', "tag-view"),
}


def _render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty-note">No results available for this group.</p>'
    medals = ["🥇", "🥈", "🥉"]
    html = (
        '<table class="rt"><thead><tr>'
        '<th>#</th><th>Model</th><th>Task</th>'
        '<th>Cls Acc</th><th>Cls F1</th><th>Cls AUC</th>'
        '<th>Dice</th><th>IoU</th><th>Hausdorff</th>'
        '<th>Score</th>'
        '</tr></thead><tbody>'
    )
    for rank, p in enumerate(rows, 1):
        name = p.get("model_name", "")
        display = _display_name(name)
        t = _task(p)
        score = _primary_score(p)
        b = _badge(score)
        tag_html, _ = _TASK_TAG.get(t, (f'<span class="tag">{t.upper()}</span>', ""))
        medal = medals[rank - 1] if rank <= 3 else str(rank)
        html += (
            f'<tr class="b-{b}">'
            f'<td class="rn">{medal}</td>'
            f'<td class="mn">{display}</td>'
            f'<td>{tag_html}</td>'
            f'{_metric_cells(p)}'
            f'<td class="sc c-{b}">{score * 100:.1f}%</td>'
            f'</tr>'
        )
    html += '</tbody></table>'
    return html


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f4f6fb;color:#1e2842;font-size:14px}
.page{max-width:1280px;margin:0 auto;padding:36px 24px}
.hdr{background:linear-gradient(135deg,#1a2a5c,#2d4a8a);color:#fff;border-radius:14px;padding:36px 40px;margin-bottom:28px}
.hdr h1{font-size:26px;font-weight:700;margin-bottom:6px}
.hdr .sub{font-size:14px;opacity:.72}
.hdr .meta{display:flex;gap:18px;margin-top:20px;flex-wrap:wrap}
.mi{background:rgba(255,255,255,.12);border-radius:8px;padding:10px 16px;min-width:120px}
.mi .lbl{font-size:11px;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.mi .val{font-size:18px;font-weight:700;margin-top:3px}
.best-box{background:#f0fdf4;border:1px solid #86efac;border-radius:12px;padding:16px 20px;margin-bottom:22px;display:flex;align-items:center;gap:20px}
.best-box .crown{font-size:36px}
.best-box .bi{flex:1}
.best-box .bl{font-size:12px;font-weight:600;color:#166534;text-transform:uppercase;letter-spacing:.05em}
.best-box .bn{font-size:20px;font-weight:700;color:#1a2a5c;margin:2px 0}
.best-box .bs{font-size:13px;color:#16a34a;font-weight:600}
h2{font-size:16px;font-weight:700;margin:30px 0 10px;padding-bottom:8px;border-bottom:2px solid #e4e8f2;color:#1a2a5c}
h3{font-size:14px;font-weight:600;margin:20px 0 8px;color:#2d4a8a}
.rt{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:20px}
.rt thead tr{background:#1a2a5c;color:#fff}
.rt th{padding:10px 11px;text-align:left;font-weight:600;font-size:12px;white-space:nowrap}
.rt td{padding:9px 11px;border-bottom:1px solid #f0f2f7;font-size:13px}
.rt tbody tr:last-child td{border-bottom:none}
.rt tbody tr:hover{background:#f5f8ff}
.b-excellent td:first-child{border-left:4px solid #22c55e}
.b-good td:first-child{border-left:4px solid #4f8ef7}
.b-fair td:first-child{border-left:4px solid #f59e0b}
.b-poor td:first-child{border-left:4px solid #ef4444}
.rn{font-weight:700;color:#1a2a5c;white-space:nowrap;font-size:15px}
.mn{font-weight:600}
.sc{font-weight:700}
.c-excellent{color:#16a34a}
.c-good{color:#1d4ed8}
.c-fair{color:#d97706}
.c-poor{color:#dc2626}
.tag{display:inline-block;border-radius:4px;padding:2px 7px;font-size:11px;font-weight:700;letter-spacing:.03em}
.tag-cls{background:#dbeafe;color:#1e40af}
.tag-seg{background:#dcfce7;color:#166534}
.tag-joint{background:#f3e8ff;color:#6b21a8}
.tag-view{background:#fef9c3;color:#854d0e}
.view-grid{display:grid;grid-template-columns:1fr;gap:24px;margin-bottom:24px}
.view-card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,.07)}
.view-card h3{margin-top:0}
.empty-note{color:#aaa;font-style:italic;padding:12px 0}
.footer{margin-top:40px;text-align:center;font-size:11px;color:#aaa;padding-top:16px;border-top:1px solid #e4e8f2}
@media print{
  body{background:#fff}
  .hdr{background:#1a2a5c!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .page{padding:16px}
  h2{page-break-before:always}
  h2:first-of-type{page-break-before:avoid}
  .rt{box-shadow:none;border:1px solid #ddd}
}
"""


def _collect_existing(settings: ServiceConfig, split: str) -> dict[str, dict[str, Any]]:
    eval_root = settings.outputs_dir / "evaluations"
    results: dict[str, dict[str, Any]] = {}
    if not eval_root.exists():
        return results
    for model_dir in eval_root.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / split / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                key = payload.get("model_name") or model_dir.name
                results[key] = payload
            except Exception:
                pass
    return results


def generate_html(results: dict[str, dict[str, Any]], split: str, generated_at: str) -> str:
    all_payloads = list(results.values())
    all_sorted = sorted(all_payloads, key=_primary_score, reverse=True)

    base = [p for p in all_payloads if _view_suffix(p.get("model_name", "")) is None]
    cls_rows = sorted([p for p in base if _task(p) in ("classification", "view_classification")], key=_primary_score, reverse=True)
    seg_rows = sorted([p for p in base if _task(p) == "segmentation"], key=_primary_score, reverse=True)
    hyb_rows = sorted([p for p in base if _task(p) == "joint"], key=_primary_score, reverse=True)

    ax_rows = sorted([p for p in all_payloads if _view_suffix(p.get("model_name", "")) == "ax"], key=_primary_score, reverse=True)
    co_rows = sorted([p for p in all_payloads if _view_suffix(p.get("model_name", "")) == "co"], key=_primary_score, reverse=True)
    sa_rows = sorted([p for p in all_payloads if _view_suffix(p.get("model_name", "")) == "sa"], key=_primary_score, reverse=True)

    best_html = ""
    if all_sorted:
        best = all_sorted[0]
        score = _primary_score(best)
        best_html = (
            '<div class="best-box">'
            '<div class="crown">🏆</div>'
            '<div class="bi">'
            '<div class="bl">Best Overall Model</div>'
            f'<div class="bn">{_display_name(best.get("model_name", ""))}</div>'
            f'<div class="bs">Score: {score * 100:.1f}% &middot; {_task(best).replace("_", " ").title()}</div>'
            '</div></div>'
        )

    view_html = ""
    for label, rows in [("Axial (AX)", ax_rows), ("Coronal (CO)", co_rows), ("Sagittal (SA)", sa_rows)]:
        if rows:
            view_html += f'<div class="view-card"><h3>{label}</h3>{_render_table(rows)}</div>'

    view_section = f'<div class="view-grid">{view_html}</div>' if view_html else '<p class="empty-note">No view-specific models evaluated yet.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRISC 2025 — Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">

<div class="hdr">
  <h1>BRISC 2025 Brain Tumor Model Evaluation Report</h1>
  <div class="sub">Automated ranked report for all trained checkpoints</div>
  <div class="meta">
    <div class="mi"><div class="lbl">Generated</div><div class="val" style="font-size:13px">{generated_at}</div></div>
    <div class="mi"><div class="lbl">Split</div><div class="val">{split.upper()}</div></div>
    <div class="mi"><div class="lbl">Models Evaluated</div><div class="val">{len(all_payloads)}</div></div>
    <div class="mi"><div class="lbl">Classification</div><div class="val">{len(cls_rows)}</div></div>
    <div class="mi"><div class="lbl">Segmentation</div><div class="val">{len(seg_rows)}</div></div>
    <div class="mi"><div class="lbl">Joint</div><div class="val">{len(hyb_rows)}</div></div>
  </div>
</div>

{best_html}

<h2>Overall Rankings — Best to Worst (All Models)</h2>
{_render_table(all_sorted)}

<h2>Classification Models</h2>
{_render_table(cls_rows)}

<h2>Segmentation Models</h2>
{_render_table(seg_rows)}

<h2>Joint / Hybrid Models</h2>
{_render_table(hyb_rows)}

<h2>View-Specific Rankings</h2>
{view_section}

<div class="footer">BRISC 2025 Brain Tumor Analysis &middot; Evaluation Report &middot; Generated {generated_at}</div>
</div>
</body>
</html>"""


def _fmt_metrics(task: str, metrics: dict[str, Any]) -> str:
    """One-line metrics summary for terminal output."""
    if task == "segmentation":
        parts = []
        if metrics.get("dice_score") is not None:
            parts.append(f"Dice={metrics['dice_score']:.3f}")
        if metrics.get("iou") is not None:
            parts.append(f"IoU={metrics['iou']:.3f}")
        if metrics.get("hausdorff_distance") is not None:
            parts.append(f"Hausdorff={metrics['hausdorff_distance']:.1f}px")
        return "  ".join(parts) or "—"
    if task == "joint":
        parts = []
        if metrics.get("classification_f1_score") is not None:
            parts.append(f"Cls-F1={metrics['classification_f1_score'] * 100:.1f}%")
        if metrics.get("segmentation_dice_score") is not None:
            parts.append(f"Seg-Dice={metrics['segmentation_dice_score']:.3f}")
        return "  ".join(parts) or "—"
    parts = []
    if metrics.get("accuracy") is not None:
        parts.append(f"Acc={metrics['accuracy'] * 100:.1f}%")
    if metrics.get("f1_score") is not None:
        parts.append(f"F1={metrics['f1_score'] * 100:.1f}%")
    if metrics.get("auc") is not None:
        parts.append(f"AUC={metrics['auc'] * 100:.1f}%")
    return "  ".join(parts) or "—"


def build_report(
    split: str = "test",
    *,
    config: ServiceConfig | None = None,
    run_missing: bool = True,
) -> dict[str, Any]:
    """Collect all results, optionally evaluate missing checkpoints, then write the HTML report."""

    settings = config or get_settings()

    _rlog.info("")
    _rlog.info("=" * 64)
    _rlog.info("  EVALUATION REPORT  |  split=%s", split.upper())
    _rlog.info("=" * 64)

    _rlog.info("  Scanning for existing evaluation results ...")
    results = _collect_existing(settings, split)
    _rlog.info("  Found %d already-evaluated model(s).", len(results))

    if run_missing:
        from ..experiments.runner import run_checkpoint_evaluation
        ckpt_dir = settings.models_dir / "checkpoints"
        if ckpt_dir.exists():
            all_ckpts = sorted(ckpt_dir.glob("*.pt"))
            pending = [f for f in all_ckpts if f.stem not in results]
            skipped = len(all_ckpts) - len(pending)

            if skipped:
                _rlog.info("  Skipping %d checkpoint(s) already evaluated.", skipped)
            if not pending:
                _rlog.info("  All checkpoints already evaluated — nothing to run.")
            else:
                _rlog.info("  %d checkpoint(s) pending evaluation.", len(pending))
                _rlog.info("")

            for idx, ckpt_file in enumerate(pending, 1):
                stem = ckpt_file.stem
                _rlog.info("  [%d/%d] Evaluating: %s", idx, len(pending), stem)
                try:
                    ev = run_checkpoint_evaluation(ckpt_file, settings, split=split)
                    task = ev.get("task", "classification")
                    metrics = ev.get("metrics", {})
                    if ev.get("status") == "completed":
                        _rlog.info(
                            "         ✓ Task: %-20s  %s",
                            task, _fmt_metrics(task, metrics),
                        )
                        results[ev.get("model_name", stem)] = {
                            "model_name": ev.get("model_name", stem),
                            "task": task,
                            "split": split,
                            "metrics": metrics,
                        }
                    else:
                        _rlog.warning("         ✗ Status: %s", ev.get("status", "unknown"))
                except Exception as exc:
                    _rlog.warning("         ✗ FAILED: %s", exc)

    _rlog.info("")
    _rlog.info("  Generating HTML report for %d model(s) ...", len(results))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = generate_html(results, split, generated_at)

    report_dir = settings.outputs_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"evaluation_report_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    (report_dir / filename).write_text(html, encoding="utf-8")

    _rlog.info("  Report saved → %s", filename)
    _rlog.info("=" * 64)
    _rlog.info("")

    return {
        "status": "completed",
        "split": split,
        "evaluated_count": len(results),
        "filename": filename,
        "report_path": str(report_dir / filename),
    }
