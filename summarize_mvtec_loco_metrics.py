#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MAX_FPRS = ["0.01", "0.05", "0.1", "0.3", "1.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MVTec LOCO evaluation metrics.json files into a Markdown table."
    )
    parser.add_argument(
        "--metrics_dir",
        default="output/1/metrics/mvtec_loco",
        help="Directory containing per-object subdirs with metrics.json.",
    )
    parser.add_argument(
        "--output_md",
        default=None,
        help="Output markdown path (default: <metrics_dir>/summary.md).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Output raw values in [0,1] instead of percent.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal digits to keep.",
    )
    parser.add_argument(
        "--max_fprs",
        default=",".join(DEFAULT_MAX_FPRS),
        help=f"Comma-separated max FPRs for AU-sPRO columns (default: {','.join(DEFAULT_MAX_FPRS)}).",
    )
    return parser.parse_args()


def fmt(value: float, *, as_percent: bool, digits: int) -> str:
    scaled = value * 100.0 if as_percent else value
    return f"{scaled:.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        raise SystemExit(f"ERROR: metrics_dir not found: {metrics_dir}")

    output_md = Path(args.output_md) if args.output_md else (metrics_dir / "summary.md")
    max_fprs = [s.strip() for s in str(args.max_fprs).split(",") if s.strip()]
    if not max_fprs:
        raise SystemExit("ERROR: --max_fprs is empty")
    as_percent = not args.raw

    metrics_paths = sorted(metrics_dir.glob("*/metrics.json"))
    if not metrics_paths:
        raise SystemExit(f"ERROR: no metrics.json found under: {metrics_dir}")

    rows_classification: list[dict[str, Any]] = []
    rows_localization_by_group: dict[str, list[dict[str, Any]]] = {
        "mean": [],
        "logical_anomalies": [],
        "structural_anomalies": [],
    }

    for p in metrics_paths:
        obj = p.parent.name
        data = load_json(p)

        auc_roc = data["classification"]["auc_roc"]
        auc_spro = data["localization"]["auc_spro"]

        rows_classification.append(
            {
                "object": obj,
                "mean": float(auc_roc["mean"]),
                "logical_anomalies": float(auc_roc["logical_anomalies"]),
                "structural_anomalies": float(auc_roc["structural_anomalies"]),
            }
        )

        for group_name, group_values in rows_localization_by_group.items():
            if group_name not in auc_spro:
                raise SystemExit(f"ERROR: missing auc_spro[{group_name}] in {p}")
            group_dict = auc_spro[group_name]
            row_loc: dict[str, Any] = {"object": obj}
            for k in max_fprs:
                if k not in group_dict:
                    raise SystemExit(f"ERROR: missing auc_spro.{group_name}[{k}] in {p}")
                row_loc[k] = float(group_dict[k])
            group_values.append(row_loc)

    # Macro averages
    rows_classification_sorted = sorted(rows_classification, key=lambda r: r["object"])

    rows_classification_sorted.append(
        {
            "object": "avg",
            "mean": mean(r["mean"] for r in rows_classification_sorted),
            "logical_anomalies": mean(r["logical_anomalies"] for r in rows_classification_sorted),
            "structural_anomalies": mean(r["structural_anomalies"] for r in rows_classification_sorted),
        }
    )

    rows_localization_sorted_by_group: dict[str, list[dict[str, Any]]] = {}
    for group_name, group_rows in rows_localization_by_group.items():
        group_sorted = sorted(group_rows, key=lambda r: r["object"])
        avg_loc: dict[str, Any] = {"object": "avg"}
        for k in max_fprs:
            avg_loc[k] = mean(r[k] for r in group_sorted)
        group_sorted.append(avg_loc)
        rows_localization_sorted_by_group[group_name] = group_sorted

    # Render markdown
    lines: list[str] = []
    lines.append("# MVTec LOCO 评测结果汇总")
    lines.append("")
    lines.append(f"- metrics_dir: `{metrics_dir}`")
    lines.append(f"- objects: {len(metrics_paths)}")
    lines.append(f"- format: {'percent(0-100)' if as_percent else 'raw(0-1)'}")
    lines.append("")

    lines.append("## Image-level (AUC ROC)")
    lines.append("")
    lines.append("| object | mean | logical_anomalies | structural_anomalies |")
    lines.append("| --- | ---: | ---: | ---: |")
    for r in rows_classification_sorted:
        lines.append(
            "| {object} | {mean} | {logical} | {struct} |".format(
                object=r["object"],
                mean=fmt(r["mean"], as_percent=as_percent, digits=args.digits),
                logical=fmt(r["logical_anomalies"], as_percent=as_percent, digits=args.digits),
                struct=fmt(r["structural_anomalies"], as_percent=as_percent, digits=args.digits),
            )
        )
    lines.append("")

    lines.append("## Localization (AU-sPRO mean)")
    lines.append("")
    def render_loc_table(group_name: str) -> None:
        headers = ["object"] + [f"@{k}" for k in max_fprs]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:"] * len(max_fprs)) + " |")
        for r in rows_localization_sorted_by_group[group_name]:
            cells = [r["object"]] + [
                fmt(r[k], as_percent=as_percent, digits=args.digits) for k in max_fprs
            ]
            lines.append("| " + " | ".join(cells) + " |")

    render_loc_table("mean")
    lines.append("")

    lines.append("## Localization (AU-sPRO logical_anomalies)")
    lines.append("")
    render_loc_table("logical_anomalies")
    lines.append("")

    lines.append("## Localization (AU-sPRO structural_anomalies)")
    lines.append("")
    render_loc_table("structural_anomalies")
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()
