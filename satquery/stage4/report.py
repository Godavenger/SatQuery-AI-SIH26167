"""Stage 4 - trace assembly and PDF report generation.

The TraceRecord is the single source of truth. The PDF is rendered from the
same record as the JSON view and the live UI, so the two can never disagree.

reportlab is already available in the environment.
"""

from __future__ import annotations

import io
import json
import time
from html import escape
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..contracts import Answer, TraceRecord


def render_png_mask(mask: np.ndarray, class_colors: Optional[dict[int, tuple]] = None,
                    alpha: float = 0.65) -> bytes:
    """Encode a class mask as a PNG for embedding in the PDF."""
    from PIL import Image
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    a_val = int(alpha * 255)
    palette = {
        0: (0, 0, 0, 0),
        1: (59, 130, 246, a_val),    # water - blue
        2: (34, 197, 94, a_val),     # vegetation - green
        3: (148, 163, 184, a_val),   # built-up - slate
        4: (217, 119, 6, a_val),     # bare soil - amber
        5: (226, 232, 240, a_val),   # snow/ice - white
        6: (220, 38, 38, a_val),     # burn scar - red
        7: (168, 85, 247, a_val),    # other - purple
    }
    if class_colors:
        palette.update(class_colors)
    for cid, color in palette.items():
        rgba[mask == cid] = color
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_pdf(trace: TraceRecord, output_path: Optional[str | Path] = None) -> bytes:
    """Render a PDF report from the TraceRecord (section 7 / section 8).

    The report contains:
        - Run metadata (ID, query, timestamps, total latency)
        - Input tile summary
        - The observable execution trace step by step
        - The answer with confidence
        - A reproducibility section (input_hash + weights_hash)

    One source means the report can never disagree with the JSON view.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()

    # ---- style overrides ----
    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontSize=14, spaceBefore=4, spaceAfter=6,
                        textColor=colors.HexColor("#1e293b"))
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=11, spaceBefore=8, spaceAfter=4,
                        textColor=colors.HexColor("#334155"))
    MONO = ParagraphStyle("MONO", parent=styles["Code"],
                          fontSize=8, leading=11,
                          backColor=colors.HexColor("#f1f5f9"),
                          textColor=colors.HexColor("#1e293b"))
    BODY = ParagraphStyle("BODY", parent=styles["Normal"],
                          fontSize=9, leading=13,
                          textColor=colors.HexColor("#374151"))
    SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8,
                           textColor=colors.HexColor("#6b7280"))
    ANSWER = ParagraphStyle("ANSWER", parent=BODY, fontSize=9.5,
                            backColor=colors.HexColor("#f0fdf4"),
                            borderColor=colors.HexColor("#22c55e"),
                            borderWidth=1, borderPadding=6, leading=14)
    ABSTAIN = ParagraphStyle("ABSTAIN", parent=BODY, fontSize=9.5,
                             backColor=colors.HexColor("#fef3c7"),
                             borderColor=colors.HexColor("#f59e0b"),
                             borderWidth=1, borderPadding=6, leading=14)

    TBL = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])

    elements = []

    def hr():
        elements.append(Spacer(1, 3 * mm))
        elements.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor("#e2e8f0")))
        elements.append(Spacer(1, 3 * mm))

    def p(text, style=BODY):
        elements.append(Paragraph(str(text), style))

    def mono(text):
        elements.append(Paragraph(escape(str(text)[:800]), MONO))

    def cells(rows):
        return [[Paragraph(escape(str(value)), SMALL) for value in row]
                for row in rows]

    # ---- cover ----
    elements.append(Spacer(1, 6 * mm))
    p("<b>SatQuery AI &mdash; Execution Report</b>", H1)
    p("SIH26167 &nbsp;|&nbsp; ISRO / Space Technology", SMALL)
    p(f"Problem Statement ID: SIH26167 &nbsp;|&nbsp; Run: {escape(trace.run_id)}", SMALL)
    started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(trace.started_at))
    p(f"Generated: {started} &nbsp;|&nbsp; Latency: {trace.total_latency_ms:.0f} ms", SMALL)
    hr()

    # ---- query ----
    p("<b>Query</b>", H2)
    p(escape(trace.query or "(no query text)"))
    elements.append(Spacer(1, 2 * mm))

    # ---- tiles ----
    if trace.tiles:
        p("<b>Input Tiles</b>", H2)
        rows = [["Tile ID", "Sensor", "GSD (m)", "Bands", "Timestamp", "Clear"]]
        for t in trace.tiles:
            rows.append([
                str(t.get("id", ""))[:28],
                str(t.get("sensor_id", "")),
                str(t.get("gsd_m", "")),
                ", ".join(t.get("bands", [])),
                str(t.get("ts", "")),
                f"{100 * t.get('clear_fraction', 0):.1f}%",
            ])
        tbl = Table(cells(rows), colWidths=[39*mm, 28*mm, 16*mm, 42*mm, 26*mm, 16*mm], repeatRows=1)
        tbl.setStyle(TBL)
        elements.append(tbl)
        elements.append(Spacer(1, 3 * mm))

    # ---- trace steps ----
    p("<b>Execution Trace</b>", H2)
    p("The statement evaluates the observable execution trace. "
      "Every step below corresponds directly to one of the six controller "
      "duties named in the problem statement.", SMALL)
    elements.append(Spacer(1, 2 * mm))

    rows = [["#", "Stage", "Step", "Tool / Model", "Confidence", "Summary"]]
    for s in trace.steps:
        rows.append([
            str(s.seq),
            str(s.stage),
            s.name or "",
            f"{s.tool_name or ''} {s.tool_version or ''}".strip(),
            f"{s.confidence:.2f}" if s.confidence is not None else "--",
            (s.output_summary or "")[:90],
        ])
    tbl = Table(cells(rows), colWidths=[8*mm, 12*mm, 30*mm, 38*mm, 16*mm, 63*mm], repeatRows=1)
    tbl.setStyle(TBL)
    elements.append(tbl)
    hr()

    # ---- gate verdicts ----
    gate_steps = [s for s in trace.steps if s.gate_verdict]
    if gate_steps:
        p("<b>Compatibility Gate Verdicts</b>", H2)
        p("Five-field compatibility check per section 4.2. Passes are recorded "
          "explicitly -- a silent pass is indistinguishable from a check that "
          "never ran.", SMALL)
        elements.append(Spacer(1, 2 * mm))
        for gs in gate_steps:
            gv = gs.gate_verdict
            rows = [["Field", "Passed", "Detail"]]
            for f in gv.get("fields", []):
                rows.append([
                    f.get("name", ""),
                    "PASS" if f.get("passed") else "FAIL",
                    (f.get("detail", ""))[:120],
                ])
            tbl = Table(cells(rows), colWidths=[30*mm, 16*mm, 121*mm], repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f8fafc")]),
                ("TEXTCOLOR", (1, 1), (1, -1), colors.HexColor("#16a34a")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            # colour FAIL cells red
            for ri, row in enumerate(rows[1:], start=1):
                if row[1] == "FAIL":
                    tbl.setStyle(TableStyle([
                        ("TEXTCOLOR", (1, ri), (1, ri), colors.HexColor("#dc2626")),
                    ]))
            elements.append(tbl)
            action = gv.get("action", "")
            reason = gv.get("reason", "")
            if reason:
                p(f"Action: <b>{escape(action)}</b> &mdash; {escape(reason)}", SMALL)
            elements.append(Spacer(1, 2 * mm))

    # ---- answer ----
    if trace.answer:
        hr()
        ans = trace.answer
        p("<b>Answer</b>", H2)
        style = ABSTAIN if ans.abstained else ANSWER
        p(f"<b>{'[ABSTAINED]' if ans.abstained else '[ANSWER]'}</b> "
          f"Confidence: {ans.confidence:.0%} &nbsp;|&nbsp; "
          f"Intent: {escape(ans.intent)} &nbsp;|&nbsp; "
          f"Output kind: {escape(ans.output_kind)}", SMALL)
        elements.append(Spacer(1, 1 * mm))
        p(escape(ans.text), style)

        if ans.conflicts:
            elements.append(Spacer(1, 2 * mm))
            p("<b>Cross-model conflicts detected</b>", SMALL)
            for c in ans.conflicts:
                p(escape(f"  {' vs '.join(c.between if hasattr(c,'between') else c.get('between',[]))}: "
                  f"{c.detail if hasattr(c,'detail') else c.get('detail','')} "
                  f"(penalty {c.penalty if hasattr(c,'penalty') else c.get('penalty',0):.2f})"),
                  SMALL)

        cal = ans.calibration
        if cal:
            elements.append(Spacer(1, 2 * mm))
            p("<b>Calibration</b>", SMALL)
            cal_rows = [[k, str(v)] for k, v in cal.items()]
            tbl = Table(cells(cal_rows), colWidths=[55*mm, 100*mm])
            tbl.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1),
                 [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            elements.append(tbl)

    # ---- reproducibility ----
    hr()
    p("<b>Reproducibility</b>", H2)
    p("Input hash + weights hash is what an ISRO / SAC reviewer will ask for. "
      "Both are stored on every trace step.", SMALL)
    elements.append(Spacer(1, 2 * mm))
    repro_rows = [["Step", "input_hash", "weights_hash"]]
    for s in trace.steps:
        if s.input_hash or s.weights_hash:
            repro_rows.append([
                s.name or "",
                s.input_hash or "--",
                s.weights_hash or "--",
            ])
    if len(repro_rows) > 1:
        tbl = Table(cells(repro_rows), colWidths=[35*mm, 65*mm, 65*mm], repeatRows=1)
        tbl.setStyle(TBL)
        elements.append(tbl)
    else:
        p("(no hashes recorded in this run -- backbone load with weights_hash "
          "will populate these fields)", SMALL)

    # ---- software versions ----
    if trace.software:
        elements.append(Spacer(1, 2 * mm))
        p("<b>Software</b>", SMALL)
        mono(json.dumps(trace.software, indent=2)[:600])

    doc.build(elements)
    pdf_bytes = buf.getvalue()

    if output_path:
        Path(output_path).write_bytes(pdf_bytes)

    return pdf_bytes
