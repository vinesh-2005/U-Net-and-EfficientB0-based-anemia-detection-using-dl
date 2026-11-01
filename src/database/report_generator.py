"""Professional PDF Report Generator for AnemiaAI Patient Progress Tracking.

Uses ReportLab and Matplotlib to produce clean, clinically structured PDF reports
containing patient demographics, longitudinal progress comparisons, risk score line graphs,
complete screening history audit logs, and medical disclaimers.
"""
import io
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_progress_chart_image(chart_data: dict) -> io.BytesIO | None:
    """
    Renders a clean, high-resolution line chart representing the patient's
    screening risk scores across chronological dates.
    Returns an in-memory PNG image buffer.
    """
    labels = chart_data.get("labels", [])
    scores = chart_data.get("scores", [])
    threshold = chart_data.get("threshold", 38.4)

    if not labels or not scores:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 2.7), dpi=160)

    # Background styling
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # Gridlines
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=1)

    # Plot threshold line
    ax.axhline(
        y=threshold,
        color="#D97706",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label=f"Clinical Cutoff ({threshold:.1f}%)",
        zorder=2,
    )

    # Plot progress line
    x_indices = list(range(len(labels)))
    ax.plot(
        x_indices,
        scores,
        color="#0F766E",
        linewidth=2.5,
        marker="o",
        markersize=7,
        markerfacecolor="#0F766E",
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.5,
        zorder=3,
        label="Patient Risk Score (%)",
    )

    # Add numeric callouts above/below points
    for x, y in zip(x_indices, scores):
        offset = 4 if y < 90 else -8
        va = "bottom" if offset > 0 else "top"
        text_color = "#DC2626" if y >= threshold else "#059669"
        ax.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.9),
            zorder=4,
        )

    # Axes configuration
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, fontsize=8, color="#334155")
    ax.set_ylabel("Predicted Risk Score (%)", fontsize=8.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(-2, 105)

    # Clean borders
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        fontsize=7.5,
    )

    plt.title("Longitudinal Model-Predicted Risk Score Trajectory", fontsize=10, fontweight="bold", color="#0F172A", pad=10)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def generate_patient_progress_pdf(progress_data: dict) -> bytes:
    """
    Generates a complete, professional clinical PDF report for the given patient's
    anemia screening progress history.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    brand_title_style = ParagraphStyle(
        "BrandTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F766E"),
    )
    report_title_style = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )
    section_heading_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=10,
        spaceAfter=5,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
    )

    story = []

    # 1. Header Banner
    header_table_data = [
        [
            Paragraph("👁️ <b>AnemiaAI</b>", brand_title_style),
            Paragraph(f"<b>Report Generated:</b> {datetime.now(timezone.utc).strftime('%B %d, %Y &bull; %I:%M %p UTC')}", meta_style),
        ],
        [
            Paragraph("Patient Anemia Progress & Longitudinal Screening Report", report_title_style),
            Paragraph("Document ID: <b>RPT-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M") + "</b>", meta_style),
        ],
    ]
    header_table = Table(header_table_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F766E"), spaceBefore=6, spaceAfter=8))

    patient = progress_data.get("patient", {})
    screenings = progress_data.get("screenings", [])
    current_screening = progress_data.get("current_screening")
    previous_screening = progress_data.get("previous_screening")

    # 2. Patient Demographics Card
    story.append(Paragraph("<b>1. PATIENT DEMOGRAPHIC PROFILE</b>", section_heading_style))

    first_date = screenings[0].get("date_formatted", "N/A") if screenings else "N/A"
    latest_date = screenings[-1].get("date_formatted", "N/A") if screenings else "N/A"

    demo_data = [
        [
            Paragraph("<b>Patient ID:</b>", table_cell_bold),
            Paragraph(f"<font color='#0F766E'><b>{patient.get('id', 'N/A')}</b></font>", table_cell_style),
            Paragraph("<b>Full Name:</b>", table_cell_bold),
            Paragraph(f"<b>{patient.get('name', 'Anonymous Patient')}</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Biological Sex:</b>", table_cell_bold),
            Paragraph(patient.get("gender") or "Not Specified", table_cell_style),
            Paragraph("<b>Age:</b>", table_cell_bold),
            Paragraph(f"{patient.get('age')} years" if patient.get("age") else "Not Specified", table_cell_style),
        ],
        [
            Paragraph("<b>Total Screenings:</b>", table_cell_bold),
            Paragraph(f"<b>{len(screenings)} assessments</b>", table_cell_style),
            Paragraph("<b>Tracking Interval:</b>", table_cell_bold),
            Paragraph(f"{first_date} &rarr; {latest_date}", table_cell_style),
        ],
    ]

    demo_table = Table(demo_data, colWidths=[105, 165, 110, 160])
    demo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # 3. Progress Comparison & Trend Summary
    story.append(Paragraph("<b>2. LONGITUDINAL PROGRESS & RISK COMPARISON</b>", section_heading_style))

    prev_score_text = f"{progress_data.get('previous_score', 0):.1f}%" if progress_data.get("previous_score") is not None else "N/A (Baseline)"
    curr_score_text = f"{progress_data.get('current_score', 0):.1f}%"

    diff_pts = progress_data.get("diff_points")
    if diff_pts is not None:
        sign = "+" if diff_pts > 0 else ""
        diff_text = f"{sign}{diff_pts:.1f} percentage points"
    else:
        diff_text = "Baseline Assessment"

    trend_status = progress_data.get("trend_status", "Baseline")
    trend_color = progress_data.get("trend_color", "#0F766E")

    summary_data = [
        [
            Paragraph("<b>Previous Risk Score</b>", table_cell_center),
            Paragraph("<b>Current Risk Score</b>", table_cell_center),
            Paragraph("<b>Risk Score Difference</b>", table_cell_center),
            Paragraph("<b>Progress Trend Status</b>", table_cell_center),
        ],
        [
            Paragraph(f"<font size=11><b>{prev_score_text}</b></font>", table_cell_center),
            Paragraph(f"<font size=11 color='#0F766E'><b>{curr_score_text}</b></font>", table_cell_center),
            Paragraph(f"<font size=10><b>{diff_text}</b></font>", table_cell_center),
            Paragraph(f"<font size=9.5 color='{trend_color}'><b>{trend_status}</b></font>", table_cell_center),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[130, 130, 130, 150])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F4F2")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0F766E")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(summary_table)

    trend_desc = progress_data.get("trend_description", "")
    if trend_desc:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Summary:</b> {trend_desc}", table_cell_style))

    story.append(Spacer(1, 10))

    # 4. Progress Trajectory Graph
    story.append(Paragraph("<b>3. RISK SCORE TRAJECTORY GRAPH</b>", section_heading_style))
    chart_data = progress_data.get("chart_data", {})
    chart_img_buf = generate_progress_chart_image(chart_data)
    if chart_img_buf:
        chart_rl_img = RLImage(chart_img_buf, width=540, height=210)
        story.append(chart_rl_img)
    else:
        story.append(Paragraph("<i>Insufficient longitudinal records to render line graph.</i>", table_cell_style))

    story.append(Spacer(1, 10))

    # 5. Complete Screening History Audit Table
    story.append(Paragraph("<b>4. COMPLETE SCREENING HISTORY AUDIT TRAIL</b>", section_heading_style))

    history_rows = [
        [
            Paragraph("<b>#</b>", table_cell_center),
            Paragraph("<b>Screening ID</b>", table_cell_bold),
            Paragraph("<b>Date & Time</b>", table_cell_bold),
            Paragraph("<b>Predicted Result</b>", table_cell_bold),
            Paragraph("<b>Risk Score</b>", table_cell_center),
            Paragraph("<b>Cutoff</b>", table_cell_center),
            Paragraph("<b>Quality</b>", table_cell_center),
        ]
    ]

    for idx, sc in enumerate(screenings, start=1):
        is_anemic = sc.get("is_anemic", False)
        status_color = "#DC2626" if is_anemic else "#059669"
        status_label = "🔴 Anemia Detected" if is_anemic else "🟢 Non-Anemic"

        history_rows.append([
            Paragraph(str(idx), table_cell_center),
            Paragraph(f"<b>#{sc.get('id')}</b>", table_cell_style),
            Paragraph(sc.get("created_at_str", "N/A"), table_cell_style),
            Paragraph(f"<font color='{status_color}'><b>{status_label}</b></font>", table_cell_style),
            Paragraph(f"<b>{sc.get('confidence_percentage', 0):.1f}%</b>", table_cell_center),
            Paragraph(f"&ge; {sc.get('threshold_percentage', 38.4):.1f}%", table_cell_center),
            Paragraph(str(sc.get("tissue_quality", "accepted")).capitalize(), table_cell_center),
        ])

    history_table = Table(history_rows, colWidths=[24, 76, 140, 130, 65, 55, 50])
    history_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(history_table)
    story.append(Spacer(1, 14))

    # 6. Medical Notice & Disclaimer Box
    disclaimer_p = Paragraph(
        "<b>CLINICAL NOTICE & MEDICAL DISCLAIMER:</b> "
        "This progress report is generated by AnemiaAI for preliminary investigational screening and longitudinal research auditing. "
        "Risk score percentage comparisons represent trends in computer-vision conjunctival pallor features only and do not replace "
        "laboratory complete blood count (CBC) or professional medical diagnosis. Always consult a licensed healthcare physician for diagnosis and treatment.",
        disclaimer_style,
    )
    disclaimer_box = Table([[disclaimer_p]], colWidths=[540])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FDE68A")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_box)

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
