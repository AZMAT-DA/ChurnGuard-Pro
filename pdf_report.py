# pdf_report.py — PDF report generator for churn predictions

import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_churn_report(customer_data: dict, prob: float, pred: int,
                          recommendations: list) -> bytes:
    """
    Generate a professional PDF report for a single customer churn prediction.
    Returns the PDF as bytes so Streamlit can offer it as a download.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # ---- Colors ----
    PRIMARY    = colors.HexColor('#1E3A5F')
    DANGER     = colors.HexColor('#C0392B')
    SUCCESS    = colors.HexColor('#1A7A4A')
    WARNING    = colors.HexColor('#D4700A')
    LIGHT_GRAY = colors.HexColor('#F5F5F5')
    MID_GRAY   = colors.HexColor('#CCCCCC')
    WHITE      = colors.white

    # ---- Styles ----
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Normal'],
        fontSize=22,
        fontName='Helvetica-Bold',
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Helvetica',
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=2
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Normal'],
        fontSize=13,
        fontName='Helvetica-Bold',
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=colors.HexColor('#333333'),
        leading=16
    )
    rec_style = ParagraphStyle(
        'Rec',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=colors.HexColor('#1A4A2E'),
        leading=16,
        leftIndent=10
    )

    # ---- Build content ----
    story = []

    # Header
    story.append(Paragraph("Churn Predictor Pro", title_style))
    story.append(Paragraph("Customer Churn Risk Report", subtitle_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y at %H:%M')}",
        subtitle_style
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=PRIMARY, spaceAfter=10))

    # ---- Risk Result Box ----
    risk_pct   = f"{prob:.0%}"
    risk_label = "HIGH CHURN RISK" if pred == 1 else "LOW CHURN RISK"
    risk_color = DANGER if pred == 1 else SUCCESS
    risk_note  = ("This customer is likely to leave. Immediate action recommended."
                  if pred == 1 else
                  "This customer is likely to stay. Continue regular monitoring.")

    result_data = [
        [Paragraph(risk_label, ParagraphStyle(
            'RL', parent=styles['Normal'],
            fontSize=16, fontName='Helvetica-Bold',
            textColor=WHITE, alignment=TA_CENTER
        ))],
        [Paragraph(f"Churn Probability: {risk_pct}", ParagraphStyle(
            'RP', parent=styles['Normal'],
            fontSize=12, fontName='Helvetica',
            textColor=WHITE, alignment=TA_CENTER
        ))],
        [Paragraph(risk_note, ParagraphStyle(
            'RN', parent=styles['Normal'],
            fontSize=10, fontName='Helvetica',
            textColor=WHITE, alignment=TA_CENTER
        ))]
    ]
    result_table = Table(result_data, colWidths=[17*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), risk_color),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [risk_color]),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 16),
        ('RIGHTPADDING',  (0,0), (-1,-1), 16),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.4*cm))

    # ---- Customer Details ----
    story.append(Paragraph("Customer Details", section_style))

    details_rows = []
    label_map = {
        'tenure'         : 'Tenure (months)',
        'monthly_charges': 'Monthly charges',
        'contract'       : 'Contract type',
        'internet'       : 'Internet service',
        'payment'        : 'Payment method',
        'senior'         : 'Senior citizen',
        'partner'        : 'Has partner',
        'dependents'     : 'Has dependents',
        'phone'          : 'Phone service',
        'paperless'      : 'Paperless billing',
    }
    for key, label in label_map.items():
        if key in customer_data:
            details_rows.append([
                Paragraph(label, ParagraphStyle(
                    'DL', parent=styles['Normal'],
                    fontSize=10, fontName='Helvetica-Bold',
                    textColor=colors.HexColor('#333333')
                )),
                Paragraph(str(customer_data[key]), body_style)
            ])

    if details_rows:
        det_table = Table(details_rows, colWidths=[7*cm, 10*cm])
        det_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), WHITE),
            ('ROWBACKGROUNDS',(0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ]))
        story.append(det_table)
    story.append(Spacer(1, 0.4*cm))

    # ---- Recommendations ----
    story.append(Paragraph("Recommended Actions", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=MID_GRAY, spaceAfter=8))

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(
                f"{i}. {rec}",
                rec_style if pred == 1 else body_style
            ))
            story.append(Spacer(1, 0.15*cm))
    else:
        story.append(Paragraph("No immediate actions required.", body_style))

    story.append(Spacer(1, 0.4*cm))

    # ---- Risk Meter (text based) ----
    story.append(Paragraph("Risk Assessment", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=MID_GRAY, spaceAfter=8))

    risk_level = "High" if prob > 0.65 else ("Medium" if prob > 0.4 else "Low")
    risk_lvl_color = (DANGER if prob > 0.65 else
                      (WARNING if prob > 0.4 else SUCCESS))

    meter_data = [
        ["Risk Level", "Churn Probability", "Action Priority"],
        [
            Paragraph(risk_level, ParagraphStyle(
                'RM', parent=styles['Normal'],
                fontSize=12, fontName='Helvetica-Bold',
                textColor=risk_lvl_color, alignment=TA_CENTER
            )),
            Paragraph(risk_pct, ParagraphStyle(
                'RMP', parent=styles['Normal'],
                fontSize=12, fontName='Helvetica-Bold',
                textColor=risk_lvl_color, alignment=TA_CENTER
            )),
            Paragraph(
                "Immediate" if pred == 1 else "Standard",
                ParagraphStyle(
                    'RMA', parent=styles['Normal'],
                    fontSize=12, fontName='Helvetica-Bold',
                    textColor=risk_lvl_color, alignment=TA_CENTER
                )
            )
        ]
    ]
    meter_table = Table(meter_data, colWidths=[5.6*cm, 5.7*cm, 5.7*cm])
    meter_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  PRIMARY),
        ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0),  10),
        ('BACKGROUND',    (0,1), (-1,-1), LIGHT_GRAY),
        ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(meter_table)
    story.append(Spacer(1, 0.5*cm))

    # ---- Footer ----
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=MID_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "This report was generated by Churn Predictor Pro | "
        "Model: XGBoost | Dataset: Telco Customer Churn (Kaggle) | "
        "For internal business use only.",
        ParagraphStyle(
            'Footer', parent=styles['Normal'],
            fontSize=8, fontName='Helvetica',
            textColor=colors.HexColor('#999999'),
            alignment=TA_CENTER
        )
    ))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()