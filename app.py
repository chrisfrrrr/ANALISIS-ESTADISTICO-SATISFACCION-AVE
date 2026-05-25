
import io
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.pdfbase.pdfmetrics import stringWidth

APP_TITLE = "Herramienta de análisis de satisfacción - AVE"
DEVELOPER_LINE = "Desarrollado por Ing. Christian Pocol, Ingeniero Electrónico"
HEADER_SUBTITLE = "Universidad del Valle de Guatemala UVG"
DEFAULT_AUTHOR = "Ing. Christian Pocol, Ingeniero Electrónico"

LIKERT_MAP = {
    5: "Totalmente de acuerdo",
    4: "De acuerdo",
    3: "Ni de acuerdo ni en desacuerdo",
    2: "En desacuerdo",
    1: "Totalmente en desacuerdo",
}

ITEMS = [
    "Los objetivos del aprendizaje fueron claros desde el inicio y a lo largo del curso.",
    "Desarrollé las habilidades planteadas en el curso.",
    "Las instrucciones son completas.",
    "El curso está bien estructurado, hay un orden lógico.",
    "Las actividades y evaluaciones estuvieron alineadas con los objetivos del curso.",
    "La carga de trabajo es equilibrada con el tiempo asignado.",
    "Los videos presentaron los temas de manera comprensible.",
    "Los materiales de apoyo fueron suficientes para mi aprendizaje.",
    "Las evaluaciones fueron claras.",
    "Las evaluaciones estaban relacionadas con el logro de las habilidades.",
    "Los criterios fueron justos y consistentes.",
    "El asesor de bienestar me contactó.",
    "La plataforma fue fácil de navegar.",
    "Me siento muy satisfecho de mis aprendizajes en este curso.",
    "El curso cumplió con mis expectativas.",
]
OPINION_16 = "¿Qué aspectos del curso (diseño, materiales, enlaces, acompañamiento u otros) fue más valioso para tu aprendizaje?"
OPINION_17 = "¿Qué debemos mejorar en este curso?"

CATEGORIES = {
    "Claridad y logro de aprendizaje": [1, 2, 5, 10, 14, 15],
    "Diseño instruccional y estructura": [3, 4, 6, 8],
    "Recursos y plataforma": [7, 13],
    "Evaluación y criterios": [9, 11],
    "Acompañamiento": [12],
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_AVE = os.path.join(ASSETS_DIR, "logo_ave.png")
LOGO_UVG = os.path.join(ASSETS_DIR, "logo_uvg.png")
APP_ICON = os.path.join(ASSETS_DIR, "app_icon.ico")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON if os.path.exists(APP_ICON) else "📊",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid #E8EDF5; border-radius: 18px; padding: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); background: #FFFFFF;
    }
    .small-muted {color: #64748B; font-size: 0.88rem;}
    .footer-dev {
        text-align: center; color:#64748B; font-size:0.82rem; margin-top: 2rem;
        border-top: 1px solid #E2E8F0; padding-top: 0.8rem;
    }
    div[data-testid="stMetricValue"] {font-size: 2.0rem;}
    .uvg-title {
        font-weight: 800; color:#17257C; letter-spacing: -0.03em;
        margin-bottom: 0;
    }
    .green-accent { color:#00A83B; font-weight:700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_selected_option(page: fitz.Page, option_blocks: List[Tuple[int, Tuple[float, float, float, float], str]]) -> Optional[int]:
    """
    Detecta la opción marcada a partir del círculo relleno de Canvas/SpeedGrader.
    La detección se hace por imagen, no por OCR, porque la respuesta seleccionada
    normalmente aparece como radio button y no como texto extraíble.
    """
    if not option_blocks:
        return None

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    arr = np.array(img)
    scale = 2.0

    scores = []
    for val, box, text in option_blocks:
        x0, y0, x1, y1 = box
        cy = (y0 + y1) / 2.0
        # En el formato Canvas el radio button está ~13 puntos a la izquierda del texto.
        cx = max(24, x0 - 13.5)
        x_start = max(0, int((cx - 9) * scale))
        x_end = min(arr.shape[1], int((cx + 9) * scale))
        y_start = max(0, int((cy - 9) * scale))
        y_end = min(arr.shape[0], int((cy + 9) * scale))
        crop = arr[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            scores.append((val, 0.0))
            continue
        gray = crop.mean(axis=2)
        dark_ratio = float((gray < 85).mean())
        scores.append((val, dark_ratio))

    best_val, best_score = max(scores, key=lambda x: x[1])
    return best_val if best_score > 0.08 else None


def parse_speedgrader_pdf(file_bytes: bytes, filename: str) -> Dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    values: Dict[int, Optional[int]] = {}
    opinions = {16: "", 17: ""}
    time_for_attempt = ""

    # Extrae tiempo para intento si está disponible.
    full_text = "\n".join([p.get_text("text") for p in doc])
    m_time = re.search(r"(\d{1,2}:\d{2})\s*\n\s*Tiempo para este intento", full_text)
    if m_time:
        time_for_attempt = m_time.group(1)

    item_counter = 1
    for page in doc:
        blocks = page.get_text("blocks")
        option_blocks = []
        for b in blocks:
            txt = normalize_text(b[4])
            m = re.match(r"^([1-5])\s*=", txt)
            if m:
                option_blocks.append((int(m.group(1)), b[:4], txt))

        # Agrupar cada 5 opciones consecutivas como una pregunta Likert.
        for i in range(0, len(option_blocks), 5):
            group = option_blocks[i:i + 5]
            if len(group) == 5 and item_counter <= 15:
                values[item_counter] = detect_selected_option(page, group)
                item_counter += 1

    # Opiniones abiertas: se extraen de la última página buscando los prompts.
    if len(doc) > 0:
        last_text = doc[-1].get_text("text")
        clean = last_text.replace("\xa0", " ")
        p16 = re.search(
            r"¿Qué aspectos del curso.*?tu aprendizaje\?\s*(.*?)\s*¿Qué debemos mejorar en este curso\?",
            clean, flags=re.S | re.I
        )
        p17 = re.search(r"¿Qué debemos mejorar en este curso\?\s*(.*?)\s*(?:\n15\n16\n17|\Z)", clean, flags=re.S | re.I)
        if p16:
            opinions[16] = normalize_text(p16.group(1))
        if p17:
            opinions[17] = normalize_text(p17.group(1))

    # Identificador amigable desde nombre de archivo.
    student_id = os.path.splitext(os.path.basename(filename))[0]
    return {
        "archivo": filename,
        "estudiante": student_id,
        "tiempo_intento": time_for_attempt,
        **{f"item_{i}": values.get(i) for i in range(1, 16)},
        "opinion_valioso": opinions[16],
        "opinion_mejora": opinions[17],
    }


def build_dataframes(records: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(records)
    item_cols = [f"item_{i}" for i in range(1, 16)]
    for c in item_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    long = raw.melt(
        id_vars=["archivo", "estudiante", "tiempo_intento", "opinion_valioso", "opinion_mejora"],
        value_vars=item_cols,
        var_name="item",
        value_name="calificacion"
    )
    long["numero_item"] = long["item"].str.extract(r"(\d+)").astype(int)
    long["enunciado"] = long["numero_item"].map(lambda i: ITEMS[i - 1])
    long["respuesta"] = long["calificacion"].map(LIKERT_MAP)
    long["satisfaccion_pct"] = long["calificacion"] / 5 * 100

    item_stats = long.groupby(["numero_item", "enunciado"], as_index=False).agg(
        respuestas_validas=("calificacion", "count"),
        promedio=("calificacion", "mean"),
        satisfaccion_pct=("satisfaccion_pct", "mean"),
        desviacion=("calificacion", "std"),
    )
    item_stats["desviacion"] = item_stats["desviacion"].fillna(0)

    cat_rows = []
    for cat, nums in CATEGORIES.items():
        vals = long[long["numero_item"].isin(nums)]["calificacion"].dropna()
        cat_rows.append({
            "categoria": cat,
            "items": ", ".join(map(str, nums)),
            "promedio": vals.mean() if len(vals) else np.nan,
            "satisfaccion_pct": vals.mean() / 5 * 100 if len(vals) else np.nan,
            "respuestas_validas": int(vals.count()),
        })
    cat_stats = pd.DataFrame(cat_rows)

    dist = long.groupby(["numero_item", "calificacion"], as_index=False).size()
    dist["respuesta"] = dist["calificacion"].map(LIKERT_MAP)
    return raw, long, item_stats, cat_stats


def satisfaction_label(value_pct: float) -> str:
    if pd.isna(value_pct):
        return "Sin datos"
    if value_pct >= 90:
        return "Excelente"
    if value_pct >= 80:
        return "Muy satisfactorio"
    if value_pct >= 70:
        return "Satisfactorio"
    if value_pct >= 60:
        return "Atención moderada"
    return "Prioridad de mejora"


def make_excel(raw, long, item_stats, cat_stats) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, sheet_name="Datos procesados")
        long.to_excel(writer, index=False, sheet_name="Base larga")
        item_stats.to_excel(writer, index=False, sheet_name="Estadísticas por ítem")
        cat_stats.to_excel(writer, index=False, sheet_name="Estadísticas por categoría")

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 58)
    return output.getvalue()


def add_pdf_watermark(canvas, doc):
    canvas.saveState()
    w, h = landscape(letter)
    canvas.setFillColor(colors.Color(0.1, 0.15, 0.49, alpha=0.08))
    canvas.setFont("Helvetica-Bold", 34)
    canvas.translate(w / 2, h / 2)
    canvas.rotate(35)
    canvas.drawCentredString(0, 0, "AVE UVG")
    canvas.setFont("Helvetica", 13)
    canvas.drawCentredString(0, -24, DEFAULT_AUTHOR)
    canvas.restoreState()

    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 0.5 * inch, 0.35 * inch, f"{DEVELOPER_LINE} | Página {doc.page}")
    canvas.restoreState()


def make_pdf_report(raw, long, item_stats, cat_stats, course_name: str, section_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(letter),
        rightMargin=0.45 * inch, leftMargin=0.45 * inch,
        topMargin=0.35 * inch, bottomMargin=0.55 * inch
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleAVE", parent=styles["Title"], textColor=colors.HexColor("#17257C"), fontSize=20, leading=24))
    styles.add(ParagraphStyle("SubAVE", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#475569"), fontSize=9))
    styles.add(ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=9))
    story = []

    # Header
    header_data = []
    left_logo = RLImage(LOGO_AVE, width=1.35*inch, height=0.48*inch) if os.path.exists(LOGO_AVE) else Paragraph("AVE UVG", styles["TitleAVE"])
    right_logo = RLImage(LOGO_UVG, width=0.65*inch, height=0.65*inch) if os.path.exists(LOGO_UVG) else Paragraph("UVG", styles["TitleAVE"])
    title = Paragraph(f"<b>{APP_TITLE}</b><br/><font size='9'>{HEADER_SUBTITLE}<br/>{DEVELOPER_LINE}</font>", styles["TitleAVE"])
    header_table = Table([[left_logo, title, right_logo]], colWidths=[1.7*inch, 7.3*inch, 1.1*inch])
    header_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,0), (1,0), "CENTER")]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))

    total_students = len(raw)
    total_answers = int(long["calificacion"].count())
    global_avg = float(long["calificacion"].mean()) if total_answers else np.nan
    global_pct = global_avg / 5 * 100 if total_answers else np.nan
    label = satisfaction_label(global_pct)

    info = [
        ["Curso", course_name or "No especificado", "Sección", section_name or "No especificada"],
        ["PDF analizados", str(total_students), "Respuestas válidas", str(total_answers)],
        ["Promedio global", f"{global_avg:.2f} / 5" if not np.isnan(global_avg) else "Sin datos",
         "Satisfacción global", f"{global_pct:.1f}% - {label}" if not np.isnan(global_pct) else "Sin datos"],
        ["Fecha de generación", datetime.now().strftime("%d/%m/%Y %H:%M"), "Desarrollador", DEFAULT_AUTHOR],
    ]
    t = Table(info, colWidths=[1.35*inch, 3.55*inch, 1.45*inch, 3.75*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8F2FF")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.18*inch))

    story.append(Paragraph("<b>Resultados por categoría</b>", styles["Heading2"]))
    cat_display = cat_stats.copy()
    cat_display["promedio"] = cat_display["promedio"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    cat_display["satisfaccion_pct"] = cat_display["satisfaccion_pct"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    cat_data = [["Categoría", "Ítems", "Promedio", "Satisfacción", "N"]] + cat_display[["categoria","items","promedio","satisfaccion_pct","respuestas_validas"]].values.tolist()
    cat_table = Table(cat_data, colWidths=[3.1*inch, 1.3*inch, 1.0*inch, 1.2*inch, 0.6*inch])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17257C")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.2),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 0.18*inch))

    story.append(Paragraph("<b>Estadísticas por ítem</b>", styles["Heading2"]))
    item_display = item_stats.copy()
    item_display["promedio"] = item_display["promedio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    item_display["satisfaccion_pct"] = item_display["satisfaccion_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    rows = [["No.", "Ítem", "N", "Prom.", "% Sat.", "Desv."]]
    for _, r in item_display.iterrows():
        rows.append([
            int(r["numero_item"]),
            Paragraph(str(r["enunciado"]), styles["Cell"]),
            int(r["respuestas_validas"]),
            r["promedio"],
            r["satisfaccion_pct"],
            f'{float(r["desviacion"]):.2f}',
        ])
    table = Table(rows, colWidths=[0.38*inch, 6.4*inch, 0.45*inch, 0.62*inch, 0.72*inch, 0.62*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#00A83B")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 6.8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(table)

    opinions = pd.concat([
        raw[["estudiante", "opinion_valioso"]].rename(columns={"opinion_valioso": "comentario"}).assign(tipo="Aspectos valiosos"),
        raw[["estudiante", "opinion_mejora"]].rename(columns={"opinion_mejora": "comentario"}).assign(tipo="Mejoras sugeridas"),
    ])
    opinions = opinions[opinions["comentario"].fillna("").str.len() > 0].head(30)

    story.append(PageBreak())
    story.append(Paragraph("<b>Comentarios cualitativos destacados</b>", styles["Heading2"]))
    if len(opinions):
        rows = [["Tipo", "Comentario"]]
        for _, r in opinions.iterrows():
            rows.append([r["tipo"], Paragraph(r["comentario"], styles["Cell"])])
        op_table = Table(rows, colWidths=[1.5*inch, 8.0*inch])
        op_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17257C")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(op_table)
    else:
        story.append(Paragraph("No se detectaron respuestas abiertas en los PDF procesados.", styles["Normal"]))

    doc.build(story, onFirstPage=add_pdf_watermark, onLaterPages=add_pdf_watermark)
    return buffer.getvalue()


def render_header():
    col1, col2, col3 = st.columns([1.3, 4.5, 1.0])
    with col1:
        if os.path.exists(LOGO_AVE):
            st.image(LOGO_AVE, use_container_width=True)
    with col2:
        st.markdown(f"<h1 class='uvg-title'>{APP_TITLE}</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='green-accent'>Desarrollado por Ing. Christian Pocol Asesor AVE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{HEADER_SUBTITLE}</div>", unsafe_allow_html=True)
    with col3:
        if os.path.exists(LOGO_UVG):
            st.image(LOGO_UVG, use_container_width=True)


def metric_card(label, value, help_text=""):
    st.metric(label, value, help=help_text if help_text else None)


def main():
    render_header()
    st.divider()

    with st.sidebar:
        st.image(LOGO_AVE, use_container_width=True) if os.path.exists(LOGO_AVE) else None
        st.subheader("Configuración del informe")
        course_name = st.text_input("Nombre del curso", value="Fundamentos de la Comunicación")
        section_name = st.text_input("Sección", value="")
        max_files = st.number_input("Límite de PDF a procesar", min_value=1, max_value=600, value=600, step=10)
        st.caption(DEVELOPER_LINE)
        st.caption("Carga PDF exportados desde Canvas SpeedGrader con 15 ítems Likert y 2 preguntas abiertas.")

    uploaded_files = st.file_uploader(
        "Cargar PDF de estudiantes",
        type=["pdf"],
        accept_multiple_files=True,
        help="Puede cargar hasta 600 PDF. Se recomienda usar los PDF originales exportados desde Canvas/SpeedGrader."
    )

    if not uploaded_files:
        st.info("Carga uno o varios PDF para iniciar el análisis estadístico de satisfacción.")
        st.stop()

    if len(uploaded_files) > max_files:
        st.error(f"Se cargaron {len(uploaded_files)} archivos, pero el límite configurado es {max_files}.")
        st.stop()

    records = []
    errors = []
    progress = st.progress(0, text="Procesando PDF...")
    for idx, f in enumerate(uploaded_files, start=1):
        try:
            records.append(parse_speedgrader_pdf(f.read(), f.name))
        except Exception as e:
            errors.append({"archivo": f.name, "error": str(e)})
        progress.progress(idx / len(uploaded_files), text=f"Procesando {idx}/{len(uploaded_files)} PDF...")
    progress.empty()

    if errors:
        with st.expander("Archivos con error de procesamiento", expanded=False):
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

    if not records:
        st.error("No se pudo procesar ningún PDF.")
        st.stop()

    raw, long, item_stats, cat_stats = build_dataframes(records)
    valid_answers = int(long["calificacion"].count())
    global_avg = long["calificacion"].mean()
    global_pct = global_avg / 5 * 100 if pd.notna(global_avg) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF procesados", f"{len(raw)}")
    c2.metric("Respuestas válidas", f"{valid_answers}")
    c3.metric("Promedio global", f"{global_avg:.2f} / 5" if pd.notna(global_avg) else "Sin datos")
    c4.metric("Satisfacción global", f"{global_pct:.1f}%" if pd.notna(global_pct) else "Sin datos")

    st.caption(f"Clasificación global: **{satisfaction_label(global_pct)}** · {DEVELOPER_LINE}")

    tabs = st.tabs(["Resumen ejecutivo", "Estadística por ítem", "Distribución Likert", "Opiniones", "Datos y exportación"])

    with tabs[0]:
        col_a, col_b = st.columns([1.15, 1])
        with col_a:
            fig_cat = px.bar(
                cat_stats.sort_values("satisfaccion_pct", ascending=True),
                x="satisfaccion_pct",
                y="categoria",
                orientation="h",
                text=cat_stats.sort_values("satisfaccion_pct", ascending=True)["satisfaccion_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
                labels={"satisfaccion_pct": "% de satisfacción", "categoria": "Categoría"},
                title="Satisfacción por categoría"
            )
            fig_cat.update_layout(height=420, margin=dict(l=10, r=20, t=55, b=10), xaxis_range=[0, 100])
            st.plotly_chart(fig_cat, use_container_width=True)
        with col_b:
            best = item_stats.sort_values("satisfaccion_pct", ascending=False).head(3)
            low = item_stats.sort_values("satisfaccion_pct", ascending=True).head(3)
            st.subheader("Hallazgos rápidos")
            st.markdown("**Ítems mejor evaluados**")
            for _, r in best.iterrows():
                st.write(f"Ítem {int(r.numero_item)}: {r.satisfaccion_pct:.1f}% · {r.enunciado}")
            st.markdown("**Ítems con mayor oportunidad de mejora**")
            for _, r in low.iterrows():
                st.write(f"Ítem {int(r.numero_item)}: {r.satisfaccion_pct:.1f}% · {r.enunciado}")

    with tabs[1]:
        fig_items = px.bar(
            item_stats,
            x="numero_item",
            y="satisfaccion_pct",
            text=item_stats["satisfaccion_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
            hover_data=["enunciado", "promedio", "respuestas_validas"],
            labels={"numero_item": "Ítem", "satisfaccion_pct": "% satisfacción"},
            title="Satisfacción promedio por ítem"
        )
        fig_items.update_layout(yaxis_range=[0, 100], height=470)
        st.plotly_chart(fig_items, use_container_width=True)

        display_stats = item_stats.copy()
        display_stats["promedio"] = display_stats["promedio"].round(2)
        display_stats["satisfaccion_pct"] = display_stats["satisfaccion_pct"].round(1)
        display_stats["desviacion"] = display_stats["desviacion"].round(2)
        st.dataframe(display_stats, use_container_width=True, hide_index=True)

    with tabs[2]:
        dist = long.dropna(subset=["calificacion"]).groupby(["numero_item", "calificacion"], as_index=False).size()
        dist["respuesta"] = dist["calificacion"].map(LIKERT_MAP)
        fig_dist = px.bar(
            dist,
            x="numero_item",
            y="size",
            color="respuesta",
            barmode="stack",
            labels={"numero_item": "Ítem", "size": "Cantidad de respuestas", "respuesta": "Respuesta"},
            title="Distribución de respuestas por ítem"
        )
        fig_dist.update_layout(height=500)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tabs[3]:
        st.subheader("Respuestas abiertas")
        q16 = raw[["estudiante", "archivo", "opinion_valioso"]].rename(columns={"opinion_valioso": "Comentario"})
        q17 = raw[["estudiante", "archivo", "opinion_mejora"]].rename(columns={"opinion_mejora": "Comentario"})
        col_q16, col_q17 = st.columns(2)
        with col_q16:
            st.markdown(f"**16. {OPINION_16}**")
            st.dataframe(q16[q16["Comentario"].fillna("").str.len() > 0], use_container_width=True, hide_index=True)
        with col_q17:
            st.markdown(f"**17. {OPINION_17}**")
            st.dataframe(q17[q17["Comentario"].fillna("").str.len() > 0], use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Datos procesados")
        st.dataframe(raw, use_container_width=True, hide_index=True)

        excel_bytes = make_excel(raw, long, item_stats, cat_stats)
        pdf_bytes = make_pdf_report(raw, long, item_stats, cat_stats, course_name, section_name)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "Descargar base y estadísticas en Excel",
                data=excel_bytes,
                file_name=f"informe_satisfaccion_AVE_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col_exp2:
            st.download_button(
                "Descargar informe ejecutivo en PDF",
                data=pdf_bytes,
                file_name=f"informe_satisfaccion_AVE_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

    st.markdown(f"<div class='footer-dev'>{DEVELOPER_LINE} · Herramienta local para análisis estadístico AVE-UVG</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
