"""Portable, printable research report built only from available case evidence."""

import base64
import html
import io

import numpy as np
from PIL import Image

from app.data import (annotation_volume, initial_slice, rgb_slice, upload_rgb_slice,
                      upload_gradcam_slice, upload_pancreas_slice, upload_display_mode)


def build_report(case, array, labels, spacing, notes=''):
    escape = lambda value: html.escape(str(value))
    value = annotation_volume(labels, spacing)
    measurement = f'{value:.2f} cm³' if value is not None else 'Unavailable'
    z = initial_slice(array, labels)
    figures = []
    for title, mode in [('CT slice', 'ct'), ('Pancreas annotation', 'pancreas'), ('Lesion annotation', 'lesion')]:
        buffer = io.BytesIO()
        Image.fromarray(rgb_slice(array, labels, z, mode)).save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        figures.append(f'<figure><figcaption>{title}</figcaption><img src="data:image/png;base64,{encoded}" '
                       f'alt="{title}"></figure>')
    rows = [('Study ID', case['study_id']), ('Patient ID', case.get('patient_id', '')),
            ('Source diagnosis label', case.get('diagnosis', 'Unavailable')),
            ('Patient split', case.get('split', 'Unavailable')),
            ('Lesion annotation tier', case.get('annotation_tier', 'Unavailable'))]
    table = ''.join(f'<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>' for k, v in rows)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>PanVision — {escape(case['study_id'])}</title>
<style>@page{{size:A4;margin:15mm}}body{{font:13px Arial,sans-serif;color:#173858;max-width:820px;margin:28px auto}}
header{{background:#173f61;color:white;padding:20px 24px;display:flex;justify-content:space-between}}
h1{{margin:0;font:28px Georgia,serif}}header p{{margin:6px 0 0;font-size:11px}}h2{{font-size:14px;background:#e8f1fc;padding:10px;margin:20px 0 10px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce5ee;padding:9px;text-align:left}}th{{width:38%;background:#f5f8fc}}
.images{{display:flex;gap:10px}}figure{{flex:1;min-width:0;margin:0;border:1px solid #dce5ee;padding:5px}}img{{width:100%}}figcaption{{font-size:11px;margin:4px}}
.note{{white-space:pre-wrap}}small{{color:#667f97}}section{{break-inside:avoid}}@media print{{body{{margin:0}}*{{print-color-adjust:exact}}}}</style></head><body>
<header><div><h1>PanVision</h1><p>Pancreatic CT research</p></div><div>Research Case Report<p>Dataset evidence · not a clinical diagnosis</p></div></header>
<section><h2>1. Study information</h2><table>{table}</table></section>
<section><h2>2. Available evidence</h2><p>Preprocessed CT and supplied dataset annotations are available. Classification probability,
confidence, segmentation predictions, Grad-CAM, and similar-case retrieval are unavailable.</p>
<table><tr><th>Annotated lesion volume</th><td>{measurement}</td></tr><tr><th>Voxel spacing (XYZ)</th>
<td>{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm</td></tr></table>
<p><small>Volume is calculated from the annotation voxel count and spacing. Diameter and anatomical location have not been computed.</small></p></section>
<section><h2>3. Visual results</h2><div class="images">{''.join(figures)}</div>
<p><small>Slice {z+1} of {len(array)}. Green: pancreas annotation. Coral: lesion annotation. Images are for visual review, not measurement.</small></p></section>
<section><h2>4. Review notes</h2><p class="note">{escape(notes) if notes else 'No review notes recorded.'}</p></section>
<p><small>Academic research prototype. Not for clinical diagnosis or treatment. Source labels and annotations are not model predictions.</small></p></body></html>'''


def build_pdf(case, array, labels, spacing, notes='', display_mode='auto', pancreas_mask=None,
              predicted_class=None, pdac_probability=None, threshold=None,
              gradcam=None, gradcam_slice=None, similar_cases=None):
    """Return a portable PDF using real case evidence and no external services."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage

    uploaded = case.get('source') == 'upload'
    if pancreas_mask is not None and (not uploaded or pancreas_mask.shape != array.shape):
        raise ValueError('Pancreas mask does not match the uploaded scan.')
    if gradcam is not None:
        if not uploaded or gradcam.shape != array.shape[1:] or not np.isfinite(gradcam).all():
            raise ValueError('Grad-CAM does not match the uploaded scan.')
        if gradcam_slice is None or not 0 <= int(gradcam_slice) < len(array):
            raise ValueError('Grad-CAM slice is outside the uploaded scan.')
    if pdac_probability is not None and not 0 <= float(pdac_probability) <= 1:
        raise ValueError('PDAC probability must be between zero and one.')
    if threshold is not None and not 0 < float(threshold) < 1:
        raise ValueError('Classification threshold must be between zero and one.')
    similar_cases = list(similar_cases or [])[:3]
    pancreas_region = pancreas_mask is not None and bool(pancreas_mask.any())
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('Section', fontSize=12, leading=17, textColor=colors.HexColor('#354b35'),
                              backColor=colors.HexColor('#e5e8d5'), spaceBefore=15, spaceAfter=10, borderPadding=7))
    styles.add(ParagraphStyle('SmallNote', fontSize=9, leading=13, textColor=colors.HexColor('#536248')))
    styles.add(ParagraphStyle('FigureLabel', fontSize=9, leading=12, alignment=TA_CENTER))
    def paragraph(value, style='BodyText'):
        return Paragraph(html.escape(str(value)).replace('\n', '<br/>'), styles[style])
    story = [paragraph('PanVision', 'Title'), paragraph('Pancreatic CT Research Report', 'Heading2'),
             paragraph('Research use only. Model overlays are estimates, not a clinical diagnosis.', 'SmallNote')]
    def section(title):
        story.append(paragraph(title, 'Section'))
    def table(rows):
        result = Table([[paragraph(k), paragraph(v)] for k, v in rows], colWidths=[175, 340], hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f6f1e3')),
                                   ('GRID', (0, 0), (-1, -1), .5, colors.HexColor('#d4d9c0')),
                                   ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('TOPPADDING', (0, 0), (-1, -1), 8),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
        story.append(result)
    section('1. Study information')
    table([('Study ID', case['study_id'])] + ([('Patient number', case.get('patient_id', 'Not provided'))] if uploaded else [])
          + [('Source', 'Uploaded scan' if uploaded else 'PANORAMA dataset')]
          + ([] if uploaded else [('Patient ID', case.get('patient_id', 'Unavailable')),
           ('Patient split', case.get('split', 'Unavailable')),
           ('Annotation tier', case.get('annotation_tier', 'Unavailable'))]))
    section('2. Diagnosis and measurements')
    value = annotation_volume(labels, spacing)
    rows = [('Diagnosis' if uploaded else 'Dataset diagnosis', case.get('diagnosis', 'Unavailable')),
            ('Model prediction', predicted_class or 'Not available')]
    if uploaded:
        rows.extend([
            ('PDAC probability', f'{float(pdac_probability):.1%}' if pdac_probability is not None else 'Not available'),
            ('Decision threshold', f'{float(threshold):.1%}' if threshold is not None else 'Not available'),
        ])
    if uploaded:
        rows.append(('Pancreas segmentation', 'Model overlay included' if pancreas_region else
                     'No region marked by model' if pancreas_mask is not None else 'Not run'))
    else:
        rows.append(('Annotated lesion volume', f'{value:.2f} cm3' if value is not None else 'Unavailable'))
    rows.append(('Voxel spacing (XYZ)', ' x '.join(f'{v:.2f}' for v in spacing) + ' mm'))
    table(rows)
    evidence = ('Uploaded volume passed structural validation. The pancreas overlay and Grad-CAM are model estimates. '
                'Grad-CAM shows classifier attention and is not a lesion segmentation.' if uploaded
                else 'Diagnosis is the PANORAMA source label. Volume comes from supplied annotations.')
    story.extend([Spacer(1, 8), paragraph(evidence, 'SmallNote')])
    section('3. Visual results')
    z = (int(pancreas_mask.sum(axis=(1, 2)).argmax()) if uploaded and pancreas_region
         else initial_slice(array, labels))
    images, captions = [], []
    panels = ([('Uploaded CT', 'ct')] + ([('Predicted pancreas overlay', 'pancreas')] if pancreas_region else [])
              if uploaded else [('CT scan', 'ct'), ('Pancreas annotation', 'pancreas'), ('Lesion annotation', 'lesion')])
    for title, mode in panels:
        buffer = io.BytesIO()
        pixels = (upload_pancreas_slice(array, pancreas_mask, z, display_mode) if uploaded and mode == 'pancreas'
                  else upload_rgb_slice(array, z, display_mode) if uploaded else rgb_slice(array, labels, z, mode))
        Image.fromarray(pixels).save(buffer, format='PNG')
        buffer.seek(0)
        image = PDFImage(buffer)
        ratio = 160 / max(image.imageWidth, image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        images.append(image)
        captions.append(paragraph(title, 'FigureLabel'))
    figures = Table([images, captions], colWidths=[515/len(panels)]*len(panels))
    figures.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(figures)
    resolved = upload_display_mode(array) if display_mode == 'auto' else display_mode
    display_label = 'normalized 0-1 display' if resolved == 'normalized' else '-100 to 300 HU display window'
    legend = (f'Uploaded CT shown with {display_label}. Green: model-estimated pancreas.' if uploaded and pancreas_region
              else f'Uploaded CT shown with {display_label}. No pancreas model output included.' if uploaded
              else 'Green: pancreas. Coral: lesion. Dataset annotations, not predicted segmentations.')
    story.append(paragraph(f'Slice {z+1} of {len(array)}. ' + legend, 'SmallNote'))
    next_section = 4
    if uploaded and gradcam is not None:
        section(f'{next_section}. Model attention')
        next_section += 1
        buffer = io.BytesIO()
        pixels = upload_gradcam_slice(array, gradcam, int(gradcam_slice), display_mode)
        Image.fromarray(pixels).save(buffer, format='PNG')
        buffer.seek(0)
        attention = PDFImage(buffer)
        ratio = 300 / max(attention.imageWidth, attention.imageHeight)
        attention.drawWidth, attention.drawHeight = attention.imageWidth * ratio, attention.imageHeight * ratio
        attention.hAlign = 'CENTER'
        story.extend([attention, paragraph(
            f'Grad-CAM on slice {int(gradcam_slice)+1} of {len(array)}. The heatmap was computed on the '
            'DenseNet pancreas crop and mapped back to the original CT for anatomical context.', 'SmallNote')])
    section(f'{next_section}. Similar reference cases' if uploaded else f'{next_section}. Review notes')
    if uploaded:
        if similar_cases:
            rows = [['Patient ID', 'Dataset label', 'Cosine similarity']] + [[
                str(item['patient_id']), 'PDAC' if int(item['label']) == 1 else 'Non-PDAC',
                f'{float(item["similarity"]):.4f}'] for item in similar_cases]
            result = Table([[paragraph(value) for value in row] for row in rows],
                           colWidths=[180, 165, 170], hAlign='LEFT')
            result.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e8d5')),
                                        ('GRID', (0, 0), (-1, -1), .5, colors.HexColor('#d4d9c0')),
                                        ('TOPPADDING', (0, 0), (-1, -1), 7),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
            story.append(result)
        else:
            story.append(paragraph('No similar cases were returned.'))
        story.append(paragraph('Reference cases are cosine-nearest training embeddings and are not diagnostic evidence.',
                               'SmallNote'))
        next_section += 1
        section(f'{next_section}. Review notes')
    story.append(paragraph(notes or 'No review notes recorded.'))
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#536248'))
        canvas.drawString(40, 24, 'PanVision | Academic research prototype | Not for clinical use')
        canvas.drawRightString(A4[0]-40, 24, str(doc.page))
    SimpleDocTemplate(output, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=32, bottomMargin=42,
                      title=f"PanVision - {case['study_id']}").build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
