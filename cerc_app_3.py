import streamlit as st
from mistralai import Mistral
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json
import os
import tempfile
import shutil
import time
from dataclasses import dataclass
from typing import Literal, Optional
from fpdf import FPDF

# ============================================================
# 📄 CLASSE GÉNÉRATION PDF (CORRIGÉE POUR UNICODE)
# ============================================================

class CSR_PDF(FPDF):
    def __init__(self):
        super().__init__()
        # On utilise une police standard qui supporte l'UTF-8
        # fpdf2 inclut nativement des polices de remplacement si besoin, 
        # mais le plus simple est d'activer le mode core fonts ou d'utiliser 'helvetica'
        # qui gère mieux l'encoding en mode 'latin-1' de base via des remplacements.
        self.set_font("helvetica", size=12)

    def header(self):
        self.set_font("helvetica", "B", 8)
        self.set_text_color(128)
        self.cell(0, 10, "CONFIDENTIAL - Clinical Study Report (CSR) - nom_entrprise_XXX AI", 0, 1, "R")

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def add_section_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 12, title.upper(), 0, 1, "L", fill=True)
        self.ln(5)

    def add_body_text(self, text):
        self.set_font("helvetica", "", 11)
        # NETTOYAGE CRITIQUE : Remplace les caractères Unicode non-Latin1 
        # par leurs équivalents lisibles pour éviter le crash
        safe_text = (
            text.replace("\u2265", ">=")
                .replace("\u2264", "<=")
                .replace("\u2212", "-")
                .replace("\u03b1", "alpha")
                .replace("\u03b2", "beta")
                .replace("\u03bc", "mu")
                .replace("\xb1", "+/-")
                .replace("**", "")
                .replace("#", "")
        )
        # encode('latin-1', 'replace') évite le plantage final au cas où
        clean_text = safe_text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 7, clean_text)
        self.ln(5)

# ============================================================
# ⚙️ MODULES D'EXTRACTION (Inchangés)
# ============================================================

tesseract_cmd = shutil.which("tesseract")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

@dataclass
class DetectionResult:
    kind: Literal["pdf_text", "pdf_scanned", "image", "unknown"]
    reason: str
    pages: Optional[int] = None

def detect_file_kind(path: str, threshold: int = 200) -> DetectionResult:
    if not os.path.isfile(path): return DetectionResult("unknown", "Fichier introuvable")
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".tiff"}: return DetectionResult("image", "Format image")
    if ext == ".pdf":
        try:
            doc = fitz.open(path)
            n_pages = doc.page_count
            sample_pages = min(n_pages, 3)
            total_chars = sum([len("".join((doc.load_page(i).get_text("text") or "").split())) for i in range(sample_pages)])
            doc.close()
            if total_chars >= threshold: return DetectionResult("pdf_text", f"Texte ({total_chars} chars)", pages=n_pages)
            else: return DetectionResult("pdf_scanned", "Scan", pages=n_pages)
        except Exception as e: return DetectionResult("unknown", f"Erreur: {e}")
    return DetectionResult("unknown", "Non supporté")

def process_to_text(path: str, det: DetectionResult) -> str:
    try:
        if det.kind == "pdf_text":
            doc = fitz.open(path)
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            return text.strip()
        elif det.kind == "pdf_scanned":
            images = convert_from_path(path, dpi=300)
            return "\n".join([f"--- PAGE {i+1} ---\n{pytesseract.image_to_string(img, lang='fra+eng')}" for i, img in enumerate(images)]).strip()
        elif det.kind == "image":
            return pytesseract.image_to_string(Image.open(path), lang="fra+eng").strip()
    except Exception as e: return f"[Erreur] : {str(e)}"
    return ""

# ============================================================
# 🧠 MODULE 2 : GÉNÉRATION
# ============================================================

def call_mistral_with_retry(client, messages, is_json=False, max_retries=5):
    for attempt in range(max_retries):
        try:
            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            resp = client.chat.complete(model="mistral-small-latest", messages=messages, response_format=response_format)
            content = resp.choices[0].message.content
            return json.loads(content) if is_json else content
        except Exception as e:
            if "429" in str(e): time.sleep((2 ** attempt) + 2)
            else: return None
    return None

def map_documents_to_sections(client, extracted_data, map_instr):
    docs_summary = "\n".join([f"- {name} (Extrait: {text[:500]}...)" for name, text in extracted_data.items()])
    sections_summary = "\n".join([f"- {k}: {v}" for k, v in map_instr.items()])
    prompt = f"""Tu es un Medical Writer expert. Analyse les documents et attribue-les aux sections.
    Réponds UNIQUEMENT en JSON: {{"mapping": {{"Nom Section": ["file1.pdf"]}}}}
    [DOCS] {docs_summary} [SECTIONS] {sections_summary}"""
    result = call_mistral_with_retry(client, [{"role": "user", "content": prompt}], is_json=True)
    return result.get("mapping", {}) if result else {}

def generate_csr_section(client, section_name, source_docs, template_instructions):
    context = "\n\n".join([f"### SOURCE: {name}\n{txt[:12000]}" for name, txt in source_docs.items()])
    prompt = f"""Tu es un Senior Medical Writer chargé de rédiger une section de CSR professionnel (ICH E3).
NOM DE LA SECTION : {section_name}. OBJECTIF : {template_instructions}.
[SOURCES] : {context}. [RÈGLES] : ÉTon neutre, passé, citations de chiffres/tables, format Markdown, pas de spéculation.vite les caractères spéciaux complexes si possible.Le contenu doit etre en français.
Rédige la section en respectant les règles et en utilisant les sources. Soit Scientifique, rigoureux et concis. Utilise des titres secondaires si nécessaire. Si les sources sont insuffisantes, indique clairement les manques d'information.
Rédige le contenu maintenant :"""
    return call_mistral_with_retry(client, [{"role": "user", "content": prompt}], is_json=False)

# ============================================================
# 🚀 INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="nom_entrprise_XXX AI Assistant", layout="wide", page_icon="🏥")
st.title("🏥 nom_entrprise_XXX - Expert CSR PDF Generator")

with st.sidebar:
    api_key = st.text_input("Clé Mistral API", type="password")
    uploaded_files = st.file_uploader("Sources", accept_multiple_files=True)

if not api_key or not uploaded_files:
    st.warning("Veuillez configurer l'API et charger des documents.")
    st.stop()

client = Mistral(api_key=api_key)

map_instr = {
    "1. Summary (Synopsis)": "Résumé structuré de l'étude.",
    "2. Investigators and Administrative Structure": "Organisation, promoteur, centres.",
    "3. Introduction": "Rationnel et besoins cliniques.",
    "4. Investigational Device and Methods": "Description du dispositif et procédure.",
    "5. Clinical Investigation Plan (CIP) Design": "Design, objectifs et critères d'éligibilité.",
    "6. Statistical Methods planned in the CIP": "Méthodes statistiques, taille échantillon.",
    "7. Results (Safety & Efficacy)": "Résultats cliniques détaillés.",
    "8. Discussion and Overall Conclusions": "Interprétation et balance bénéfice/risque.",
    "9. Ethics and Regulatory": "Conformité ISO 14155, Helsinki."
}

if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
            tmp.write(f.getvalue())
            temp_path = tmp.name
        det = detect_file_kind(temp_path)
        st.session_state.extracted_data[f.name] = process_to_text(temp_path, det)
        os.remove(temp_path)

if st.button("🚀 Générer le Rapport PDF Complet", use_container_width=True):
    pdf = CSR_PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Page de garde
    pdf.set_font("helvetica", "B", 24)
    pdf.ln(60)
    pdf.cell(0, 20, "CLINICAL STUDY REPORT (CSR)", 0, 1, "C")
    pdf.set_font("helvetica", "", 16)
    pdf.cell(0, 10, "Generated by nom_entrprise_XXX AI Expert System", 0, 1, "C")
    pdf.ln(10)
    pdf.cell(0, 10, f"Date: {time.strftime('%d/%m/%Y')}", 0, 1, "C")
    
    doc_mapping = map_documents_to_sections(client, st.session_state.extracted_data, map_instr)
    
    progress_bar = st.progress(0)
    for i, (title, instr) in enumerate(map_instr.items()):
        relevant_docs = {name: st.session_state.extracted_data[name] for name in doc_mapping.get(title, []) if name in st.session_state.extracted_data}
        if not relevant_docs: relevant_docs = st.session_state.extracted_data

        content = generate_csr_section(client, title, relevant_docs, instr)
        
        if content:
            pdf.add_page()
            pdf.add_section_title(title)
            pdf.add_body_text(content)
            st.success(f"Section {title} rédigée.")
        
        progress_bar.progress((i + 1) / len(map_instr))

    # Sauvegarde et téléchargement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                label="📥 Télécharger le Rapport CSR (PDF)",
                data=f,
                file_name="Clinical_Study_Report_Expert.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    os.remove(tmp_pdf.name)


    
