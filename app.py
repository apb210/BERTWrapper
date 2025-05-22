from transformers import AutoTokenizer, AutoModel
import torch
import docx
import fitz
import pandas as pd
import tempfile
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load LegalBERT
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# --- STREAMLIT UI ---
st.title("🔍 NDA Compliance Checker with LegalBERT")

uploaded_file = st.file_uploader("Upload an NDA (.docx or .pdf)", type=["docx", "pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Extract NDA text
    if uploaded_file.name.endswith(".pdf"):
        document_text = extract_text_from_pdf(tmp_path)
    else:
        document_text = extract_text_from_docx(tmp_path)

    # Load standards
    term_sheet_file = "standards.csv"
    term_sheet_df = pd.read_csv(term_sheet_file)

    # Compare with LegalBERT
    doc_embedding = encode_text(document_text)
    results = []

    for _, row in term_sheet_df.iterrows():
        issue = str(row.get('Issue', '')).strip()
        preferred = str(row.get('Preferred Language', '')).strip()

        if not preferred or preferred.lower() == 'nan':
            continue

        preferred_embedding = encode_text(preferred)
        score = cosine_similarity([doc_embedding], [preferred_embedding])[0][0]
        status = "Compliant" if score > 0.75 else ("Non-compliant" if score > 0.5 else "Missing")

        results.append({
            "Issue": issue,
            "Compliance Status": status,
            "Similarity Score": f"{score:.2f}",
            "Preferred Language": preferred
        })

    compliance_df = pd.DataFrame(results)
    st.subheader("🧾 Compliance Table")
    st.dataframe(compliance_df, use_container_width=True)
