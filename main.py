import os
import io
import pickle
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import re

# ─── Configuration du logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "./vector_db"
os.makedirs(BASE_DIR, exist_ok=True)

# ─── Modèle instruct QA ───────────────────────────────────────────────────────
try:
    instruct_tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
    instruct_model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-base")
    logging.info(" Modèle instruct chargé.")
except Exception as e:
    logging.error(f" Erreur de chargement modèle instruct : {e}")
    raise

# ─── Lazy loading du modèle embeddings ────────────────────────────────────────
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logging.info(" Chargement du modèle paraphrase-MiniLM-L3-v2...")
        _embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return _embed_model

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)
    return text.strip()

def build_faiss_index(chunks: list[str]):
    try:
        model = get_embed_model()
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        vectors = model.encode(chunks, show_progress_bar=False)
        index.add(vectors)
        return index
    except Exception as e:
        logging.error(f" Erreur création index FAISS : {e}")
        raise

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        data = await file.read()
        reader = PdfReader(io.BytesIO(data))
        text = "".join(p.extract_text() or "" for p in reader.pages)
        cleaned_text = clean_text(text)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_text(cleaned_text)

        logging.info(f" Fichier reçu : {file.filename}, {len(chunks)} chunks générés.")

        faiss_index = build_faiss_index(chunks)

        doc_id = file.filename.replace(".pdf", "") + "_" + os.urandom(4).hex()
        persist_path = os.path.join(BASE_DIR, doc_id)
        os.makedirs(persist_path, exist_ok=True)

        with open(os.path.join(persist_path, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)
        with open(os.path.join(persist_path, "index.pkl"), "wb") as f:
            pickle.dump(faiss_index, f)

        logging.info(f" Document indexé : {doc_id}")
        return {"doc_id": doc_id}

    except Exception as e:
        logging.error(f" Erreur upload_pdf : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement du fichier PDF.")

@app.get("/query")
async def query(doc_id: str, q: str = Query(..., description="Votre question")):
    try:
        persist_dir = os.path.join(BASE_DIR, doc_id)
        if not os.path.isdir(persist_dir):
            raise HTTPException(status_code=404, detail="Document introuvable.")

        with open(os.path.join(persist_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        with open(os.path.join(persist_dir, "index.pkl"), "rb") as f:
            faiss_index = pickle.load(f)

        model = get_embed_model()
        q_vec = model.encode([q])
        D, I = faiss_index.search(q_vec, 3)

        selected_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(selected_chunks)

        prompt = f"Document:\n{context}\n\nQuestion: {q}\nRéponds de manière claire :"
        inputs = instruct_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = instruct_model.generate(**inputs, max_new_tokens=200)
        answer = instruct_tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info(f" Réponse générée pour doc_id={doc_id}")
        return {"answer": answer.strip() or "❌ Réponse vide."}

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f" Erreur query : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {e}")
