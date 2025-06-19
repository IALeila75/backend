import os
import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from dotenv import load_dotenv
import openai
import re

# ─── Chargement de la clé API OpenAI ─────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Configuration de FastAPI ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Stockage en mémoire des documents ───────────────────────────────────────
document_store = {}

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        data = await file.read()
        reader = PdfReader(io.BytesIO(data))
        text = "".join(p.extract_text() or "" for p in reader.pages)
        cleaned_text = clean_text(text)
        chunks = split_text(cleaned_text)

        doc_id = file.filename.replace(".pdf", "") + "_" + os.urandom(4).hex()
        document_store[doc_id] = chunks
        logging.info(f" Document {doc_id} indexé avec {len(chunks)} morceaux.")
        return {"doc_id": doc_id}
    except Exception as e:
        logging.error(f" Erreur upload : {e}")
        raise HTTPException(status_code=500, detail="Erreur de lecture du PDF.")

@app.get("/query")
async def query(doc_id: str, q: str = Query(...)):
    try:
        if doc_id not in document_store:
            logging.warning(f"❌ doc_id introuvable : {doc_id}")
            raise HTTPException(status_code=404, detail="Document introuvable.")

        chunks = document_store[doc_id]
        context = "\n\n".join(chunks[:3])  # réduit à 3 morceaux

        messages = [
            {"role": "system", "content": "Tu es un assistant qui répond à partir d'un document PDF."},
            {"role": "user", "content": f"Document :\n{context}\n\nQuestion : {q}"}
        ]

        logging.info(f" Question reçue : {q}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message["content"]
        logging.info(f" Réponse : {answer[:80]}...")
        return {"answer": answer}

    except openai.error.InvalidRequestError as e:
        logging.error(f" OpenAI error: {e}")
        raise HTTPException(status_code=400, detail="Erreur OpenAI : prompt trop long ou invalide.")
    except Exception as e:
        logging.error(f" Erreur serveur : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne.")







