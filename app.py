import logging
import re

import nltk
from datasets import load_dataset
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    tokens = [re.sub(r"[^a-z0-9]", "", token) for token in tokens]
    tokens = [token for token in tokens if token and token not in STOP_WORDS]
    return [STEMMER.stem(token) for token in tokens]


def load_search_index():
    logger.info("Loading NFCorpus from Hugging Face...")
    corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    documents = {
        row["_id"]: {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }
        for row in corpus
    }

    document_ids = list(documents)
    tokenized_documents = [
        preprocess(documents[doc_id]["title"] + " " + documents[doc_id]["text"])
        for doc_id in document_ids
    ]

    logger.info("Building TF-IDF and BM25 indexes for %s documents...", len(document_ids))
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(
        [" ".join(tokens) for tokens in tokenized_documents]
    )
    bm25_index = BM25Okapi(tokenized_documents, k1=1.5, b=0.75)
    return documents, document_ids, vectorizer, tfidf_matrix, bm25_index


app = Flask(__name__)

for resource in ("punkt", "punkt_tab", "stopwords"):
    if not nltk.download(resource, quiet=True, raise_on_error=True):
        raise RuntimeError(f"Could not download NLTK resource: {resource}")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
(
    docs,
    doc_ids,
    tfidf_vectorizer,
    tfidf_matrix,
    bm25,
) = load_search_index()


def search_tfidf(query: str, top_k: int) -> list[dict]:
    query_tokens = preprocess(query)
    query_vector = tfidf_vectorizer.transform([" ".join(query_tokens)])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {
            "rank": rank,
            "document_id": doc_ids[index],
            "score": round(float(scores[index]), 6),
            "title": docs[doc_ids[index]]["title"],
            "snippet": docs[doc_ids[index]]["text"][:200],
        }
        for rank, index in enumerate(top_indices, start=1)
    ]


def search_bm25(query: str, top_k: int) -> list[dict]:
    scores = bm25.get_scores(preprocess(query))
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {
            "rank": rank,
            "document_id": doc_ids[index],
            "score": round(float(scores[index]), 6),
            "title": docs[doc_ids[index]]["title"],
            "snippet": docs[doc_ids[index]]["text"][:200],
        }
        for rank, index in enumerate(top_indices, start=1)
    ]


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "corpus": "NFCorpus",
            "docs": len(doc_ids),
            "models": ["tfidf", "bm25"],
        }
    )


@app.post("/search")
def search():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "El cuerpo debe ser un objeto JSON."}), 400

    query = body.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "El campo 'query' es obligatorio."}), 400

    model = body.get("model", "bm25")
    if not isinstance(model, str) or model.lower() not in ("bm25", "tfidf"):
        return jsonify({"error": "El campo 'model' debe ser 'bm25' o 'tfidf'."}), 400
    model = model.lower()

    top_k = body.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
        return jsonify({"error": "'top_k' debe ser un entero entre 1 y 1000."}), 400

    results = (
        search_bm25(query.strip(), top_k)
        if model == "bm25"
        else search_tfidf(query.strip(), top_k)
    )
    return jsonify({"model": model, "query": query.strip(), "results": results})
