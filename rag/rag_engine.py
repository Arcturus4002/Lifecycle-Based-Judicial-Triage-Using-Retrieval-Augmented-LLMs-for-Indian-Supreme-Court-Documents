"""
rag_engine.py
=============
Fast RAG engine for 26k+ legal judgments.
GPU embeddings (CUDA), 8 parallel PDF extractors, resume support.

Stop llama.cpp server before indexing (frees GPU VRAM).

Usage:
    python rag_engine.py --index "C:/path/to/cases"
    python rag_engine.py --search "murder conviction section 302"
    python rag_engine.py
"""

import os
import json
import hashlib
import argparse
import time
import re
import sys
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INDEX_DIR = Path("./rag_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
EMBED_BATCH_SIZE = 16   # larger batch = faster on GPU
EXTRACT_WORKERS = 8
SAVE_EVERY = 1000         # checkpoint every N PDFs
DEVICE = "cuda"           # cuda or cpu


# ──────────────────────────────────────────────
# TEXT EXTRACTION (standalone function for multiprocessing)
# ──────────────────────────────────────────────
def _extract_one(pdf_path: str) -> tuple[str, str]:
    """Extract text from one PDF. Returns (path, text)."""
    import pdfplumber
    import re as _re

    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
                except Exception:
                    continue
    except Exception:
        pass

    if len(text.strip()) < 50:
        try:
            from pypdf import PdfReader
            for page in PdfReader(pdf_path).pages:
                try:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
                except Exception:
                    continue
        except Exception:
            pass

    text = _re.sub(r'Indian Kanoon\s*-\s*http://indiankanoon\.org/\S*\s*\d*', '', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return (pdf_path, text)


# ──────────────────────────────────────────────
# CHUNKING
# ──────────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        ct = " ".join(words[start:end])
        if len(ct.strip()) > 30:
            chunks.append({
                "text": ct,
                "start_word": start,
                "end_word": min(end, len(words)),
                "total_words": len(words),
            })
        start += chunk_size - overlap
        if end >= len(words):
            break
    return chunks


# ──────────────────────────────────────────────
# EMBEDDING
# ──────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            print("  CUDA not available, falling back to CPU")
            device = "cpu"

        print(f"Loading {EMBED_MODEL} on {device.upper()}...")
        _model = SentenceTransformer(EMBED_MODEL, device=device)

        if device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

        dim = _model.get_sentence_embedding_dimension()
        print(f"  Embedding dim: {dim}")
    return _model


def embed_texts(texts, batch_size=EMBED_BATCH_SIZE):
    model = get_model()
    # Truncate long texts to prevent OOM (embedding only needs first ~256 tokens)
    truncated = [t[:1500] if len(t) > 1500 else t for t in texts]
    emb = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=len(truncated) > 200,
        normalize_embeddings=True,
    )
    return np.array(emb, dtype=np.float32)


# ──────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────
def find_pdfs(folder):
    folder = Path(folder)
    pdfs = set()
    for ext in ("*.pdf", "*.PDF"):
        pdfs.update(str(p) for p in folder.rglob(ext))
    return sorted(pdfs)


# ──────────────────────────────────────────────
# PROGRESS
# ──────────────────────────────────────────────
def _progress_path(index_dir):
    return Path(index_dir) / "progress.json"

def load_progress(index_dir):
    p = _progress_path(index_dir)
    if p.exists():
        with open(p, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(index_dir, done):
    with open(_progress_path(index_dir), "w") as f:
        json.dump(list(done), f)


# ──────────────────────────────────────────────
# BUILD INDEX
# ──────────────────────────────────────────────
def build_index(pdf_folder, index_dir=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import faiss

    if index_dir is None:
        index_dir = INDEX_DIR
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Find PDFs
    print(f"Scanning {pdf_folder}...")
    all_pdfs = find_pdfs(pdf_folder)
    print(f"Found {len(all_pdfs):,} PDFs")
    if not all_pdfs:
        print("No PDFs found")
        return

    # Resume
    done = load_progress(index_dir)
    remaining = [p for p in all_pdfs if p not in done]
    if done:
        print(f"Resuming: {len(done):,} done, {len(remaining):,} remaining")
    if not remaining:
        print("All PDFs already indexed")
        return

    # Load existing data
    all_meta = []
    all_texts = []
    old_embeddings = None

    meta_path = index_dir / "metadata.json"
    texts_path = index_dir / "texts.json"
    embeddings_path = index_dir / "embeddings.npy"

    if meta_path.exists() and done:
        print("Loading existing index...")
        with open(meta_path, "r", encoding="utf-8") as f:
            all_meta = json.load(f)
        with open(texts_path, "r", encoding="utf-8") as f:
            all_texts = json.load(f)
        if embeddings_path.exists():
            old_embeddings = np.load(str(embeddings_path))
        print(f"  {len(all_texts):,} existing chunks loaded")

    # ── PHASE 1: Extract text ──
    total = len(remaining)
    t0 = time.time()
    new_meta = []
    new_texts = []
    processed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"  Phase 1: extracting text ({total:,} PDFs, {EXTRACT_WORKERS} workers)")
    print(f"{'='*60}")

    batch_idx = 0
    while batch_idx < total:
        batch_end = min(batch_idx + SAVE_EVERY, total)
        batch = remaining[batch_idx:batch_end]

        with ProcessPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            futures = {pool.submit(_extract_one, p): p for p in batch}
            for fut in as_completed(futures):
                processed += 1
                try:
                    pdf_path, text = fut.result()
                except Exception:
                    processed += 0
                    skipped += 1
                    done.add(futures[fut])
                    continue

                done.add(pdf_path)

                if len(text) < 50:
                    skipped += 1
                    continue

                chunks = chunk_text(text)
                name = Path(pdf_path).name
                doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]

                for ch in chunks:
                    new_meta.append({
                        "source_file": name,
                        "doc_id": doc_id,
                        "start_word": ch["start_word"],
                        "end_word": ch["end_word"],
                        "total_words": ch["total_words"],
                        "text_preview": ch["text"][:200],
                    })
                    new_texts.append(ch["text"])

                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    eta = (total - processed) / rate if rate > 0 else 0
                    print(f"  [{processed:,}/{total:,}] {len(new_texts):,} chunks | "
                          f"{skipped} skipped | {rate:.1f}/s | ETA {eta/60:.0f}m")

        save_progress(index_dir, done)
        batch_idx = batch_end

    extract_time = time.time() - t0
    print(f"\n  Extraction done: {processed:,} PDFs, {len(new_texts):,} new chunks, "
          f"{skipped} skipped, {extract_time/60:.1f} min")

    if not new_texts and not all_texts:
        print("No text extracted")
        return

    # ── PHASE 2: Embed ──
    if new_texts:
        print(f"\n{'='*60}")
        print(f"  Phase 2: embedding {len(new_texts):,} chunks on {DEVICE.upper()}")
        print(f"{'='*60}")

        t1 = time.time()
        new_embeddings = embed_texts(new_texts, EMBED_BATCH_SIZE)
        embed_time = time.time() - t1
        print(f"  Embedded in {embed_time/60:.1f} min ({len(new_texts)/embed_time:.0f} chunks/s)")

        if old_embeddings is not None:
            all_embeddings = np.vstack([old_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings

        all_meta.extend(new_meta)
        all_texts.extend(new_texts)
    else:
        all_embeddings = old_embeddings if old_embeddings is not None else np.array([])
        if all_embeddings.size == 0:
            print("Nothing to index")
            return

    # ── PHASE 3: Build FAISS index ──
    print(f"\n{'='*60}")
    print(f"  Phase 3: building FAISS index ({all_embeddings.shape[0]:,} vectors)")
    print(f"{'='*60}")

    dim = all_embeddings.shape[1]
    n = all_embeddings.shape[0]

    if n > 10000:
        nlist = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(all_embeddings)
        index.add(all_embeddings)
        index.nprobe = min(nlist // 4, 32)
        print(f"  IVF index: nlist={nlist}, nprobe={index.nprobe}")
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(all_embeddings)
        print(f"  Flat index")

    # Save everything
    faiss.write_index(index, str(index_dir / "index.faiss"))
    np.save(str(embeddings_path), all_embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False)
    save_progress(index_dir, done)

    total_time = time.time() - t0
    idx_mb = (index_dir / "index.faiss").stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")
    print(f"  PDFs:      {len(all_pdfs):,} total ({processed:,} this run, {skipped} skipped)")
    print(f"  Chunks:    {len(all_texts):,}")
    print(f"  Docs:      {len(set(m['doc_id'] for m in all_meta)):,}")
    print(f"  Index:     {idx_mb:.1f} MB")
    print(f"  Time:      {total_time/60:.1f} min")
    print(f"  Saved:     {index_dir}/")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────
def load_index(index_dir=None):
    import faiss
    if index_dir is None:
        index_dir = INDEX_DIR
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        return None, None, None

    index = faiss.read_index(str(index_dir / "index.faiss"))
    if hasattr(index, 'nprobe'):
        index.nprobe = min(32, getattr(index, 'nlist', 32) // 4)

    with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    texts = []
    tp = index_dir / "texts.json"
    if tp.exists():
        with open(tp, "r", encoding="utf-8") as f:
            texts = json.load(f)
    return index, metadata, texts


def search(query_text, top_k=TOP_K, index_dir=None, exclude_file=None):
    index, metadata, texts = load_index(index_dir)
    if index is None:
        return []

    qe = embed_texts([query_text])
    n_search = min(top_k * 5, index.ntotal)
    scores, indices = index.search(qe, n_search)

    seen = set()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        did = meta["doc_id"]
        if did in seen:
            continue
        # Skip self-reference
        if exclude_file and meta["source_file"].lower() == exclude_file.lower():
            continue
        seen.add(did)
        snippet = texts[idx][:300] if idx < len(texts) else meta.get("text_preview", "")
        results.append({
            "source_file": meta["source_file"],
            "doc_id": did,
            "score": float(score),
            "snippet": snippet,
            "chunk_position": f"words {meta['start_word']}-{meta['end_word']} of {meta['total_words']}",
        })
        if len(results) >= top_k:
            break
    return results


def get_index_stats(index_dir=None):
    if index_dir is None:
        index_dir = INDEX_DIR
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        return {"indexed": False, "num_chunks": 0, "num_docs": 0}
    with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "indexed": True,
        "num_chunks": len(metadata),
        "num_docs": len(set(m["doc_id"] for m in metadata)),
        "index_size_mb": (index_dir / "index.faiss").stat().st_size / 1e6,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG Engine (GPU)")
    parser.add_argument("--index", type=str, help="PDF folder (recursive)")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    parser.add_argument("--workers", type=int, default=EXTRACT_WORKERS)
    parser.add_argument("--cpu", action="store_true", help="Force CPU embeddings")
    args = parser.parse_args()

    if args.workers:
        EXTRACT_WORKERS = args.workers
    if args.cpu:
        DEVICE = "cpu"

    if args.index:
        build_index(args.index, args.index_dir)
    elif args.search:
        results = search(args.search, args.top_k, args.index_dir)
        if not results:
            print("No results found")
        else:
            print(f"\nTop {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['source_file']} ({r['score']:.4f})")
                print(f"     {r['snippet'][:150]}...")
                print()
    else:
        stats = get_index_stats(args.index_dir)
        if stats["indexed"]:
            print(f"Index: {stats['num_docs']:,} docs, {stats['num_chunks']:,} chunks, {stats['index_size_mb']:.1f} MB")
        else:
            print("No index. Run: python rag_engine.py --index ./pdf_folder")