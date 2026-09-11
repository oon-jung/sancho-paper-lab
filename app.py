"""산초 논문 랩 — 논문 한 편을 올려 파싱·구조·청킹·검색·답변·골드 전사까지 한 흐름으로 확인하는 앱.

배포: Hugging Face Spaces (무료 CPU). 파서는 PyMuPDF와 Docling, 임베딩·답변은 OpenAI.
논문 파일은 앱에 포함하지 않는다. 사용자가 올린 PDF는 메모리에만 두고 디스크에 남기지 않는다.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent
GOLD_DIR = Path(os.environ.get("GOLD_DIR", APP_DIR / "gold_store"))
GOLD_DIR.mkdir(parents=True, exist_ok=True)
PASSWORD = os.environ.get("LAB_PASSWORD", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "40"))
SCALE = 1.5

app = FastAPI(title="산초 논문 랩")
SESSIONS: dict[str, dict[str, Any]] = {}

norm = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


GOLD_DATASET = os.environ.get("GOLD_DATASET", "")  # 예: oon-jung/sancho-gold
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def push_gold(path: Path) -> bool:
    """골드 기록을 HF 데이터셋에 올린다. Space가 다시 뜨면 로컬 파일이 사라지기 때문이다.

    토큰이나 데이터셋 이름이 없으면 아무것도 하지 않는다(로컬 파일만 남는다).
    """
    if not (GOLD_DATASET and HF_TOKEN):
        return False
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(path), path_in_repo="gold.jsonl",
            repo_id=GOLD_DATASET, repo_type="dataset")
        return True
    except Exception:
        return False


def pull_gold(path: Path) -> None:
    """Space가 시작할 때 데이터셋에 있던 골드를 되가져온다."""
    if not (GOLD_DATASET and HF_TOKEN) or path.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=GOLD_DATASET, filename="gold.jsonl",
                              repo_type="dataset", token=HF_TOKEN)
        path.write_text(Path(got).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


pull_gold(GOLD_DIR / "gold.jsonl")


# ──────────────────────────────────────────── 파싱
def parse_pymupdf(doc: pymupdf.Document) -> list[dict]:
    """블록 단위 추출. 순서는 PDF에 기록된 순서 그대로."""
    items = []
    for pno, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        for i, b in enumerate(page.get_text("blocks", sort=False)):
            if len(b) < 5 or not b[4].strip():
                continue
            items.append({
                "id": f"pm{pno}_{i}", "kind": "text", "page": pno + 1,
                "bbox": [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)],
                "text": " ".join(b[4].split()), "level": None,
            })
    return items


_DOCLING = None


def docling_converter():
    global _DOCLING
    if _DOCLING is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        # OCR은 끈다. 본문에 글자층이 있는 논문에서는 구조 출력이 같고 훨씬 빠르다.
        opts = PdfPipelineOptions(do_ocr=False, generate_parsed_pages=False)
        _DOCLING = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _DOCLING


def parse_docling(pdf_bytes: bytes, sizes: dict[int, tuple[float, float]]) -> list[dict]:
    """문단·제목·표·캡션을 구조로 받는다. bbox는 좌하단 원점이라 상단 기준으로 뒤집는다."""
    from docling.datamodel.base_models import DocumentStream
    stream = DocumentStream(name="upload.pdf", stream=io.BytesIO(pdf_bytes))
    result = docling_converter().convert(stream)
    doc = result.document
    items = []
    for idx, (item, _level) in enumerate(doc.iterate_items()):
        label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", "")) or "")
        text = (getattr(item, "text", "") or "").strip()
        html = ""
        if label == "table" and hasattr(item, "export_to_html"):
            try:
                html = item.export_to_html(doc=doc)
            except Exception:
                html = ""
        for pv in getattr(item, "prov", None) or []:
            page = int(pv.page_no)
            if page not in sizes:
                continue
            W, H = sizes[page]
            b = pv.bbox
            x0, x1 = min(b.l, b.r) / W, max(b.l, b.r) / W
            yt, yb = min(b.t, b.b), max(b.t, b.b)
            if getattr(b, "coord_origin", None) and "BOTTOM" in str(b.coord_origin):
                y0, y1 = (H - yb) / H, (H - yt) / H
            else:
                y0, y1 = yt / H, yb / H
            items.append({
                "id": f"dl{idx}", "kind": label, "page": page,
                "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                "text": text, "html": html, "level": getattr(item, "level", None),
            })
    return items


@app.post("/api/parse")
async def api_parse(file: UploadFile = File(...), parser: str = Form("pymupdf"), password: str = Form("")):
    if PASSWORD and password != PASSWORD:
        raise HTTPException(401, "비밀번호가 맞지 않습니다")
    data = await file.read()
    if not data[:5] == b"%PDF-":
        raise HTTPException(400, "PDF 파일이 아닙니다")
    doc = pymupdf.open(stream=data, filetype="pdf")
    if doc.page_count > MAX_PAGES:
        raise HTTPException(400, f"{MAX_PAGES}쪽까지만 처리합니다 (올린 파일은 {doc.page_count}쪽)")
    sizes = {i + 1: (p.rect.width, p.rect.height) for i, p in enumerate(doc)}

    pages = []
    for pno, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE))
        pages.append({"n": pno + 1, "w": pix.width, "h": pix.height,
                      "img": base64.b64encode(pix.tobytes("jpeg", jpg_quality=78)).decode()})

    started = time.perf_counter()
    if parser == "docling":
        try:
            items = parse_docling(data, sizes)
        except Exception as exc:  # 모델 내려받기 실패 등
            raise HTTPException(500, f"docling 실행에 실패했습니다: {exc}")
    else:
        parser = "pymupdf"
        items = parse_pymupdf(doc)
    elapsed = round(time.perf_counter() - started, 2)

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"items": items, "pages": len(pages), "parser": parser,
                     "name": file.filename, "at": time.time()}
    # 오래된 세션 정리 (2시간)
    for key in [k for k, v in SESSIONS.items() if time.time() - v["at"] > 7200]:
        SESSIONS.pop(key, None)

    kinds: dict[str, int] = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    chars = sum(len(it["text"]) for it in items)
    hangul = re.findall(r"[가-힣]+", " ".join(it["text"] for it in items))
    broken = round(sum(1 for t in hangul if len(t) == 1) / max(len(hangul), 1) * 100, 1)
    return {"sid": sid, "parser": parser, "name": file.filename, "pages": pages, "items": items,
            "stats": {"elapsed": elapsed, "per_page": round(elapsed / max(len(pages), 1), 2),
                      "items": len(items), "kinds": kinds, "chars": chars, "broken_spacing": broken}}


# ──────────────────────────────────────────── 청킹
HEAD_KINDS = {"section_header", "title"}
DROP_KINDS = {"page_header", "page_footer", "footnote", "picture"}


def build_chunks(items: list[dict], strategy: str, size: int, overlap: int,
                 table_mode: str, heads: set[str] | None) -> list[dict]:
    """items를 청크로 묶는다.

    절 단위: 제목을 만나면 새 청크를 연다. 표는 table_mode에 따라 따로 두거나 본문에 섞는다.
    고정 N: 문서 전체 텍스트를 N자 창으로 자른다(절 경계를 무시).
    """
    heads = heads or set()
    body = [it for it in items if it["kind"] not in DROP_KINDS]
    chunks: list[dict] = []

    def add(texts, its, section):
        text = "\n".join(t for t in texts if t.strip())
        if not text.strip():
            return
        chunks.append({"id": f"c{len(chunks):04d}", "text": text, "section": section,
                       "pages": sorted({i["page"] for i in its}),
                       "boxes": [{"page": i["page"], "bbox": i["bbox"]} for i in its],
                       "kind": "table" if all(i["kind"] == "table" for i in its) else "body"})

    if strategy == "fixed":
        stream, marks = "", []
        for it in body:
            piece = (it.get("html") if it["kind"] == "table" and table_mode == "html" and it.get("html") else it["text"])
            if not piece:
                continue
            marks.append((len(stream), len(stream) + len(piece), it))
            stream += piece + "\n"
        i = 0
        while i < len(stream):
            j = min(i + size, len(stream))
            piece = stream[i:j]
            if piece.strip():
                its = [it for s, e, it in marks if s < j and e > i]
                add([piece], its or body[:1], "")
            if j >= len(stream):
                break
            i = j - overlap
        return chunks

    cur_texts, cur_items, cur_sec = [], [], ""
    for it in body:
        is_head = it["kind"] in HEAD_KINDS or it["id"] in heads
        if it["kind"] == "table" and table_mode in {"separate", "html"}:
            if cur_texts:
                add(cur_texts, cur_items, cur_sec)
                cur_texts, cur_items = [], []
            piece = it.get("html") if table_mode == "html" and it.get("html") else it["text"]
            add([piece], [it], cur_sec or "표")
            continue
        if is_head:
            if cur_texts:
                add(cur_texts, cur_items, cur_sec)
                cur_texts, cur_items = [], []
            cur_sec = it["text"][:80]
        cur_texts.append(it["text"])
        cur_items.append(it)
    if cur_texts:
        add(cur_texts, cur_items, cur_sec)

    if strategy == "section_merge":  # 짧은 절을 인접 절과 합친다
        merged: list[dict] = []
        for c in chunks:
            if merged and c["kind"] == "body" and merged[-1]["kind"] == "body" and len(merged[-1]["text"]) < size:
                prev = merged[-1]
                prev["text"] += "\n" + c["text"]
                prev["pages"] = sorted(set(prev["pages"]) | set(c["pages"]))
                prev["boxes"] += c["boxes"]
            else:
                merged.append(dict(c))
        for i, c in enumerate(merged):
            c["id"] = f"c{i:04d}"
        return merged
    return chunks


@app.post("/api/chunk")
async def api_chunk(req: Request):
    p = await req.json()
    s = SESSIONS.get(p.get("sid"))
    if not s:
        raise HTTPException(404, "세션이 만료되었습니다. 논문을 다시 올려 주세요")
    chunks = build_chunks(s["items"], p.get("strategy", "section"), int(p.get("size", 1000)),
                          int(p.get("overlap", 100)), p.get("table_mode", "inline"),
                          set(p.get("heads") or []))
    s["chunks"] = chunks
    s.pop("emb", None)
    lens = [len(c["text"]) for c in chunks] or [0]
    warn = ""
    strategy = p.get("strategy", "section")
    if strategy.startswith("section") and len(chunks) < 3:
        warn = ("이 파서는 제목을 구분하지 않아 절 경계가 없습니다. 문서 전체가 청크 하나가 됐습니다. "
                "3단계에서 지면을 눌러 제목을 지정하거나, Docling으로 다시 파싱하거나, 고정 길이를 쓰세요.")
    elif max(lens) > 20000:
        warn = f"가장 긴 청크가 {max(lens):,}자입니다. 임베딩이 앞부분만 보게 되니 절을 더 나누는 편이 좋습니다."
    return {"chunks": chunks, "warn": warn,
            "stats": {"count": len(chunks), "mean": int(sum(lens) / len(lens)),
                      "p50": sorted(lens)[len(lens) // 2], "max": max(lens)}}


# ──────────────────────────────────────────── 임베딩·검색·답변
def openai_post(path: str, payload: dict, key: str) -> dict:
    import urllib.request
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{path}", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def embed(texts: list[str], model: str, key: str) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), 64):
        out = openai_post("embeddings", {"model": model, "input": [t[:8000] for t in texts[i:i + 64]]}, key)
        vecs += [d["embedding"] for d in out["data"]]
    M = np.array(vecs, dtype="float32")
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


@app.post("/api/ask")
async def api_ask(req: Request):
    p = await req.json()
    s = SESSIONS.get(p.get("sid"))
    if not s or not s.get("chunks"):
        raise HTTPException(404, "먼저 청킹을 실행해 주세요")
    key = (p.get("api_key") or OPENAI_KEY).strip()
    if not key:
        raise HTTPException(400, "OpenAI API 키가 없습니다")
    question = (p.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "질문을 입력해 주세요")
    model = p.get("embed_model", "text-embedding-3-small")
    chunks = s["chunks"]
    sig = hashlib.sha1((model + "|" + "|".join(c["id"] + str(len(c["text"])) for c in chunks)).encode()).hexdigest()
    if s.get("emb", {}).get("sig") != sig:
        s["emb"] = {"sig": sig, "M": embed([c["text"] for c in chunks], model, key)}
    M = s["emb"]["M"]
    qv = embed([question], model, key)[0]
    sims = M @ qv
    order = np.argsort(-sims)[: int(p.get("k", 6))]
    ranked = [{"rank": i + 1, "chunk_id": chunks[int(j)]["id"], "score": round(float(sims[j]), 3),
               "section": chunks[int(j)]["section"], "pages": chunks[int(j)]["pages"],
               "text": chunks[int(j)]["text"]} for i, j in enumerate(order)]

    answer = None
    if p.get("generate"):
        ctx = "\n\n".join(f"[{r['chunk_id']}] {r['text'][:3000]}" for r in ranked)
        out = openai_post("chat/completions", {
            "model": p.get("chat_model", "gpt-4.1-mini"), "temperature": 0,
            "messages": [
                {"role": "system", "content":
                 "너는 논문 근거만으로 답하는 조수다. 주어진 발췌 안에 답이 없으면 '근거 없음'이라고만 답한다. "
                 "답은 한국어로 두 문장 이내로 쓰고, 문장 끝에 근거 청크 식별자를 (c0007)처럼 붙인다. "
                 "발췌가 영어나 일본어나 중국어여도 답은 한국어로 쓴다."},
                {"role": "user", "content": f"발췌:\n{ctx}\n\n질문: {question}"}],
        }, key)
        answer = out["choices"][0]["message"]["content"].strip()
    return {"ranked": ranked, "answer": answer}


# ──────────────────────────────────────────── 골드 전사
@app.post("/api/gold")
async def api_gold(req: Request):
    p = await req.json()
    s = SESSIONS.get(p.get("sid"))
    span = (p.get("answer_span") or "").strip()
    question = (p.get("question") or "").strip()
    if not (span and question):
        raise HTTPException(400, "질문과 정답 발췌가 모두 필요합니다")
    entry = {
        "question_id": p.get("question_id") or f"u{uuid.uuid4().hex[:6]}",
        "doc_name": (s or {}).get("name", p.get("doc_name", "")),
        "parser": (s or {}).get("parser"), "question": question, "answer_span": span,
        "evidence_type": p.get("evidence_type", "body"), "page": p.get("page"),
        "verdict": p.get("verdict", "O"), "note": p.get("note", ""),
        "retrieval": p.get("retrieval"), "answer": p.get("answer"),
        "by": p.get("by", ""), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = GOLD_DIR / "gold.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    total = sum(1 for _ in open(path, encoding="utf-8"))
    return {"ok": True, "entry": entry, "total": total, "synced": push_gold(path)}


@app.get("/api/gold")
async def api_gold_list():
    path = GOLD_DIR / "gold.jsonl"
    if not path.exists():
        return {"entries": []}
    return {"entries": [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]}


@app.get("/api/config")
async def api_config():
    return {"password_required": bool(PASSWORD), "server_key": bool(OPENAI_KEY),
            "max_pages": MAX_PAGES, "docling": True,
            "gold_synced": bool(GOLD_DATASET and HF_TOKEN)}


@app.get("/")
async def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8831")))
