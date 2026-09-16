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
# Docling은 실행 중 2GB 가까이 쓴다. 512MB 무료 호스팅에서는 꺼 둔다.
ENABLE_DOCLING = os.environ.get("ENABLE_DOCLING", "1") != "0"
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
            if not text and html:
                # docling 표는 text가 비어 있다. 행은 줄바꿈, 셀은 ' | '로 펴서 검색·번역에 쓴다.
                rows = re.findall(r"<tr.*?</tr>", html, flags=re.S)
                text = "\n".join(" | ".join(re.sub(r"<.*?>", "", c).strip() for c in re.findall(r"<t[dh].*?</t[dh]>", r, flags=re.S)) for r in rows)
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
    if parser == "docling" and not ENABLE_DOCLING:
        raise HTTPException(400, "이 서버에서는 Docling을 끄고 운영합니다. PyMuPDF로 파싱한 뒤 3단계에서 제목을 직접 지정하세요")
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
                     "name": file.filename, "at": time.time(),
                     "pdf": data, "sizes": sizes,  # 파서 비교(/api/compare)에서 다른 파서를 돌리기 위해 메모리에만 둔다
                     "compare": {parser: {"items": items, "elapsed": elapsed}}}
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


# ──────────────────────────────────────────── 파서 3종 비교
MINERU_BIN = os.environ.get("MINERU_BIN", "/Users/woon/sancho-local-secure/pdf-parser-pilot/vlm/.venv/bin/mineru")
MINERU_KIND = {"title": "section_header", "text": "text", "table": "table", "image": "picture", "figure": "picture",
               "header": "page_header", "footer": "page_footer", "page_number": "page_footer", "page_footnote": "footnote",
               "image_caption": "caption", "table_caption": "caption", "figure_caption": "caption",
               "image_footnote": "footnote", "table_footnote": "footnote", "list": "list_item", "equation": "formula",
               "interline_equation": "formula", "code": "code", "ref_text": "text", "aside_text": "text",
               "chart": "picture", "image_block": "picture", "image_body": "picture", "table_body": "table"}


def parse_mineru(pdf_bytes: bytes, name: str) -> list[dict]:
    """MinerU(VLM)를 명령행으로 돌리고 model.json(쪽마다 항목·정규화 bbox)을 읽는다. 쪽당 3초쯤, 시작 비용 20초."""
    import subprocess, tempfile, shutil
    if not Path(MINERU_BIN).exists():
        raise HTTPException(400, "이 서버에는 MinerU가 없습니다")
    work = Path(tempfile.mkdtemp(prefix="mineru_"))
    try:
        src = work / "doc.pdf"
        src.write_bytes(pdf_bytes)
        subprocess.run([MINERU_BIN, "-p", str(src), "-o", str(work / "out"), "-b", "vlm-engine"],
                       check=True, capture_output=True, timeout=1800)
        models = list((work / "out").rglob("*_model.json"))
        if not models:
            raise HTTPException(500, "MinerU가 model.json을 내지 않았습니다")
        pages = json.loads(models[0].read_text(encoding="utf-8"))
        items = []
        for pno, blocks in enumerate(pages, start=1):
            for i, b in enumerate(blocks):
                kind = MINERU_KIND.get(b.get("type", "text"), b.get("type", "text"))
                bb = b.get("bbox") or [0, 0, 0, 0]
                content = b.get("content")
                text = content if isinstance(content, str) else ""
                html = ""
                if kind == "table" and isinstance(content, str) and "<table" in content:
                    html = content
                    rows = re.findall(r"<tr.*?</tr>", content, flags=re.S)
                    text = "\n".join(" | ".join(re.sub(r"<.*?>", "", c).strip() for c in re.findall(r"<t[dh].*?</t[dh]>", r, flags=re.S)) for r in rows)
                items.append({"id": f"mu{pno}_{i}", "kind": kind, "page": pno,
                              "bbox": [round(float(v), 4) for v in bb[:4]], "text": text.strip(), "html": html, "level": None})
        return items
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"MinerU 실행 실패: {(exc.stderr or b'')[-400:].decode(errors='ignore')}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parser_stats(items: list[dict], elapsed: float, pages: int) -> dict:
    kinds: dict[str, int] = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    body = [it for it in items if it["kind"] not in {"page_header", "page_footer", "picture"}]
    hangul = re.findall(r"[가-힣]+", " ".join(it["text"] for it in body))
    per_page: dict[int, int] = {}
    for it in items:
        per_page[it["page"]] = per_page.get(it["page"], 0) + 1
    return {"elapsed": round(elapsed, 2), "per_page": round(elapsed / max(pages, 1), 2), "items": len(items),
            "headings": kinds.get("section_header", 0) + kinds.get("title", 0), "tables": kinds.get("table", 0),
            "pictures": kinds.get("picture", 0), "captions": kinds.get("caption", 0),
            "chars": sum(len(it["text"]) for it in body),
            "broken_spacing": round(sum(1 for t in hangul if len(t) == 1) / max(len(hangul), 1) * 100, 1),
            "kinds": kinds, "per_page_counts": per_page}


@app.post("/api/compare")
def api_compare(p: dict):
    """올린 논문 한 편을 PyMuPDF·Docling·MinerU로 각각 파싱해 같은 지면 위에 겹쳐 볼 수 있게 돌려준다."""
    s = SESSIONS.get(p.get("sid"))
    if not s or not s.get("pdf"):
        raise HTTPException(404, "세션이 만료되었습니다. 논문을 다시 올려 주세요")
    want = [x for x in (p.get("parsers") or ["pymupdf", "docling"]) if x in {"pymupdf", "docling", "mineru"}]
    out = {}
    for name in want:
        if name not in s["compare"]:
            started = time.perf_counter()
            if name == "pymupdf":
                items = parse_pymupdf(pymupdf.open(stream=s["pdf"], filetype="pdf"))
            elif name == "docling":
                if not ENABLE_DOCLING:
                    raise HTTPException(400, "이 서버에서는 Docling을 끄고 운영합니다")
                items = parse_docling(s["pdf"], s["sizes"])
            else:
                items = parse_mineru(s["pdf"], s["name"])
            s["compare"][name] = {"items": items, "elapsed": round(time.perf_counter() - started, 2)}
        c = s["compare"][name]
        out[name] = {"items": c["items"], "stats": parser_stats(c["items"], c["elapsed"], s["pages"])}
    return {"parsers": out, "current": s["parser"]}


@app.post("/api/use_parser")
async def api_use_parser(req: Request):
    """비교 결과 중 하나를 이후 단계(구조·청킹·질문)의 파싱 결과로 채택한다."""
    p = await req.json()
    s = SESSIONS.get(p.get("sid"))
    name = p.get("parser")
    if not s or name not in (s or {}).get("compare", {}):
        raise HTTPException(404, "먼저 파서 비교를 실행해 주세요")
    s["items"] = s["compare"][name]["items"]
    s["parser"] = name
    for k in ("items_ko", "items_ko_target", "chunks", "emb"):
        s.pop(k, None)
    return {"ok": True, "parser": name, "items": s["items"]}


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
                       # 답변의 근거 문장을 지면 위 문단으로 되찾기 위해 항목 텍스트를 같이 둔다
                       "items": [{"id": i["id"], "page": i["page"], "bbox": i["bbox"], "text": i["text"],
                                  "orig_text": i.get("orig_text", i["text"])} for i in its],
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
                prev["items"] += c["items"]
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
    corpus = p.get("corpus", "orig")
    if corpus == "ko" and not s.get("items_ko"):
        raise HTTPException(400, "먼저 한국어 번역을 실행해 주세요")
    items = s["items_ko"] if corpus == "ko" else s["items"]
    chunks = build_chunks(items, p.get("strategy", "section"), int(p.get("size", 1000)),
                          int(p.get("overlap", 100)), p.get("table_mode", "inline"),
                          set(p.get("heads") or []))
    s["chunks"] = chunks
    s["corpus"] = corpus
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
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def chat_json(messages: list[dict], key: str, model: str = "gpt-4.1-mini") -> dict:
    out = openai_post("chat/completions", {"model": model, "temperature": 0,
                                           "response_format": {"type": "json_object"},
                                           "messages": messages}, key)
    return json.loads(out["choices"][0]["message"]["content"])


LANG_NAME = {"ko": "Korean", "en": "English"}


def guess_lang(text: str) -> str:
    letters = re.findall(r"[A-Za-z가-힣一-鿿ぁ-ヿ]", text)
    if not letters:
        return "en"
    ko = sum(1 for c in letters if "가" <= c <= "힣") / len(letters)
    cjk = sum(1 for c in letters if "一" <= c <= "鿿" or "ぁ" <= c <= "ヿ") / len(letters)
    return "ko" if ko > 0.3 else ("zh" if cjk > 0.3 else "en")


# ──────────────────────────────────────────── 번역 (항목 단위 → 좌표 보존)
@app.post("/api/translate")
async def api_translate(req: Request):
    """파싱 항목(문단·표)을 하나씩 한국어로 옮긴다. 항목 번호와 좌표는 그대로라 번역 코퍼스에서도 원문 위치를 되찾는다."""
    p = await req.json()
    s = SESSIONS.get(p.get("sid"))
    if not s:
        raise HTTPException(404, "세션이 만료되었습니다. 논문을 다시 올려 주세요")
    key = (p.get("api_key") or OPENAI_KEY).strip()
    if not key:
        raise HTTPException(400, "OpenAI API 키가 없습니다")
    target = p.get("target", "ko")
    if s.get("items_ko") and s.get("items_ko_target") == target:
        return {"ok": True, "cached": True, "items": len(s["items_ko"])}
    items = s["items"]
    src = guess_lang(" ".join(it["text"] for it in items[:60]))
    if src == target:
        s["items_ko"] = [dict(it, orig_text=it["text"]) for it in items]
        s["items_ko_target"] = target
        return {"ok": True, "same_language": True, "items": len(items)}
    batches, cur, n = [], [], 0
    for k, it in enumerate(items):
        if not it["text"].strip() or it["kind"] in {"picture", "page_header", "page_footer"}:
            continue
        if cur and n + len(it["text"]) > 6000:
            batches.append(cur); cur, n = [], 0
        cur.append(k); n += len(it["text"])
    if cur:
        batches.append(cur)
    sysmsg = (f"You translate fragments of a scientific paper into {LANG_NAME.get(target, target)}. Translate every entry faithfully. "
              "Keep numbers, units, statistics (mean±SD), chemical names, gene/protein names, abbreviations, citations and reference entries unchanged in form. "
              "For kind='table' the text is rows separated by newlines and cells by ' | ': translate the words in the cells but keep the exact same number of rows and cells and the separators. "
              'Return JSON {"items":[{"id":..., "text":...}, ...]} with exactly one output per input, same ids.')
    started = time.perf_counter()
    got: dict[int, str] = {}

    def run(idx):
        entries = [{"id": k, "kind": items[k]["kind"], "text": items[k]["text"]} for k in idx]
        out = chat_json([{"role": "system", "content": sysmsg},
                         {"role": "user", "content": json.dumps({"items": entries}, ensure_ascii=False)}], key)
        return {int(x["id"]): x["text"] for x in out.get("items", []) if "id" in x}

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(4) as ex:
        for part in ex.map(run, batches):
            got.update(part)
    s["items_ko"] = [dict(it, text=got.get(k, it["text"]), orig_text=it["text"]) for k, it in enumerate(items)]
    s["items_ko_target"] = target
    s.pop("chunks", None); s.pop("emb", None)
    return {"ok": True, "items": len(items), "translated": len(got), "calls": len(batches),
            "elapsed": round(time.perf_counter() - started, 1), "source_lang": src}


_BGE = None


def embed(texts: list[str], model: str, key: str) -> np.ndarray:
    vecs: list[list[float]] = []
    if model == "bge-m3":
        # 로컬 BGE-M3. 이전 실측에서 recall@6 35 vs OpenAI 3-small 26 — 다국어 논문에 유리하다.
        global _BGE
        if _BGE is None:
            try:
                from pdf_parser_pilot.chunkers import LocalBgeM3Embedder
            except Exception as exc:
                raise HTTPException(400, f"이 서버에는 BGE-M3가 없습니다: {exc}")
            _BGE = LocalBgeM3Embedder()
        for i in range(0, len(texts), 16):
            vecs += _BGE.embed(texts[i:i + 16])
    else:
        for i in range(0, len(texts), 64):
            out = openai_post("embeddings", {"model": model, "input": [t[:8000] for t in texts[i:i + 64]]}, key)
            vecs += [d["embedding"] for d in out["data"]]
    M = np.array(vecs, dtype="float32")
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _norm_loose(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s\"'`‘-‟‐-―−-]", "", s)


def locate_item(chunk: dict, quote: str) -> dict | None:
    """인용문이 들어 있는 문단을 찾는다. 없으면 단어가 가장 많이 겹치는 문단."""
    nq = _norm_loose(quote)
    if not nq:
        return None
    for it in chunk.get("items", []):
        if nq in _norm_loose(it["text"]):
            return it
    words = set(re.findall(r"\w{2,}", quote.lower()))
    best, score = None, 0
    for it in chunk.get("items", []):
        n = len(words & set(re.findall(r"\w{2,}", it["text"].lower())))
        if n > score:
            best, score = it, n
    return best


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

    answer, evidence, score = None, None, None
    if p.get("generate"):
        # 청크가 크면 앞부분만 넣다가 정답을 놓친다. 전체 예산을 나눠 쓰고, 남는 몫은 되돌린다.
        budget = int(p.get("context_chars", 40000))
        share = max(budget // max(len(ranked), 1), 1000)
        spare = sum(max(share - len(r["text"]), 0) for r in ranked)
        big = [r for r in ranked if len(r["text"]) > share]
        bonus = spare // max(len(big), 1) if big else 0
        parts = []
        for r in ranked:
            cap = share + (bonus if len(r["text"]) > share else 0)
            body = r["text"] if len(r["text"]) <= cap else r["text"][:cap] + " …(이하 생략)"
            parts.append(f"[{r['chunk_id']}] {body}")
        ctx = "\n\n".join(parts)
        out = chat_json([
            {"role": "system", "content":
             "너는 논문 근거만으로 답하는 조수다. 주어진 발췌 안에 답이 있으면 한국어로 두 문장 이내로 답하라. "
             "발췌가 영어나 일본어나 중국어여도 답은 한국어로 쓴다. "
             "그림·그래프에서만 읽을 수 있는 값이라 발췌에 수치가 없으면, 또는 발췌 어디에도 답이 없으면 answer를 정확히 \"근거 없음\"으로 하라. 추측하지 말라. "
             'JSON으로만 답하라: {"answer": "...", "evidence_chunk": "c0007 같은 청크 식별자 또는 null", '
             '"evidence_quote": "그 청크에서 그대로 복사한 한 문장(원문 그대로, 번역·요약 금지)"}'},
            {"role": "user", "content": f"발췌:\n{ctx}\n\n질문: {question}"}], key, p.get("chat_model", "gpt-4.1-mini"))
        answer = (out.get("answer") or "").strip()
        cid = out.get("evidence_chunk")
        ch = next((c for c in chunks if c["id"] == cid), None)
        it = locate_item(ch, out.get("evidence_quote") or "") if ch else None
        if ch and it:
            orig = it.get("orig_text", it["text"])
            if s.get("corpus") == "ko":
                translated = it["text"]  # 번역 코퍼스: 청크 문단이 곧 한국어 번역
            elif guess_lang(orig) == "ko":
                translated = orig
            else:
                translated = chat_json([{"role": "system", "content": "다음 논문 문단을 한국어로 번역하라. 숫자와 단위는 그대로. JSON {\"text\": ...}로만 답하라."},
                                        {"role": "user", "content": orig[:1500]}], key).get("text", "")
            evidence = {"chunk_id": cid, "item_id": it["id"], "page": it["page"], "bbox": it["bbox"],
                        "quote": out.get("evidence_quote"), "orig_text": orig, "translated": translated}
        # 자동 채점: 정답 키워드가 모두 있으면 정답, 일부면 부분, 없으면 오답. "근거 없음"은 키워드 자리에 그 말을 넣는다.
        kws = [k.strip() for k in (p.get("expected") or "").split(",") if k.strip()]
        if kws:
            na = unicodedata.normalize("NFKC", answer)
            hits = [bool(re.search(k, na, flags=re.I)) for k in kws]
            score = {"keywords": kws, "hits": hits,
                     "verdict": "정답" if all(hits) else ("부분" if any(hits) else "오답")}
    return {"ranked": ranked, "answer": answer, "evidence": evidence, "score": score, "corpus": s.get("corpus", "orig")}


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
            "max_pages": MAX_PAGES, "docling": ENABLE_DOCLING, "mineru": Path(MINERU_BIN).exists(),
            "gold_synced": bool(GOLD_DATASET and HF_TOKEN)}


@app.get("/")
async def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8831")))
