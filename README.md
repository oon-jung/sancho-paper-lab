---
title: 산초 논문 랩
emoji: 📄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 산초 논문 랩

논문 한 편을 올려 **파싱 → 구조 지정 → 청킹 → 검색 → 답변 → 골드 전사**까지 한 흐름으로 확인하는 도구입니다.

- 파서: PyMuPDF(빠름, 구조 없음) · Docling(제목·문단·표 구조, 쪽당 1~5초)
- 임베딩·답변: OpenAI
- 올린 PDF는 서버 메모리에만 두고 디스크에 남기지 않습니다. 골드 기록만 저장됩니다.

## 환경변수
| 이름 | 뜻 |
|---|---|
| `LAB_PASSWORD` | 공유 비밀번호. 비우면 누구나 들어옵니다 |
| `OPENAI_API_KEY` | 서버 키. 없으면 사용자가 화면에서 자기 키를 넣습니다 |
| `MAX_PAGES` | 처리할 최대 쪽수 (기본 40) |
| `GOLD_DIR` | 골드 저장 경로 (기본 `gold_store/`) |

## 로컬 실행
```
pip install -r requirements.txt
python app.py            # http://127.0.0.1:8831
```

## 배포 구성 두 가지

| | 전체 | 경량 |
|---|---|---|
| Dockerfile | `Dockerfile` | `Dockerfile.light` |
| 파서 | PyMuPDF + Docling | PyMuPDF만 (`ENABLE_DOCLING=0`) |
| 메모리 | 2GB 이상 필요 (실측 2,057MB) | 512MB로 충분 (실측 223MB) |
| 쓸 수 있는 곳 | Cloud Run 2GB · Oracle Always Free · HF PRO | Render 무료 · Koyeb 무료 |

경량에서도 3단계에서 지면을 눌러 제목을 직접 지정하면 절 단위 청킹이 그대로 됩니다.
