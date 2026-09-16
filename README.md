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


[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/oon-jung/sancho-paper-lab)

위 버튼을 누르면 Render 계정에 로그인한 뒤 render.yaml대로 무료 웹 서비스가 만들어집니다. 만들 때 `OPENAI_API_KEY`를 넣으세요.

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

## 2026-09-16 추가 기능
- **파서 3종 비교**: 2단계에서 PyMuPDF·Docling·MinerU(VLM)를 골라 같은 지면 위에 색으로 겹쳐 봅니다(파랑·초록·주황, 표는 이중선, 제목은 굵은 선). 비교표에서 "이 파서로"를 누르면 이후 단계에 그 결과를 씁니다.
- **한국어 번역 코퍼스**: 4단계에서 "한국어로 번역한 뒤"를 고르면 문단마다 한국어로 옮긴 뒤 청킹·검색합니다. 문단 단위라 지면 위치는 그대로 남습니다.
- **근거 표시·자동 채점**: 5단계 답변에 근거 문단이 지면 위 빨간 상자로 표시되고 원문·한국어 번역이 나란히 나옵니다. 정답 키워드를 넣으면 정답·부분·오답이 붙습니다.
- **임베딩 BGE-M3(로컬)** 선택 추가.

## 어디서 무엇이 되나
| 기능 | 로컬(맥북) | Render 무료(Dockerfile.light) |
|---|---|---|
| PyMuPDF 파싱·청킹·검색·답변·골드 | ○ | ○ |
| 한국어 번역 코퍼스·근거 표시·자동 채점 | ○ | ○ (OpenAI 키 필요) |
| Docling | ○ | ✗ (메모리 2GB) |
| MinerU(VLM) | ○ (vlm/.venv의 mineru) | ✗ |
| BGE-M3 임베딩 | ○ | ✗ (OpenAI 임베딩 사용) |

로컬 실행: 파일럿 venv로 `OPENAI_API_KEY=... python app.py` (8831). MinerU 경로는 `MINERU_BIN` 환경변수로 바꿀 수 있습니다.
Render 배포: 이 저장소를 Render에 Blueprint로 연결하면 render.yaml대로 올라갑니다. 환경변수 `OPENAI_API_KEY`(필수), `LAB_PASSWORD`(선택)를 Render 대시보드에서 넣으세요.
