# 재생에너지 x PEM 수전해 — 대회 시연용 웹앱

지역/에너지원을 고르면, 서버가 그 자리에서 실제로 `pem_model.py`를 돌려서
결과 그래프를 폰 화면에 보여주는 앱입니다.

## 지금 상태 (중요)

- **태양광 17개 지역 전부 정상 동작 확인함** (강원/경기/경남/경북/광주/대구/대전/부산/서울/세종/울산/인천/전남/전북/제주/충남/충북)
- **풍력은 아직 못 씀.** 원본 데이터를 열어보니:
  - 부산·울산: 발전량이 전부 0 (데이터 자체가 깨져 있음)
  - 나머지 9개 지역: 파일명은 "2023_2025"인데 실제로는 2023년 데이터만 있고 2024~2025년이 없음
  - 제주만 3개년 다 있는데, 2025년 구간 이용률이 비정상적으로 낮게 나옴(0.35%)
  - 그래서 지금 프론트엔드에서 풍력 버튼은 비활성화("준비중")해뒀어요. 코드 자체는 손 안 대도 되니, 나중에 정상적인 2024~2025 풍력 데이터로 파일만 교체하면 바로 켜집니다 (`data/지역별 태양광, 풍력/풍력/` 폴더 안 파일 이름/컬럼 구조를 태양광 파일과 동일하게 맞추면 됩니다).
- 지역 하나 계산에 로컬 테스트 기준 약 7~9초 걸려요 (Render 무료 플랜은 이보다 조금 더 걸릴 수 있어요).

## 로컬에서 먼저 테스트해보기

```bash
pip install -r requirements.txt
python3 app.py
# http://localhost:5000 접속
```

## Render에 배포하기 (무료, 대회 당일 이 세션과 무관하게 계속 살아있음)

1. 이 폴더 전체를 GitHub 저장소로 올린다 (public이든 private이든 상관없음).
   ```bash
   git init
   git add .
   git commit -m "PEM demo webapp"
   git branch -M main
   git remote add origin <본인 깃허브 저장소 URL>
   git push -u origin main
   ```
2. [render.com](https://render.com) 가입 (GitHub 계정으로 로그인 가능).
3. 대시보드에서 **New + → Web Service** 선택 → 방금 올린 저장소 연결.
4. 설정값:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:$PORT`
     (Procfile에 이미 적어뒀어서 Render가 자동으로 인식할 수도 있음)
   - **Instance Type**: Free
5. **Create Web Service** 누르면 몇 분 안에 빌드되고, `https://<이름>.onrender.com` 같은 주소가 생김.
6. 그 주소를 그대로 QR코드로 만들면 끝. (QR 생성: [qr-code-generator.com](https://www.qr-code-generator.com) 같은 무료 사이트에 주소만 붙여넣으면 바로 나와요.)

### 꼭 알아야 할 것 — 무료 플랜의 "잠자기"

Render 무료 플랜은 **15분간 요청이 없으면 서버가 잠들어요.** 잠든 상태에서 첫 요청이 오면
깨어나는 데 30~60초 정도 걸려요 — 심사위원이 QR 찍었는데 첫 화면이 한참 안 뜨면
당황할 수 있어요.

**대회 당일 발표 5~10분 전에 본인 폰으로 먼저 그 주소에 한 번 접속해서 "깨워두는 것"을
꼭 권장드려요.** 그 뒤로는 계속 요청이 이어지는 한 안 잠들어요.

## 왜 `--workers 1 --threads 1` 로 고정했는지

`pem_model.py`는 `RE_REGION_NAME`, `J_MAX` 같은 값들을 **모듈 전역변수**로 두고 여러 함수가
직접 읽는 구조예요. 만약 두 심사위원이 거의 동시에 서로 다른 지역을 눌러서 요청이 겹치면,
한쪽이 계산하는 도중에 다른 쪽이 전역변수를 바꿔버려서 결과가 섞일 수 있어요.
그래서 서버를 통째로 1개 프로세스·1개 스레드로만 돌리고(`app.py`의 `COMPUTE_LOCK`과 이중으로 방어),
요청이 몰려도 항상 한 번에 하나씩만 계산하도록 만들어뒀어요. 계산 자체가 7~9초라
여러 명이 몰리면 뒷사람은 그만큼 기다리게 되는데, 대회 시연 규모에서는 문제없을 거예요.

## 파일 구성

```
app.py                 Flask 서버 (지역 스캔, /api/run 계산·JSON 반환)
pem_model.py            원본 파이썬 모델 (수정 없이 그대로 사용)
templates/index.html    모바일용 프론트엔드 (지역 선택 + 확인 버튼 + 그래프 6종)
data/                   지역별 태양광·풍력 1MW 정규화 엑셀 데이터
requirements.txt        Python 패키지 목록
Procfile                배포 시 서버 실행 명령
```

## 화면에 보여주는 그래프 6종

1. 월별 재생에너지 발전량 (해당 지역, 전해조 배정분)
2. Case 1/2/3 연간 수소 생산량 비교
3. Case 1/2/3 수소 생산단가(LCOH) 비교
4. LCOH 항목별 구성(설비비/전기요금/O&M 등, Case별)
5. Case 1/2/3 월별 수소 생산량 추이
6. 전력 사용 구조 (실제 사용 vs 출력제한/정지/ESS손실로 버려진 비율)
