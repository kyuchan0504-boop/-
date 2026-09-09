"""
PEM 수전해 + 재생에너지 모델 — 대회 현장 시연용 웹앱

심사위원이 폰으로 QR을 스캔해 지역/종류를 고르고 "확인"을 누르면,
이 서버가 실제로 pem_model.py의 계산 함수들을 그 지역 데이터로 돌려서
결과 그래프 데이터를 즉석에서 만들어 돌려준다.

중요한 설계 포인트 (왜 이렇게 짰는지):
1. pem_model.py는 RE_REGION_NAME, J_MAX 같은 "모듈 전역변수"를 여러 함수가
   직접 참조하는 구조다. 그래서 요청 두 개가 동시에 들어와 지역이 다르면
   전역변수가 서로 덮어써져서 엉뚱한 결과가 섞일 위험이 있다.
   -> 계산 전체를 하나의 Lock으로 감싸서 항상 한 번에 한 요청만 계산하게 한다.
   -> 배포 시에도 gunicorn을 반드시 --workers 1 로 띄운다 (README 참고).
2. main()은 콘솔에 참고용 로그를 많이 찍는다 (문제 없음, 서버 로그에만 남음).
3. 그래프는 서버에서 이미지로 안 그리고, 숫자 데이터만 JSON으로 내려서
   프론트엔드(Chart.js)가 그린다 — 폰에서 더 빠르고 매끈하게 보인다.
"""
from __future__ import annotations

import os
import glob
import threading
import traceback
import unicodedata

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

import pem_model as m

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "지역별 태양광, 풍력")

# 계산은 항상 한 번에 하나씩만 — 전역변수 경합 방지 (위 설명 참고)
COMPUTE_LOCK = threading.Lock()

# 풍력 원본 데이터 점검 결과 (2026-09 확인):
#   - 부산·울산: 발전량이 전부 0 (데이터 자체가 깨져있음) -> 영구 제외
#   - 강원·경기·경남·경북·인천·전남·전북·충남: 파일명은 "2023_2025"이지만
#     실제로는 2023년 데이터만 있고(그마저 8,016시간=약 334일, 꽉 채운 1년 아님) 2024~2025년이 없음
#     -> 그중 경기/인천/충남은 이용률(CF)이 5~10%대로 육상풍력 평균(20~25%대)보다 비정상적으로 낮아 제외
#     -> 강원/경남/경북/전남/전북 5곳만 CF가 20~30%대로 정상 범위라 "2023년 데이터"라고 명시하고 사용
#   - 제주: 2023~2025년 다 있지만 2025년 구간만 이용률이 비정상적으로 낮음(3%대) -> 2025 기준으론 제외
# 태양광은 전부 2025년 기준으로 맞춰서 쓰므로, 풍력만 2023년을 쓴다는 걸 화면에 항상 명시한다 (index.html 참고).
WIND_OK_REGIONS = ["강원", "경남", "경북", "전남", "전북"]
WIND_YEAR = 2023
SOLAR_YEAR = 2025


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def scan_regions() -> dict:
    """data 폴더를 읽어서 {"태양광": [...지역들...], "풍력": [...]} 형태로 반환."""
    out: dict[str, list[str]] = {"태양광": [], "풍력": []}
    for kind in out:
        folder = os.path.join(DATA_ROOT, kind)
        if not os.path.isdir(folder):
            continue
        regions = []
        for p in glob.glob(os.path.join(folder, "*.xlsx")):
            base = _nfc(os.path.basename(p))
            region = base.split("_")[0]
            regions.append(region)
        out[kind] = sorted(set(regions))
    # 풍력은 데이터 품질이 확인된 5개 지역만 노출 (위 WIND_OK_REGIONS 설명 참고)
    out["풍력"] = sorted(set(out["풍력"]) & set(WIND_OK_REGIONS))
    return out


REGIONS = scan_regions()


def _monthly_sum(values: np.ndarray, months: np.ndarray) -> list[float]:
    s = pd.Series(np.asarray(values, dtype=float), index=months)
    grouped = s.groupby(level=0).sum()
    return [float(grouped.get(mo, 0.0)) for mo in range(1, 13)]


LCOH_ITEM_LABELS = {
    "annualized_capex": "설비비(연환산)",
    "fixed_OM": "고정 O&M",
    "stack_replacement": "스택 교체",
    "battery_replacement": "배터리 교체",
    "electricity": "전기요금",
    "curtailment_penalty": "출력제한 위약금",
    "variable_OM": "변동 O&M",
    "water": "용수",
}


def run_model(region: str, kind: str) -> dict:
    # 이번 요청에 맞게 모델의 전역 설정을 세팅
    m.RE_DATA_ROOT = DATA_ROOT
    m.RE_REGION_NAME = region
    m.RE_KIND = kind
    # 풍력은 정상 데이터가 2023년치뿐이라 그 해로 고정, 태양광은 2025년 기준 유지
    m.RE_YEAR = WIND_YEAR if kind == "풍력" else SOLAR_YEAR
    m.FIG_DIR = None
    m.SHOW_FIGURES = False
    m.PLOT_DETAIL_FIGURES = False
    m.PLOT_BATTERY_STRESS = False
    m.FAST_MODE = True

    # 지역이 바뀌었을 수 있으니 프로파일 캐시를 비워서 새로 로드하게 한다
    m._PROFILE_CACHE = None
    m._PROFILE_SIG = None

    out = m.main(make_plots=False, run_case1_sweep=False, run_hysteresis=False)

    profile = out["profile"]
    results = out["results"]
    lcohs = out["lcohs"]
    stack = out["stack"]
    months = profile.month

    cases = {}
    for name, r in results.items():
        lc = lcohs[name]
        items = {LCOH_ITEM_LABELS.get(k, k): round(float(v), 4)
                 for k, v in lc["items_usd_per_kg"].items()}
        ledger = r.ledger
        e_paid = max(r.E_paid, 1e-9)
        hrs = m._production_hours(r, getattr(r, "dt", 1.0) or 1.0)
        cases[name] = {
            "annual_h2_t": round(r.H2_total / 1000.0, 3),
            "lcoh": round(float(lc["LCOH"]), 3),
            "sec_eff": round(r.SEC_eff(), 2),
            "op_hours": round(r.op_hours, 0),
            "curtail_pct": round(r.E_curtail / e_paid * 100, 2),
            "ess_kwh": round(r.E_rated, 1),
            "ess_capex_usd": round(float(lc["capex_ess"]), 0),
            "lcoh_items": items,
            "ledger_pct": {
                "curtail": round(ledger["curtail"] / e_paid * 100, 2),
                "idle": round(ledger["idle"] / e_paid * 100, 2),
                "rte": round(ledger["rte"] / e_paid * 100, 2),
                "used": round((ledger["bop"] + ledger["stack"]) / e_paid * 100, 2),
            },
            "monthly_h2_kg": [round(v, 1) for v in _monthly_sum(r.H2_series, months)],
            "hour_ledger": {
                "producing": round(hrs["producing"], 0),
                "idle_with_gen": round(hrs["idle_with_gen"], 0),
                "no_gen": round(hrs["no_gen"], 0),
            },
        }

    best_lcoh = min(cases.items(), key=lambda kv: kv[1]["lcoh"])
    best_h2 = max(cases.items(), key=lambda kv: kv[1]["annual_h2_t"])

    return {
        "region": region,
        "kind": kind,
        "year": profile.year,
        "capacity_factor_pct": round(profile.capacity_factor * 100, 2),
        "peak_kw": round(profile.peak, 1),
        "mean_kw": round(profile.mean, 1),
        "e_paid_mwh": round(profile.E_paid / 1000.0, 1),
        "monthly_gen_mwh": [round(v / 1000.0, 1) for v in _monthly_sum(profile.P_in, months)],
        "stack": {
            "a_tot_cm2": round(stack.A_tot, 0),
            "n_cells": round(stack.N_cells, 1),
            "j_rated": stack.j_rated,
            "capex_stack_usd": round(stack.capex_stack, 0),
            "capex_bop_usd": round(stack.capex_bop, 0),
        },
        "cases": cases,
        "best_lcoh_case": best_lcoh[0],
        "best_h2_case": best_h2[0],
    }


@app.route("/")
def index():
    return render_template("index.html", regions=REGIONS)


@app.route("/api/regions")
def api_regions():
    return jsonify(REGIONS)


@app.route("/api/run")
def api_run():
    region = _nfc(request.args.get("region", ""))
    kind = _nfc(request.args.get("kind", "태양광"))
    if not region:
        return jsonify({"error": "region 파라미터가 필요합니다."}), 400
    if region not in REGIONS.get(kind, []):
        return jsonify({"error": f"'{kind}'에 '{region}' 데이터가 없습니다."}), 400

    with COMPUTE_LOCK:
        try:
            payload = run_model(region, kind)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return jsonify({"error": f"계산 중 오류: {exc}"}), 500

    return jsonify(payload)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
