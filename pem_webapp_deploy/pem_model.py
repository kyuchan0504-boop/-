# %%

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:                                    # pragma: no cover
    matplotlib = None
    plt = None

# %%
# ###################################################################################
# ###################################################################################
##
##   PART 1 — CONFIG
##
# ###################################################################################
# ###################################################################################
FIG_DIR: str | None = "figures"      # None이면 저장 안 함
FIG_DPI = 130
SHOW_FIGURES = True

PLOT_DETAIL_FIGURES = False          # plot_case1 / plot_case2 / plot_case3
PLOT_BATTERY_STRESS = False          # 배터리 혹사도 3분할 그림
FAST_MODE = True                     # 탐색 격자를 성기게 → 전체 1회전 약 15초
RANDOM_SEED = 20260101

# ── 1.2 촉매 (PART 3) ────────────────────────────────────────────────────────

CATALYST_NAME = "IrO2"                                     
CATALYST_PARAM_FILE: str | None = "NMO_tafel_parameters.csv"     
TAFEL_N_ELECTRON = 2.0                                           # [OK] 검증됨
TAFEL_T_REF_K = 353.15                                           # [OK] = T_OPER
EXTRAPOLATION_WARN_DECADES = 1.0            # 피팅 상한 대비 이만큼 넘게 외삽하면 경고

T_OPER = 353.15          # [OK] 운전 온도 [K] (80 C)
P_AN = 1.0               # [OK] 양극 압력 [bar]
P_CAT = 30.0             # [OK] 음극 압력 [bar]
J0_CA = 0.2              # [OK] 음극 교환전류밀도 [A/cm2]
ALPHA_CA = 0.5           # [OK] 음극 전달계수
MEMB_T_UM = 180.0        # [OK] 막 두께 [um] (~Nafion 117)
MEMB_LAMBDA = 22.0       # [OK] 막 함수율 lambda
SIGMA_SCALE = 1.0        # [OK] Springer sigma 배율
J_LIM = 6.1              # [OK] 물질전달 한계 전류밀도 [A/cm2]
BOP_EFFICIENCY = 4.2     # [OK] BOP 소비 [kWh/kg-H2]

STACK_SIZING_MODE = "nameplate_kW"                               # [OK] 상용 1 MW급 고정
STACK_NAMEPLATE_KW = 1000.0                                      # [OK] PEM 명판 [kW]
STACK_PV_FRACTION = 0.70                                         # [TMP]
STACK_N_CELLS = 200.0                                            # [TMP]
A_CELL_CM2 = 1000.0                                              # [OK]
N_CELL_PER_STACK = 150.0                                         # [OK]
INTEGER_STACKS = False                                           # [OK] True면 1 MW가 안 맞음

J_MAX = 2.0              # [TODO] 정격 전류밀도 [A/cm2]. 상용 PEM 2~3. 설계 선택
J_X_mAcm2 = 2.0          # [TODO] 크로스오버 등가 전류밀도 [mA/cm2]
J_MIN_SAFETY = 2.0       # [TMP] 안전계수. 1.0 -> j_min=50*j_x, 2.0 -> 100*j_x
J_MIN = 50.0 * J_MIN_SAFETY * (J_X_mAcm2 / 1000.0)
FARADAY_MODE = "constant"        # [TMP] "constant"(=1.0) | "crossover"(=1-j_x/j)

J_STAR_MODE = "optimize"                       # "fixed" | "optimize"
J_STAR_FIXED = 1.80                         # 공통 고정 전류밀도 [A/cm2]
J_STAR_FIXED_CASE2: float | None = None     # None이면 J_STAR_FIXED 사용
J_STAR_FIXED_CASE3: float | None = None     # None이면 J_STAR_FIXED 사용

RE_SOURCE = "region"

# [set]
RE_DATA_ROOT = r"./지역별 발전소 데이터"
RE_REGION_NAME = "울산"                   # 강원 경기 경남 경북 광주 대구 대전 부산 서울
RE_KIND = "태양광"                        # "태양광" | "풍력"
RE_CAPACITY_MW = 3                    # ★ 발전량 정규화 규모 [MW]
RE_VALUE_COL = "1MW_정규화_발전량(kWh)"    # 원본 '발전량(kWh)' 컬럼과 헷갈리지 않도록 명시
RE_DATETIME_COL = "datetime"

RE_FILE_PATH: str | None = None          # 8760행 (timestamp, 발전량[kW])
RE_TIME_COL: str | None = None           # None이면 자동 탐지
RE_POWER_COL: str | None = None

RE_YEAR = 2025                           # 2023·2025 = 8760 h / 2024 = 8784 h (윤년)
RE_REGION = "(unset)"                    
RE_SOURCE_URL = ""                       # [TODO] 출처 URL
ALLOCATION_N = 1.00                      # [TODO] 배정 비율 n (0~1)
DT_HOURS = 1.0

ELEC_PRICE = 0.089       # [set] 전기 단가 [$/kWh]. 잉여 재생에너지 활용 전제.
WATER_PRICE = 0.0018     # [OK]  [$/kg]
LIFETIME_YR = 15         # [OK]
INTEREST_RATE = 0.08     # [OK]
FIXED_OM_FRAC = 0.05     # [OK]  총 CAPEX 대비 고정 O&M
VARIABLE_OM = 0.024      # [OK]  [$/kg-H2]
STACK_LIFETIME_H = 40000.0   # [OK]
REPL_STACK_FRAC = 1.0        # [OK]
MANUFACTURING_RATE = 100.0   # [OK] 생산규모 [MW/yr] (BOP 단가 보간용)
MARKUP = 0.5                 # [OK] 제조원가 대비 마크업
OVHD_FRAC = 0.5              # [OK] 스택 오버헤드 비율
FIXED_OM_INCLUDES_ESS = True                                     # [TMP]
ELEC_ACCOUNTING = "allocated"                                    # [TMP] 팀 확정 필요

ELEC_PENALTY_MULT = 2.5      # [set] 위약금 배율 (무차원)

ETA_RTE = 0.9                # [TMP] 왕복효율. 확정 범위 0.85~0.90
C_E_USD_KWH = 310           # [TODO] ESS 에너지 단가 [$/kWh]
C_P_USD_KW = 310           # [TODO] ESS 전력 단가 [$/kW]
ANNUAL_BATT_REPL_USD = 0.0    # [TODO] 배터리 교체 충당 훅 (조사 중)
CHARGE_RATING_POLICY = "p95"  # [TMP] "raw" | "p95" | "p99" | ...
CHARGE_RATING_PCTL = 95.0     # "raw" 계열이 아닐 때만 사용

def _use_charge_pctl() -> bool:
    return str(CHARGE_RATING_POLICY).strip().lower() not in ("raw", "peak", "max", "none")

ESS_SIZING_MODE = "surplus_q"      # "hours" | "surplus_q"
_ESS_AUTO_EXTEND_Q = True
SURPLUS_ETA_CHG = 1.0             # None -> sqrt(ETA_RTE) (편도 충전효율)
SURPLUS_DAY_HOURS = 24.0           # 적분 리셋 주기 [h]. PV 전용 가정 (§1.11 주석 참조)

CASE1 = dict(
    rating_sweep_factors=(0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0),   # 명판 kW 배수
)
CASE2 = dict(
    soc_max=0.95,             # [OK]
    soc_floor=0.05,           # [OK] Case 2는 정지선 = floor
    soc_restart=0.05,         # [TMP] sweep 대상
    restart_hours_of_Pstar=None,   # 절대량 정의(C2-7). 예: 1.5 -> P_star 1.5시간분
    cap_grid_hours=(0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48),
    cap_grid_hours_fast=(0.5, 1, 2, 4, 8, 16, 32),
    n_j_grid=40, n_j_grid_fast=16,
    restart_sweep=(0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
)
CASE3 = dict(
    soc_max=0.95,             # [OK] C3-3
    soc_floor=0.05,           # [OK] 안전 floor
    soc_stop=0.15,            # [OK] 실질 정지선
    soc_restart=0.25,         # [OK] 재기동선
    restart_hours_of_Pstar=None,
    direct_restart=True,
    cap_grid_hours=(0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24),
    cap_grid_hours_fast=(0.5, 1, 2, 4, 8, 16),
    q_grid=(2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100),
    q_grid_fast=(2, 5, 10, 20, 30, 40, 60, 80, 100),
    n_j_grid=40, n_j_grid_fast=20,
)

HEADLINE_J_MODE = "own"             # "own" | "manual"
J_STAR_MANUAL: float | None = None  # "manual" 이면 반드시 숫자 [A/cm2]
BALANCE_TOL = 0.01                 # 손실 원장 수지 허용오차 (기획서 P3)
INVERSION_TOL = 1.0e-6             # P<->j 왕복 상대오차 (기획서 P1)
INVERSION_GRID_N = 4000
WARM_START = True                  # C2-6 : 연 2회 실행, 2회차만 사용

def _n_j_grid(case: str) -> int:
    d = {"case2": CASE2, "case3": CASE3}[case]
    return d["n_j_grid_fast"] if FAST_MODE else d["n_j_grid"]

def _cap_grid(case: str):
    d = {"case2": CASE2, "case3": CASE3}[case]
    return d["cap_grid_hours_fast"] if FAST_MODE else d["cap_grid_hours"]

def _q_grid(case: str):
    d = {"case2": CASE2, "case3": CASE3}[case]
    k = "q_grid_fast" if FAST_MODE else "q_grid"
    return d.get(k, CASE3[k])          # CASE2는 아직 q 격자를 안 쓴다 -> CASE3 것 차용

def j_is_fixed() -> bool:
    return str(J_STAR_MODE).lower() == "fixed"

def fixed_j(case: str, verbose: bool = True) -> float:
    per = {"case2": J_STAR_FIXED_CASE2, "case3": J_STAR_FIXED_CASE3}[case]
    raw = per if per is not None else J_STAR_FIXED
    try:
        j = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{case}: 지정 전류밀도 값이 숫자가 아닙니다 -> {raw!r}")
    if not np.isfinite(j) or j <= 0:
        raise ValueError(f"{case}: 지정 전류밀도가 유효하지 않습니다 -> {j}")
    if j < J_MIN or j > J_MAX:
        j_c = float(min(max(j, J_MIN), J_MAX))
        if verbose:
            print(f"  [경고] {case} 지정 j = {j:.4f} A/cm2 가 운전 창 "
                  f"[{J_MIN:.4f}, {J_MAX:.4f}] 밖입니다 -> {j_c:.4f} 로 클램프합니다.")
        j = j_c
    return j

def _ensure_fig_dir() -> str | None:
    if FIG_DIR is None:
        return None
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    return FIG_DIR

_SURPLUS_WARNED: set[str] = set()

def daily_storage_need(P_in, P_star, P_rc, dt=1.0, eta_c=None,
                       threshold_is_Pstar=True, day_hours=None) -> np.ndarray:
    P = np.asarray(P_in, dtype=float)
    if eta_c is None:
        eta_c = float(SURPLUS_ETA_CHG)
    day_hours = SURPLUS_DAY_HOURS if day_hours is None else day_hours

    spd = int(round(day_hours / dt))                    # steps per day
    if spd < 1:
        raise ValueError(f"day_hours({day_hours}) / dt({dt}) 가 1스텝 미만입니다.")
    n_days, rem = divmod(len(P), spd)
    if n_days < 1:
        raise ValueError(f"프로파일 길이 {len(P)} 가 하루({spd}스텝)보다 짧습니다.")
    if rem and "trunc" not in _SURPLUS_WARNED:
        _SURPLUS_WARNED.add("trunc")
        print(f"  [WARN] profile length {len(P)} not divisible by day step {spd}; "
              f"truncate last {rem} step(s) for sizing")
    P = P[:n_days * spd]

    sur = np.maximum(P - P_star, 0.0) if threshold_is_Pstar else P.copy()
    sur = np.minimum(sur, max(float(P_rc), 0.0))        # 충전정격 클리핑
    return eta_c * sur.reshape(n_days, spd).sum(axis=1) * dt

def size_ess_from_surplus(P_in, P_star, P_rc, q, dt=1.0, case="case3",
                          soc_max=None, soc_stop=None, return_need=False):
    d = {"case2": CASE2, "case3": CASE3}[case]
    soc_max = d["soc_max"] if soc_max is None else soc_max
    if soc_stop is None:
        soc_stop = d.get("soc_stop", d.get("soc_floor", 0.0))
    usable = float(soc_max) - float(soc_stop)
    if usable <= 0:
        raise ValueError(f"{case}: SOC 사용 창이 0 이하입니다 "
                         f"(max {soc_max} - stop {soc_stop}).")

    e_draw = daily_storage_need(P_in, P_star, P_rc, dt=dt,
                              threshold_is_Pstar=(case == "case3"))
    E_rated = float(np.percentile(e_draw, q)) / usable
    return (E_rated, e_draw) if return_need else E_rated


# ###################################################################################
# ###################################################################################
##
##      PART 2 — 상수 & 원가 데이터
##
# ###################################################################################
# ###################################################################################

R_CONST = 8.314462618        # 기체상수 [J/mol/K]
F_CONST = 96485.33212        # 패러데이 상수 [C/mol]
MW_H2 = 2.01588e-3           # 수소 분자량 [kg/mol]
MW_H2O = 18.01528e-3         # 물 분자량 [kg/mol]
N_ELECTRON_H2 = 2.0          # 수소 1몰당 전자수
EUR_TO_USD = 1.17
WATER_STOICH = MW_H2O / MW_H2    # 화학량론 물 소요 [kg/kg-H2]
E_OER_EQ = 1.23                  # OER 평형전위 [V vs RHE]

METAL_PRICE = {
    "Mn": 0.0020 * EUR_TO_USD,     # earth-abundant
    "Nd": 0.101,                   # Nd2O3, (출처: SMM 2025-12)
    "Pt": 33487.0 / 1000.0,
    "Au": 49.000 * EUR_TO_USD,
    "Ir": 122906.0/1000.0,         
}

MATERIAL_PRICE = {
    "membrane_180um": 766.0 * EUR_TO_USD,   # Nafion 180 um [USD/m2]
    "membrane_80um":  371.0 * EUR_TO_USD,   # Nafion 80 um [USD/m2]
    "carbon_cloth":   168.0 * EUR_TO_USD,   # 음극 bulk PTL [USD/m2]
    "PPS40GF":        15.4 * EUR_TO_USD,    # 프레임/가스켓 [USD/kg]
    "ti_powder":      0.400 * EUR_TO_USD,   # 양극 bulk PTL [USD/g]
    "ss316":          0.0013 * EUR_TO_USD,  # BPP [USD/g]
    "ptfe":           8.0 * EUR_TO_USD,     # 프레임/가스켓 [USD/kg]
    "a356_al":        82.0 * EUR_TO_USD,    # 엔드플레이트 [USD/m2]
}

DENSITY = {"Ti": 4.50, "SS316": 8.00, "Au": 19.32, "Pt": 21.45}

CA_LOAD_mgcm2 = 0.2      # 음극 Pt 로딩
CA_CAT = "Pt"
PTL_AN_MAT = "Ti"        # 양극 PTL 소재
PTL_AN_T_mm = 1.5
PTL_AN_COAT = "Pt"       # 양극 PTL 코팅
PTL_AN_COAT_T_um = 0.1
PTL_CA_CC = 1.0          # 음극 carbon cloth 계수
BPP_T_mm = 1.0
BPP_COAT = "Au"
BPP_COAT_T_um = 0.1
FRAME_GASKET = 1.0
EP_AREA = 1.25           # 엔드플레이트 면적 계수

BOP_USD_KW = {10.0: 575.0, 100.0: 520.0, 1000.0: 377.0}

def _log_interpolate(x_value: float, table: dict) -> float:
    xs = np.array(sorted(table), dtype=float)
    ys = np.array([table[x] for x in xs], dtype=float)
    if x_value <= xs[0]:
        return float(ys[0])
    if x_value >= xs[-1]:
        lx = np.log10(xs[-2:])
        slope = (ys[-1] - ys[-2]) / (lx[1] - lx[0])
        return float(ys[-1] + slope * (np.log10(x_value) - lx[1]))
    return float(np.interp(np.log10(x_value), np.log10(xs), ys))

def _mem_cost_per_m2(thickness_um: float) -> float:
    x1, y1 = 180.0, MATERIAL_PRICE["membrane_180um"]
    x2, y2 = 80.0, MATERIAL_PRICE["membrane_80um"]
    price = y1 + (y2 - y1) * (thickness_um - x1) / (x2 - x1)
    return max(price, 0.30 * y2)

def _mass_material_g(area_cm2: float, thickness_um: float, density_gcm3: float) -> float:
    return area_cm2 * (thickness_um * 1.0e-4) * density_gcm3

def CRF(interest_rate: float | None = None, lifetime: int | None = None) -> float:
    i = INTEREST_RATE if interest_rate is None else interest_rate
    n = LIFETIME_YR if lifetime is None else lifetime
    return i * (1 + i) ** n / ((1 + i) ** n - 1.0)


# ###################################################################################
# ###################################################################################
##
##   PART 3 — 촉매 : 고정 Tafel 파라미터  
##
# ###################################################################################
# ###################################################################################

@dataclass
class CatalystParams:
    name: str
    slope_mV_dec: float                     # b [mV/decade]  ← 동역학의 유일한 입력
    j0_Acm2: float                          # 교환전류밀도 [A/cm2]
    alpha: float | None = None              # 검증용 (계산 미사용)
    R2: float | None = None
    n_fit: int | None = None
    fit_window_Acm2: tuple = (np.nan, np.nan)
    NdMn_atomic: float = 0.0                # 양극 blend 단가 산정용 (ICP Nd/Mn 원자비)
    metal_mass_frac: dict | None = None
    an_load_mgcm2: float = 8.0
    source: str = ""

    @property
    def b_V_dec(self) -> float:
        return self.slope_mV_dec / 1000.0

    def eta(self, j_Acm2):
        j = np.clip(np.atleast_1d(np.asarray(j_Acm2, dtype=float)), 1e-30, None)
        return self.b_V_dec * np.log10(j / self.j0_Acm2)

    def eta_fn(self):
        b, j0 = self.b_V_dec, self.j0_Acm2

        def _fn(j_Acm2):
            j = np.clip(np.atleast_1d(np.asarray(j_Acm2, dtype=float)), 1e-30, None)
            return b * np.log10(j / j0)
        return _fn

    def alpha_implied(self) -> float:
        return 2.303 * R_CONST * TAFEL_T_REF_K / (TAFEL_N_ELECTRON * F_CONST * self.b_V_dec)

    def convention_error(self) -> float | None:
        if self.alpha is None:
            return None
        return abs(self.alpha_implied() - self.alpha) / self.alpha

    def anode_price_usd_g(self) -> float:
        if self.metal_mass_frac:          
            return sum(w * METAL_PRICE[m] for m, w in self.metal_mass_frac.items())
        MW_Mn, MW_Nd = 54.938, 144.242
        m_Nd_rel = self.NdMn_atomic * MW_Nd
        w_Nd = m_Nd_rel / (MW_Mn + m_Nd_rel)
        return (1.0 - w_Nd) * METAL_PRICE["Mn"] + w_Nd * METAL_PRICE["Nd"]

    def extrapolation_decades(self, j_op_max: float) -> float:      # 외삽 확인용
        hi = self.fit_window_Acm2[1]
        if not np.isfinite(hi) or hi <= 0:
            return float("nan")
        return math.log10(max(j_op_max, 1e-30) / hi)

    def signature(self) -> tuple:                                   # 검산용
        return (self.name, self.slope_mV_dec, self.j0_Acm2)

_FIT_WIN = (3.162e-3, 3.162e-2)
_NDMN = {"MnO2": 0.0, "NMO-0.05": 0.0267, "NMO-0.1": 0.0651,
         "NMO-0.2": 0.1465, "NMO-0.3": 0.2610, "NMO-0.4": 0.3755}

PARAM_TABLE: dict[str, dict] = {
    "MnO2":     dict(slope_mV_dec=94.56, alpha=0.3706, j0_Acm2=9.70817771504645e-07,
                     R2=0.99943, n_fit=31),
    "NMO-0.05": dict(slope_mV_dec=89.30, alpha=0.3924, j0_Acm2=1.6808979002399986e-06,
                     R2=0.99976, n_fit=30),
    "NMO-0.1":  dict(slope_mV_dec=84.84, alpha=0.4131, j0_Acm2=1.6398759258616468e-06,
                     R2=0.99982, n_fit=28),
    "NMO-0.2":  dict(slope_mV_dec=88.28, alpha=0.3969, j0_Acm2=3.310545552558983e-06,
                     R2=0.99973, n_fit=29),     
    "NMO-0.3":  dict(slope_mV_dec=90.18, alpha=0.3886, j0_Acm2=3.7277310173498353e-06,
                     R2=0.99966, n_fit=30),
    "NMO-0.4":  dict(slope_mV_dec=89.10, alpha=0.3933, j0_Acm2=3.843397535469364e-06,
                     R2=0.99971, n_fit=30),
}
for _k, _v in PARAM_TABLE.items():
    _v.update(fit_window_Acm2=_FIT_WIN, NdMn_atomic=_NDMN.get(_k, 0.0),
              an_load_mgcm2=8.0, source="NMO_tafel_parameters.csv (내장)")
    
PARAM_TABLE["IrO2"] = dict(
    slope_mV_dec = 50.0,            # ← 확보한 b [mV/dec]
    j0_Acm2      = 1.0e-6,          # ← 확보한 j0 [A/cm2, geometric]
    alpha        = None,            # 규약 다르면 검증 끄기
    R2 = None, n_fit = None,
    fit_window_Acm2 = (1e-3, 1e-1), # ← 실제 측정 구간. 외삽 경고의 근거   
    NdMn_atomic  = 0.0,
    an_load_mgcm2 = 1.0,            # ★ Ir 기준 1~2 (상용 PEM)
    metal_mass_frac = {"Ir": 1.0},  # 로딩이 IrO2 질량 기준이면 0.8573
    source = "○○ et al. (연도), 0.5 M H2SO4, 25 C",     
    )
_TABLE_CACHE: dict | None = None
_ACTIVE_CATALYST: CatalystParams | None = None

def _param_table() -> dict:
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    table = {k: dict(v) for k, v in PARAM_TABLE.items()}
    if CATALYST_PARAM_FILE:
        for cand in (Path(CATALYST_PARAM_FILE),
                     Path(__file__).resolve().parent / CATALYST_PARAM_FILE,
                     Path.cwd() / CATALYST_PARAM_FILE):
            if cand.is_file():
                try:
                    df = pd.read_csv(cand)
                    for _, r in df.iterrows():
                        nm = str(r["catalyst"]).strip()
                        base = dict(table.get(nm, {}))
                        win = base.get("fit_window_Acm2", _FIT_WIN)
                        w = r.get("fit_window_log10j", None)
                        if isinstance(w, str) and w.strip().startswith("("):
                            lo, hi = [float(x) for x in w.strip('(") ').split(",")]
                            win = (10.0 ** lo / 1000.0, 10.0 ** hi / 1000.0)
                        base.update(slope_mV_dec=float(r["slope_mV_dec"]),
                                    j0_Acm2=float(r["j0_Acm2"]),
                                    alpha=None if pd.isna(r.get("alpha")) else float(r["alpha"]),
                                    R2=None if pd.isna(r.get("R2")) else float(r["R2"]),
                                    n_fit=None if pd.isna(r.get("n_fit")) else int(r["n_fit"]),
                                    fit_window_Acm2=win, source=cand.name)
                        base.setdefault("NdMn_atomic", _NDMN.get(nm, 0.0))
                        base.setdefault("an_load_mgcm2", 8.0)
                        table[nm] = base
                    print(f"[catalyst] 파라미터 CSV 반영: {cand}")
                except Exception as exc:
                    print(f"[catalyst][WARN] CSV 로드 실패 ({exc}) — 내장 표 사용")
                break
    _TABLE_CACHE = table
    return table

def get_catalyst() -> CatalystParams:
    global _ACTIVE_CATALYST
    if _ACTIVE_CATALYST is not None and _ACTIVE_CATALYST.name == CATALYST_NAME:
        return _ACTIVE_CATALYST
    table = _param_table()
    if CATALYST_NAME not in table:
        raise KeyError(f"촉매 '{CATALYST_NAME}' 파라미터가 없습니다. "
                       f"사용 가능: {list(table)}\n"
                       f"  -> PARAM_TABLE에 추가하거나 CSV에 행을 넣으세요.")
    _ACTIVE_CATALYST = CatalystParams(name=CATALYST_NAME, **table[CATALYST_NAME])
    return _ACTIVE_CATALYST

def set_catalyst(name: str) -> CatalystParams:
    global CATALYST_NAME, _ACTIVE_CATALYST
    CATALYST_NAME = name
    _ACTIVE_CATALYST = None
    return get_catalyst()

def report_catalyst(j_points=(0.1, 0.5, 1.0, 1.5, 2.0)) -> None:
    p = get_catalyst()
    lo, hi = p.fit_window_Acm2
    dec = p.extrapolation_decades(J_MAX)
    note = (f" | extrapolation {dec:.2f} dec" if np.isfinite(dec)
            and dec > EXTRAPOLATION_WARN_DECADES else "")
    print(f"[catalyst] {p.name}: b={p.slope_mV_dec:.2f} mV/dec, "
          f"j0={p.j0_Acm2:.3e} A/cm2, load={p.an_load_mgcm2:g} mg/cm2" + note)
    print(f"  source={p.source} | fit={lo*1000:.1f}-{hi*1000:.1f} mA/cm2")
    j = np.array(j_points, dtype=float)
    print(pd.DataFrame({"j [A/cm2]": j,
                        "eta_an [mV]": np.round(p.eta(j) * 1000, 1)})
          .to_string(index=False))

def list_catalysts() -> pd.DataFrame:
    rows = []
    for k, v in _param_table().items():
        cp = CatalystParams(name=k, **v)
        rows.append({"catalyst": k, "b [mV/dec]": cp.slope_mV_dec,
                     "j0 [A/cm2]": cp.j0_Acm2, "alpha": cp.alpha,
                     "Nd/Mn": cp.NdMn_atomic,
                     "anode $/g": round(cp.anode_price_usd_g(), 6),
                     "active": "★" if k == CATALYST_NAME else ""})
    return pd.DataFrame(rows)


# ###################################################################################
# ###################################################################################
##
##   PART 4 — PEM 셀 전압 모델
##
# ###################################################################################
# ###################################################################################

def membrane_conductivity(T_K: float | None = None,                 # Springer model
                          lam: float | None = None,
                          scale: float | None = None) -> float:
    T_K = T_OPER if T_K is None else T_K
    lam = MEMB_LAMBDA if lam is None else lam
    scale = SIGMA_SCALE if scale is None else scale
    sigma_30 = 0.005139 * lam - 0.00326
    sigma = sigma_30 * math.exp(1268.0 * (1.0 / 303.0 - 1.0 / T_K))
    return max(scale * sigma, 1.0e-4)

def cell_voltage_parts(j_Acm2) -> dict:
    j = np.atleast_1d(np.asarray(j_Acm2, dtype=float))
    j = np.clip(j, 1.0e-30, None)
    if np.any(j >= J_LIM):
        raise ValueError(f"전류밀도가 j_lim({J_LIM})에 도달했습니다 — ln(1-j/j_lim) 발산")

    T = T_OPER
    p_sat = 10.0 ** (35.4462 - 3343.93 / T - 10.9 * np.log10(T) + 4.1645e-3 * T)
    p_h2 = max(P_CAT - p_sat, 1.0e-12)
    p_o2 = max(P_AN - p_sat, 1.0e-12)
    U0 = 1.5184 - 1.5421e-3 * T + 9.523e-5 * T * np.log(T) + 9.84e-8 * T ** 2
    e_rev = U0 + (R_CONST * T / (2.0 * F_CONST)) * np.log(p_h2 * np.sqrt(p_o2))

    eta_act_an = get_catalyst().eta(j)

    eta_act_ca = (R_CONST * T / (N_ELECTRON_H2 * ALPHA_CA * F_CONST)) \
        * np.arcsinh(j / (2.0 * J0_CA))

    memb_asr = (MEMB_T_UM * 1.0e-4) / membrane_conductivity()
    eta_ohm = j * memb_asr

    eta_mt = -(R_CONST * T / (N_ELECTRON_H2 * F_CONST)) * np.log(1.0 - j / J_LIM)

    return {"j": j, "e_rev": np.full_like(j, e_rev), "eta_act_an": eta_act_an,
            "eta_act_ca": eta_act_ca, "eta_ohm": eta_ohm, "eta_mt": eta_mt,
            "memb_asr": memb_asr,
            "V": e_rev + eta_act_an + eta_act_ca + eta_ohm + eta_mt}

def report_cell_voltage(j_points=(0.1, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    j = np.array([x for x in j_points if 0 < x < J_LIM], dtype=float)
    d = cell_voltage_parts(j)
    df = pd.DataFrame({
        "j [A/cm2]": j,
        "e_rev [V]": np.round(d["e_rev"], 4),
        "eta_an [mV]": np.round(d["eta_act_an"] * 1000, 1),
        "eta_ca [mV]": np.round(d["eta_act_ca"] * 1000, 1),
        "eta_ohm [mV]": np.round(d["eta_ohm"] * 1000, 1),
        "eta_mt [mV]": np.round(d["eta_mt"] * 1000, 1),
        "V [V]": np.round(d["V"], 4),
    })
    print(f"\n[PART 4] cell voltage | catalyst={get_catalyst().name}, "
          f"T={T_OPER - 273.15:.0f} C, membrane={MEMB_T_UM:.0f} um, "
          f"ASR={d['memb_asr']:.5f} ohm-cm2")
    print(df.to_string(index=False))
    return df


# ###################################################################################
# ###################################################################################
##
##   PART 5 — 물리 커널 래퍼 + P <-> j 역산       
##
# ###################################################################################
# ###################################################################################

def faraday_effi(j):
    j = np.asarray(j, dtype=float)
    if FARADAY_MODE == "crossover":
        jx = J_X_mAcm2 / 1000.0
        return np.clip(1.0 - jx / np.maximum(j, 1e-12), 0.0, 1.0)
    return np.ones_like(j)

def V(j):
    j = np.atleast_1d(np.asarray(j, dtype=float))
    out = np.zeros_like(j)
    on = j > 0.0
    if np.any(on):
        out[on] = cell_voltage_parts(j[on])["V"]
    return out

def SEC_stack(j):
    return V(j) * 2.0 * F_CONST / MW_H2 / 3.6e6

def SEC_system(j):
    return SEC_stack(j) + BOP_EFFICIENCY

def h2_area(j):
    j = np.atleast_1d(np.asarray(j, dtype=float))
    return faraday_effi(j) * j / (2.0 * F_CONST) * MW_H2 * 3600.0

def p_sys(j):
    j = np.atleast_1d(np.asarray(j, dtype=float))
    return j * V(j) / 1000.0 + BOP_EFFICIENCY * h2_area(j)

def P_of_j(j, A_tot):
    return A_tot * p_sys(j)

def split_power(j, A_tot):
    j = float(np.atleast_1d(j)[0])
    P_stack = A_tot * j * float(V(j)[0]) / 1000.0
    P_bop = A_tot * BOP_EFFICIENCY * float(h2_area(j)[0])
    P_tot = A_tot * float(p_sys(j)[0])
    assert abs(P_stack + P_bop - P_tot) / max(P_tot, 1e-12) < 1e-10, \
        "split_power: P_stack + P_bop != P_star (플랜트 경계 위반)"
    return P_stack, P_bop

@dataclass
class OperatingWindow:
    j_min: float
    j_max: float
    P_min: float
    P_max: float
    A_tot: float

    @property
    def current_turndown(self) -> float:
        return self.j_min / self.j_max

    @property
    def power_turndown(self) -> float:
        return self.P_min / self.P_max

    def describe(self) -> str:
        return (f"  j window : {self.j_min:.4f} ~ {self.j_max:.4f} A/cm2\n"
                f"  P window : {self.P_min:,.1f} ~ {self.P_max:,.1f} kW\n"
                f"  turndown : current {self.current_turndown * 100:.2f} %  vs  "
                f"power {self.power_turndown * 100:.2f} %   "
                f"(전력 turndown이 더 깊다 — 저 j에서 V(j)도 낮아지므로)")

def operating_window(A_tot: float, j_min: float | None = None,
                     j_max: float | None = None) -> OperatingWindow:
    j_min = J_MIN if j_min is None else j_min
    j_max = J_MAX if j_max is None else j_max
    if j_max >= J_LIM:
        raise ValueError(f"j_max({j_max}) must be < j_lim({J_LIM})")
    return OperatingWindow(j_min, j_max,
                           float(P_of_j(j_min, A_tot)[0]),
                           float(P_of_j(j_max, A_tot)[0]), A_tot)

def j_min_derivation() -> str:
    return (f"j_min = 50 x {J_MIN_SAFETY:.1f}(안전계수) x j_x({J_X_mAcm2:.1f} mA/cm2) "
            f"= {J_MIN:.4f} A/cm2  ({J_MIN / J_MAX * 100:.1f} % of j_max)\n"
            f"   근거: 양극 O2=j/(4F), 크로스오버 H2=j_x/(2F) -> x_H2 ~ 2*j_x/j = 4 %(LFL)")

def _kernel_signature() -> tuple:
    return (T_OPER, MEMB_T_UM, MEMB_LAMBDA, SIGMA_SCALE, J_LIM, BOP_EFFICIENCY,
            get_catalyst().signature(), FARADAY_MODE, J_X_mAcm2, J0_CA, ALPHA_CA)

class PowerToJ:

    def __init__(self, A_tot: float, j_lo: float, j_hi: float, n: int | None = None):
        self.A_tot, self.j_lo, self.j_hi = float(A_tot), float(j_lo), float(j_hi)
        self.n = int(n or INVERSION_GRID_N)
        self.signature = _kernel_signature()
        self._jg = np.linspace(self.j_lo, self.j_hi, self.n)
        self._pg = self.A_tot * p_sys(self._jg)
        if not np.all(np.diff(self._pg) > 0):
            raise RuntimeError("p_sys(j)가 단조증가가 아닙니다 — 역산 불가")
        self.max_roundtrip_error = self._roundtrip_error()

    def _roundtrip_error(self, n_test: int = 997) -> float:      
        jt = np.linspace(self.j_lo, self.j_hi, n_test)
        return float(np.max(np.abs(self(self.A_tot * p_sys(jt)) - jt)
                            / np.maximum(jt, 1e-12)))

    def __call__(self, P):
        return np.interp(np.asarray(P, dtype=float), self._pg, self._jg)

    def validate(self, tol: float | None = None, verbose: bool = True) -> float:   
        tol = INVERSION_TOL if tol is None else tol
        err = self.max_roundtrip_error
        if verbose:
            print(f"  P<->j 역산 왕복 상대오차 : {err:.3e}  (tol {tol:.0e})  "
                  f"[{'PASS' if err < tol else 'FAIL'}]   grid N={self.n}")
        assert err < tol, f"역산 왕복오차 {err:.3e} >= {tol:.0e} — 그리드 N을 늘리세요"
        return err

    def stale(self) -> bool:
        return self.signature != _kernel_signature()

_INVERTER_CACHE: dict = {}

def get_inverter(A_tot: float, j_lo: float, j_hi: float,        # j->P 표를 새로 만들기 (온도, 촉매가 바뀔 시)
                 n: int | None = None) -> PowerToJ:
    key = (round(A_tot, 6), round(j_lo, 8), round(j_hi, 8),
           int(n or INVERSION_GRID_N), _kernel_signature())
    inv = _INVERTER_CACHE.get(key)
    if inv is None or inv.stale():
        inv = PowerToJ(A_tot, j_lo, j_hi, n)
        _INVERTER_CACHE[key] = inv
    return inv

def report_kernel(j_points=(0.1, 0.5, 1.0, 1.5, 2.0, 2.5)) -> None:
    p = get_catalyst()
    j = np.array([x for x in j_points if 0 < x < J_LIM], dtype=float)
    df = pd.DataFrame({
        "j [A/cm2]": j,
        "V [V]": np.round(V(j), 4),
        "SEC_stack [kWh/kg]": np.round(SEC_stack(j), 2),
        "SEC_system [kWh/kg]": np.round(SEC_system(j), 2),
        "h2_area [g/h/cm2]": np.round(h2_area(j) * 1000.0, 5),
        "p_sys [W/cm2]": np.round(p_sys(j) * 1000.0, 4),
    })
    print(f"\n[PART 5] kernel | catalyst={p.name}, T={T_OPER - 273.15:.0f} C, "
          f"BOP={BOP_EFFICIENCY} kWh/kg, faraday={FARADAY_MODE}")
    print(df.to_string(index=False))


# ###################################################################################
# ###################################################################################
##
##   PART 6 — 스택 원가 + 고정 스택 사이징           
##
# ###################################################################################
# ###################################################################################

def stack_material_costs(N_total_cells: float, N_stacks: float,
                         A_cell_cm2: float | None = None) -> dict:
    A_cell_cm2 = A_CELL_CM2 if A_cell_cm2 is None else A_cell_cm2
    area_cm2 = N_total_cells * A_cell_cm2
    area_m2 = area_cm2 / 1.0e4
    cat = get_catalyst()

    membrane = _mem_cost_per_m2(MEMB_T_UM) * area_m2

    an_cat_g = cat.an_load_mgcm2 * area_cm2 / 1000.0
    an_cat = an_cat_g * cat.anode_price_usd_g() * 1.3

    ca_cat_g = CA_LOAD_mgcm2 * area_cm2 / 1000.0
    ca_cat = ca_cat_g * METAL_PRICE[CA_CAT] * 1.3

    ptl_an_bulk = _mass_material_g(area_cm2, PTL_AN_T_mm * 1000.0,
                                   DENSITY[PTL_AN_MAT]) * MATERIAL_PRICE["ti_powder"]
    ptl_an_coat = _mass_material_g(area_cm2, PTL_AN_COAT_T_um,
                                   DENSITY[PTL_AN_COAT]) * METAL_PRICE[PTL_AN_COAT]
    ptl_ca = area_m2 * MATERIAL_PRICE["carbon_cloth"] * PTL_CA_CC

    bpp_bulk = _mass_material_g(area_cm2, BPP_T_mm * 1000.0,
                                DENSITY["SS316"]) * MATERIAL_PRICE["ss316"]
    bpp_coat = _mass_material_g(area_cm2, BPP_COAT_T_um,
                                DENSITY[BPP_COAT]) * METAL_PRICE[BPP_COAT]

    gasket_kg = area_m2 * 2.0 * FRAME_GASKET
    frame_gasket = (0.7 * gasket_kg * MATERIAL_PRICE["PPS40GF"]
                    + 0.3 * gasket_kg * MATERIAL_PRICE["ptfe"])

    endplate = N_stacks * 2.0 * (A_cell_cm2 / 1.0e4) * EP_AREA * MATERIAL_PRICE["a356_al"]

    total = (membrane + an_cat + ca_cat + ptl_an_bulk + ptl_an_coat + ptl_ca
             + bpp_bulk + bpp_coat + frame_gasket + endplate)
    return {"membrane": membrane, "anode_catalyst": an_cat, "cathode_catalyst": ca_cat,
            "ptl_anode_bulk": ptl_an_bulk, "ptl_anode_coating": ptl_an_coat,
            "ptl_cathode": ptl_ca, "bpp_bulk": bpp_bulk, "bpp_coating": bpp_coat,
            "frame_gasket": frame_gasket, "endplate": endplate,
            "total_material_cost": total, "total_active_area_m2": area_m2}

def stack_total_cost(N_total_cells: float, N_stacks: float,
                     A_cell_cm2: float | None = None) -> dict:
    mat = stack_material_costs(N_total_cells, N_stacks, A_cell_cm2)
    direct = mat["total_material_cost"] * (8.0 + 4.0 + 1.0) / 8.0
    return {**mat, "direct_stack_cost": direct,
            "total_stack_cost": direct / (1.0 - OVHD_FRAC)}

def bop_total_capex(P_plant_kW: float) -> dict:
    usd_kW = _log_interpolate(MANUFACTURING_RATE, BOP_USD_KW)
    manufactured = usd_kW * P_plant_kW
    return {"bop_usd_kW": usd_kW, "bop_manufactured": manufactured,
            "bop_total": manufactured * (1.0 + MARKUP)}

@dataclass
class FixedStack:
    A_cell: float
    N_cells: float
    N_stacks: float
    A_tot: float
    j_rated: float
    P_nameplate: float
    P_stack_rated: float
    P_bop_rated: float
    capex_stack: float
    capex_bop: float
    N_cells_stack_boundary: float
    oversize_pct: float
    cost_breakdown: dict = field(default_factory=dict)

    @property
    def capex_fix(self) -> float:
        return self.capex_stack + self.capex_bop

    def report(self) -> None:
        resid = self.P_stack_rated + self.P_bop_rated - self.P_nameplate
        print("\n[PART 6] fixed stack sizing")
        print(f"  cells={self.N_cells:,.1f} ({self.N_stacks:.2f} stacks), "
              f"A_tot={self.A_tot:,.0f} cm2, j_rated={self.j_rated:.3f} A/cm2")
        print(f"  nameplate={self.P_nameplate:,.1f} kW "
              f"(stack={self.P_stack_rated:,.1f}, BOP={self.P_bop_rated:,.1f}, "
              f"resid={resid:+.3e} kW)")
        print(f"  CAPEX stack={self.capex_stack:,.0f}, BOP={self.capex_bop:,.0f}, "
              f"fixed={self.capex_fix:,.0f} USD")

def size_fixed_stack(P_target_kW: float, j_rated: float | None = None) -> FixedStack:
    j_rated = J_MAX if j_rated is None else float(j_rated)
    p_cell_plant = float(p_sys(j_rated)[0]) * A_CELL_CM2          # [kW/cell] 플랜트 경계
    p_cell_stack = j_rated * float(V(j_rated)[0]) * A_CELL_CM2 / 1000.0

    N_cells = P_target_kW / p_cell_plant
    N_cells_sb = P_target_kW / p_cell_stack
    oversize = (N_cells_sb / N_cells - 1.0) * 100.0

    N_stacks = N_cells / N_CELL_PER_STACK
    if INTEGER_STACKS:
        N_stacks = math.ceil(N_stacks)
        N_cells = N_stacks * N_CELL_PER_STACK

    A_tot = N_cells * A_CELL_CM2
    P_stack_r, P_bop_r = split_power(j_rated, A_tot)
    P_nameplate = P_stack_r + P_bop_r

    cost = stack_total_cost(N_cells, N_stacks, A_CELL_CM2)
    return FixedStack(A_cell=A_CELL_CM2, N_cells=N_cells, N_stacks=N_stacks, A_tot=A_tot,
                      j_rated=j_rated, P_nameplate=P_nameplate,
                      P_stack_rated=P_stack_r, P_bop_rated=P_bop_r,
                      capex_stack=float(cost["total_stack_cost"]),
                      capex_bop=float(bop_total_capex(P_nameplate)["bop_total"]),
                      N_cells_stack_boundary=N_cells_sb, oversize_pct=oversize,
                      cost_breakdown=cost)

def build_fixed_stack(profile=None, verbose: bool = True) -> FixedStack:
    if STACK_SIZING_MODE == "nameplate_kW":
        P_target, basis = STACK_NAMEPLATE_KW, f"명판 직접 지정 {STACK_NAMEPLATE_KW:,.0f} kW"
    elif STACK_SIZING_MODE == "pv_fraction":
        if profile is None:
            raise ValueError("'pv_fraction' 모드는 profile이 필요합니다.")
        P_target = STACK_PV_FRACTION * profile.peak
        basis = f"배정 PV 피크 {profile.peak:,.0f} kW x {STACK_PV_FRACTION:.2f}"
    elif STACK_SIZING_MODE == "cells":
        P_target = STACK_N_CELLS * A_CELL_CM2 * float(p_sys(J_MAX)[0])
        basis = f"셀 수 직접 지정 {STACK_N_CELLS:,.0f} cells"
    else:
        raise ValueError(f"Unknown STACK_SIZING_MODE: {STACK_SIZING_MODE}")

    st = size_fixed_stack(P_target)
    if verbose:
        print(f"\n[sizing] mode={STACK_SIZING_MODE}, target={P_target:,.1f} kW ({basis})")
        st.report()
    return st

def rippl_capacity(P_in: np.ndarray, P_star: float, eta_RTE: float,
                   dt: float = 1.0) -> float:
    C = np.cumsum((np.asarray(P_in, dtype=float) - P_star / eta_RTE) * dt)
    return float(C.max() - C.min())


# ###################################################################################
# ###################################################################################
##
##   PART 7 — 재생에너지 8760 h 프로파일 로더        
##
# ###################################################################################
# ###################################################################################

HOURS_PER_YEAR = 8760
_TIME_HINTS = ("time", "timestamp", "date", "datetime", "일시", "시간", "날짜")
_POWER_HINTS = ("kw", "power", "gen", "output", "발전", "출력", "전력", "kwh")

@dataclass
class REProfile:
    P_in: np.ndarray                  # PEM 공급전력 [kW]
    P_gen: np.ndarray                 # PV 발전량 [kW]
    timestamps: pd.DatetimeIndex
    dt: float = 1.0
    allocation_n: float = 1.0
    region: str = ""
    year: int = 0
    source: str = ""
    url: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_hours(self) -> int:
        return len(self.P_in)

    @property
    def E_paid(self) -> float:
        return float(self.P_in.sum() * self.dt)

    @property
    def peak(self) -> float:
        return float(self.P_in.max())

    @property
    def mean(self) -> float:
        return float(self.P_in.mean())

    @property
    def capacity_factor(self) -> float:
        cap_mw = self.meta.get("re_capacity_MW", None)
        try:
            cap_kw = float(cap_mw) * 1000.0
        except (TypeError, ValueError):
            cap_kw = 0.0
        if np.isfinite(cap_kw) and cap_kw > 0:
            return self.mean / cap_kw
        return self.mean / self.peak if self.peak > 0 else 0.0
    
    @property
    def month(self) -> np.ndarray:
        return self.timestamps.month.to_numpy()

    def pctl(self, q: float) -> float:
        return float(np.percentile(self.P_in, q))

    def daily_energy(self) -> np.ndarray:
        s = pd.Series(self.P_in * self.dt, index=self.timestamps)
        return s.groupby(s.index.dayofyear).sum().to_numpy()

    def representative_days(self) -> dict:
        de = self.daily_energy()
        doy = self.timestamps.dayofyear.to_numpy()
        best = int(np.argmax(de)) + 1
        pool = np.where(de > de.max() * 0.05)[0]
        worst = int(pool[np.argmin(de[pool])]) + 1 if len(pool) else 1
        return {"summer_peak": np.where(doy == best)[0],
                "winter_low": np.where(doy == worst)[0]}

    def report(self) -> None:
        nz = self.P_in > 0
        print("\n[PART 7] renewable profile")
        print(f"  {self.region} / {self.year} | source={self.source}" +
              (f" ({self.url})" if self.url else ""))
        print(f"  hours={self.n_hours}, allocation={self.allocation_n * 100:.1f} %, "
              f"E_paid={self.E_paid / 1e3:,.1f} MWh")
        print(f"  peak/mean={self.peak:,.1f}/{self.mean:,.1f} kW, "
              f"CF={self.capacity_factor * 100:.2f} %, nonzero={int(nz.sum()):,d} h")

def _pick_column(df, hints, exclude=()):
    for col in df.columns:
        if col in exclude:
            continue
        if any(h in str(col).strip().lower() for h in hints):
            return col
    return None

def load_profile_from_file(path: str, time_col=None, power_col=None,
                           allocation_n: float | None = None) -> REProfile:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"재생 프로파일 파일 없음: {p}")
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)

    tcol = time_col or _pick_column(df, _TIME_HINTS)
    pcol = power_col or _pick_column(df, _POWER_HINTS, exclude=(tcol,))
    if pcol is None:
        num = df.select_dtypes("number").columns.tolist()
        pcol = num[-1] if num else None
    if pcol is None:
        raise ValueError(f"발전량 컬럼을 찾지 못했습니다. columns={list(df.columns)}")

    P_gen = np.clip(np.nan_to_num(pd.to_numeric(df[pcol], errors="coerce")
                                  .to_numpy(dtype=float), nan=0.0), 0.0, None)
    ts = None
    if tcol is not None:
        t = pd.to_datetime(df[tcol], errors="coerce")
        ts = pd.DatetimeIndex(t) if not t.isna().all() else None
    if ts is None:
        ts = pd.date_range(f"{RE_YEAR}-01-01", periods=len(P_gen), freq="h")

    if len(P_gen) != HOURS_PER_YEAR:
        print(f"[WARN] 프로파일 길이 {len(P_gen)} != 8760 -> 8760으로 절단/패딩")
        if len(P_gen) > HOURS_PER_YEAR:
            P_gen, ts = P_gen[:HOURS_PER_YEAR], ts[:HOURS_PER_YEAR]
        else:
            P_gen = np.concatenate([P_gen, np.zeros(HOURS_PER_YEAR - len(P_gen))])
            ts = pd.date_range(ts[0], periods=HOURS_PER_YEAR, freq="h")

    n = ALLOCATION_N if allocation_n is None else allocation_n
    return REProfile(P_in=P_gen * n, P_gen=P_gen, timestamps=ts, dt=DT_HOURS,
                     allocation_n=n, region=RE_REGION, year=RE_YEAR,
                     source=f"file:{p.name}", url=RE_SOURCE_URL,
                     meta={"time_col": tcol, "power_col": pcol})

_PROFILE_CACHE: REProfile | None = None

def find_region_file(region: str | None = None, kind: str | None = None,
                     data_root: str | None = None) -> str:
    import glob
    import os
    import unicodedata

    def nfc(s):
        return unicodedata.normalize("NFC", str(s))

    root = data_root or RE_DATA_ROOT
    region = nfc(region or RE_REGION_NAME)
    kind = nfc(kind or RE_KIND)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {os.path.abspath(root)}\n"
            f"  -> PART 1 의 RE_DATA_ROOT 를 실제 경로로 고쳐주세요.")

    hits, all_regions = [], set()
    for p in glob.glob(os.path.join(root, "**", "*.xlsx"), recursive=True):
        n = nfc(p).replace("\\", "/")
        if "1MW" not in n or f"/{kind}/" not in n:
            continue
        base = nfc(os.path.basename(p))
        all_regions.add(base.split("_")[0])
        if base.startswith(region + "_"):
            hits.append(p)
    if not hits:
        raise FileNotFoundError(
            f"'{region} {kind}' 파일이 없습니다.\n"
            f"  사용 가능한 지역: {', '.join(sorted(all_regions)) or '(없음)'}")
    if len(hits) > 1:
        print(f"[WARN] 후보 {len(hits)}개 -> 첫 번째 사용: {os.path.basename(hits[0])}")
    return hits[0]

def load_profile_from_region(region: str | None = None, kind: str | None = None,
                             year: int | None = None, re_mw: float | None = None,
                             data_root: str | None = None,
                             verbose: bool = True) -> REProfile:
    import os

    region = region or RE_REGION_NAME
    kind = kind or RE_KIND
    year = int(RE_YEAR if year is None else year)
    re_mw = float(RE_CAPACITY_MW if re_mw is None else re_mw)

    path = find_region_file(region, kind, data_root)
    df = pd.read_excel(path)
    for col in (RE_DATETIME_COL, RE_VALUE_COL):
        if col not in df.columns:
            raise KeyError(f"'{col}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")

    ts_all = pd.to_datetime(df[RE_DATETIME_COL], errors="coerce")
    df = df.loc[ts_all.dt.year == year].copy()
    df = df.sort_values(RE_DATETIME_COL, kind="stable")
    ts = pd.DatetimeIndex(pd.to_datetime(df[RE_DATETIME_COL]))

    raw = pd.to_numeric(df[RE_VALUE_COL], errors="coerce").to_numpy(dtype=float)
    n_nan = int(np.sum(~np.isfinite(raw)))
    if n_nan:
        print(f"[WARN] 결측 {n_nan:,}개 ({n_nan / len(raw) * 100:.2f} %) -> 0 kW 처리")
    p_unit = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)

    n = len(p_unit)
    if n == 0:
        raise ValueError(f"{region} {kind} {year}년 데이터가 0행입니다. RE_YEAR 확인.")
    leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    expected = 8784 if leap else 8760
    if n != expected:
        missing = [m for m in range(1, 13) if m not in set(ts.month)]
        print(f"[WARN] {region} {kind} {year}: {n:,} h (expected {expected:,} h) "
              + (f"missing months {missing}" if missing else "missing timestamps"))
    cf = p_unit.mean() / 1000.0
    if cf < 0.01:
        raise ValueError(f"{region} {kind} {year} 이용률 {cf * 100:.2f}% — 사용 불가 데이터")
    if cf < 0.05:
        print(f"[WARN] 이용률 {cf * 100:.2f}% — 비정상적으로 낮습니다. 원자료 확인 필요")

    P_gen = p_unit * re_mw
    P_in = P_gen * float(ALLOCATION_N)

    prof = REProfile(
        P_in=P_in, P_gen=P_gen, timestamps=ts, dt=DT_HOURS,
        allocation_n=float(ALLOCATION_N),
        region=f"{region} {kind}", year=year,
        source=f"file:{os.path.basename(path)} x {re_mw:g} MW",
        url=RE_SOURCE_URL,
        meta={"re_capacity_MW": re_mw, "value_col": RE_VALUE_COL,
              "unit_CF": float(cf), "path": path, "kind": kind},
    )
    if verbose:
        print(f"\n[RE loader] {region} {kind} {year} ({os.path.basename(path)}): "
              f"{n:,} h, {re_mw:g} MW, CF={cf * 100:.2f} %, "
              f"peak/mean={P_in.max():,.1f}/{P_in.mean():,.1f} kW, "
              f"E_paid={P_in.sum() / 1e3:,.1f} MWh")
    return prof

def _profile_signature() -> tuple:
    return (RE_SOURCE, RE_REGION_NAME, RE_KIND, RE_YEAR,
            RE_CAPACITY_MW, ALLOCATION_N, RE_FILE_PATH, RE_DATA_ROOT)

_PROFILE_CACHE: REProfile | None = None
_PROFILE_SIG: tuple | None = None

def get_profile(force_reload: bool = False) -> REProfile:
    global _PROFILE_CACHE, _PROFILE_SIG
    sig = _profile_signature()
    if _PROFILE_CACHE is not None and not force_reload and _PROFILE_SIG == sig:
        return _PROFILE_CACHE

    if RE_SOURCE == "region":
        prof = load_profile_from_region()
    elif RE_SOURCE == "file":
        if not RE_FILE_PATH:
            raise ValueError("RE_SOURCE='file' 인데 RE_FILE_PATH 가 비어 있습니다.")
        prof = load_profile_from_file(RE_FILE_PATH, RE_TIME_COL, RE_POWER_COL)
    else:
        raise ValueError(f"RE_SOURCE 값이 잘못되었습니다 -> {RE_SOURCE!r}")

    _PROFILE_CACHE, _PROFILE_SIG = prof, sig
    return _PROFILE_CACHE


# ###################################################################################
# ###################################################################################
##
##   PART 8 — 결과 스키마 · 손실 원장 · 지표          [3케이스 공통 '그릇']
##
# ###################################################################################
# ###################################################################################

@dataclass
class DispatchResult:
    case: str
    dt: float = 1.0
    P_in: np.ndarray = field(default_factory=lambda: np.zeros(0))
    j: np.ndarray = field(default_factory=lambda: np.zeros(0))
    P_used: np.ndarray = field(default_factory=lambda: np.zeros(0))
    H2_series: np.ndarray = field(default_factory=lambda: np.zeros(0))
    soc: np.ndarray | None = None
    running: np.ndarray | None = None
    charge: np.ndarray | None = None
    discharge: np.ndarray | None = None
    curtail_series: np.ndarray | None = None
    E_paid: float = 0.0
    E_curtail: float = 0.0
    E_idle: float = 0.0
    E_rte: float = 0.0
    E_bop: float = 0.0
    E_stack: float = 0.0
    E_direct: float = 0.0
    E_via_ess: float = 0.0
    H2_total: float = 0.0
    op_hours: float = 0.0
    hours_curtail: float = 0.0
    hours_idle: float = 0.0
    hours_night: float = 0.0
    restarts: int = 0
    eq_cycles: float = 0.0
    E_end: float = 0.0
    E_rated: float = 0.0
    P_rated_chg: float = 0.0
    P_rated_dis: float = 0.0
    j_star: float | None = None
    P_star: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def E_used(self) -> float:
        return self.E_bop + self.E_stack

    @property
    def E_unabsorbed(self) -> float:
        return self.E_curtail + self.E_idle
    
    @property
    def ledger(self) -> dict:
        return {"curtail": self.E_curtail, "idle": self.E_idle, "rte": self.E_rte,
                "bop": self.E_bop, "stack": self.E_stack}

    def balance_error(self) -> float:
        if self.E_paid <= 0:
            return 0.0
        return abs(self.E_paid - sum(self.ledger.values())) / self.E_paid

    def assert_balance(self, tol: float | None = None, verbose: bool = True) -> float:
        tol = BALANCE_TOL if tol is None else tol
        err = self.balance_error()
        if verbose:
            print(f"  balance error={err * 100:.4f} % "
                  f"[{'PASS' if err < tol else 'FAIL'}]")
        assert err < tol, (f"[{self.case}] 손실 원장이 닫히지 않습니다: "
                           f"{err * 100:.3f} % >= {tol * 100:.0f} %")
        return err

    def SEC_eff(self) -> float:
        return self.E_paid / self.H2_total if self.H2_total > 0 else float("nan")

    def SEC_used(self) -> float:
        return self.E_used / self.H2_total if self.H2_total > 0 else float("nan")

    def SEC_stack_avg(self) -> float:
        return self.E_stack / self.H2_total if self.H2_total > 0 else float("nan")

    def util(self, h2_area_at: float, A_tot: float, n_hours=None) -> float:
        n_hours = n_hours or len(self.P_in) or HOURS_PER_YEAR
        d = h2_area_at * A_tot * n_hours * self.dt
        return self.H2_total / d if d > 0 else float("nan")

    def ledger_table(self) -> pd.DataFrame:
        labels = {"curtail": "낭비 · curtailment", "idle": "낭비 · 하한미만 정지",
                  "rte": "낭비 · ESS 왕복손실", "bop": "실사용 · BOP(오버헤드)",
                  "stack": "실사용 · stack(생산)"}
        tot = self.E_paid
        rows = [{"bucket": labels[k], "MWh": v / 1e3,
                 "share_%": (v / tot * 100) if tot > 0 else np.nan}
                for k, v in self.ledger.items()]
        rows.append({"bucket": "합계 (= E_paid)", "MWh": tot / 1e3, "share_%": 100.0})
        return pd.DataFrame(rows)

    def report(self, h2_area_at_rated=None, h2_area_at_jstar=None, A_tot=None) -> None:
        n = len(self.P_in) or HOURS_PER_YEAR
        self.assert_balance(verbose=False)
        print(f"\n[{self.case}] H2={self.H2_total / 1000:,.2f} t | "
              f"SEC_eff={self.SEC_eff():.2f} kWh/kg | op={self.op_hours:,.0f} h | "
              f"balance={self.balance_error() * 100:.4f} %")
        if self.E_paid > 0:
            print(f"  ledger share: curtail={self.E_curtail / self.E_paid * 100:.2f} %, "
                  f"idle={self.E_idle / self.E_paid * 100:.2f} %, "
                  f"rte={self.E_rte / self.E_paid * 100:.2f} %, "
                  f"used={self.E_used / self.E_paid * 100:.2f} %")
        if A_tot and h2_area_at_rated:
            print(f"  util vs rated={self.util(h2_area_at_rated, A_tot, n) * 100:.2f} %")
        if A_tot and h2_area_at_jstar:
            print(f"  util vs j*={self.util(h2_area_at_jstar, A_tot, n) * 100:.2f} %")
        if self.E_rated > 0:
            print(f"  ESS: E={self.E_rated:,.0f} kWh, "
                  f"Pchg/Pdis={self.P_rated_chg:,.0f}/{self.P_rated_dis:,.0f} kW, "
                  f"restarts={self.restarts:,d}, cycles={self.eq_cycles:,.1f}, "
                  f"SOC_end={self.E_end / self.E_rated * 100:.1f} %")
        if self.E_via_ess > 0 or self.E_direct > 0:
            tot = self.E_direct + self.E_via_ess
            print(f"  path: direct={self.E_direct / 1e3:,.1f} MWh "
                  f"({self.E_direct / tot * 100:.1f} %), "
                  f"via_ESS={self.E_via_ess / 1e3:,.1f} MWh "
                  f"({self.E_via_ess / tot * 100:.1f} %)")

def compare_table(results: dict, lcohs: dict | None = None) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"case": name, "annual_H2 [t]": r.H2_total / 1000.0,
               "SEC_eff [kWh/kg]": r.SEC_eff(), "op_hours [h]": r.op_hours,
               "curtail [%]": r.E_curtail / r.E_paid * 100 if r.E_paid else np.nan,
               "idle [%]": r.E_idle / r.E_paid * 100 if r.E_paid else np.nan,
               "RTE loss [%]": r.E_rte / r.E_paid * 100 if r.E_paid else np.nan,
               "E_rated [kWh]": r.E_rated, "eq_cycles": r.eq_cycles,
               "restarts": r.restarts, "balance err [%]": r.balance_error() * 100}
        if lcohs and name in lcohs:
            row["LCOH [$/kg]"] = lcohs[name]["LCOH"]
            row["ESS share [%]"] = lcohs[name].get("ess_share_pct", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ###################################################################################
# ###################################################################################
##
##   PART 9 — LCOH 경제성                             [3케이스 공통 뼈대]
##
# ###################################################################################
# ###################################################################################

def ess_capex(E_rated_kWh: float, P_rated_kW: float,
              c_E: float | None = None, c_P: float | None = None) -> float:
    c_E = C_E_USD_KWH if c_E is None else c_E
    c_P = C_P_USD_KW if c_P is None else c_P
    return c_E * E_rated_kWh + c_P * P_rated_kW

def _lcoh_core(*, annual_H2, op_hours, capex_stack, capex_bop, capex_ess,
               elec_cost, annual_batt_repl=None, curtail_penalty=0.0) -> dict:
    if annual_H2 <= 0:
        return {"LCOH": float("inf"), "annual_H2": 0.0}
    capex_total = capex_stack + capex_bop + capex_ess
    om_base = capex_total if FIXED_OM_INCLUDES_ESS else (capex_stack + capex_bop)
    batt = ANNUAL_BATT_REPL_USD if annual_batt_repl is None else annual_batt_repl

    items = {
        "annualized_capex": capex_total * CRF(),
        "fixed_OM": om_base * FIXED_OM_FRAC,
        "stack_replacement": capex_stack * (op_hours / STACK_LIFETIME_H) * REPL_STACK_FRAC,
        "battery_replacement": batt,
        "electricity": elec_cost,
        "curtailment_penalty": curtail_penalty,
        "variable_OM": annual_H2 * VARIABLE_OM,
        "water": annual_H2 * WATER_STOICH * WATER_PRICE,
    }
    total = sum(items.values())
    ess_annual = (capex_ess * CRF()
                  + (capex_ess * FIXED_OM_FRAC if FIXED_OM_INCLUDES_ESS else 0.0) + batt)
    return {"LCOH": total / annual_H2, "annual_cost": total, "annual_H2": annual_H2,
            "items_usd": items,
            "items_usd_per_kg": {k: v / annual_H2 for k, v in items.items()},
            "capex_stack": capex_stack, "capex_bop": capex_bop, "capex_ess": capex_ess,
            "ess_annual_usd": ess_annual,
            "ess_share_pct": ess_annual / total * 100 if total > 0 else np.nan,
            "op_hours": op_hours}

def lcoh_compare(res: DispatchResult, stack: FixedStack, capex_ess: float = 0.0,
                 elec_price: float | None = None,
                 penalty_mult: float | None = None) -> dict:
    price = ELEC_PRICE if elec_price is None else elec_price
    elec = (res.E_used if ELEC_ACCOUNTING == "consumed" else res.E_paid) * price
    mult = ELEC_PENALTY_MULT if penalty_mult is None else penalty_mult
    penalty = price * (mult - 1.0) * res.E_unabsorbed
    out = _lcoh_core(annual_H2=res.H2_total, op_hours=res.op_hours,
                     capex_stack=stack.capex_stack, capex_bop=stack.capex_bop,
                     capex_ess=capex_ess, elec_cost=elec, curtail_penalty=penalty)
    out["accounting"] = ELEC_ACCOUNTING
    out["basis"] = "compare (배정 kWh)"
    out["penalty_mult"] = mult
    out["E_unabsorbed"] = res.E_unabsorbed
    return out

def lcoh_internal(res: DispatchResult, stack: FixedStack,
                  capex_ess: float = 0.0, elec_price: float | None = None,
                  penalty_mult: float | None = None) -> float:
    price = ELEC_PRICE if elec_price is None else elec_price
    mult = ELEC_PENALTY_MULT if penalty_mult is None else penalty_mult
    penalty = price * (mult - 1.0) * res.E_unabsorbed
    return float(_lcoh_core(annual_H2=res.H2_total, op_hours=res.op_hours,
                            capex_stack=stack.capex_stack, capex_bop=stack.capex_bop,
                            capex_ess=capex_ess,
                            elec_cost=res.E_used * price,
                            curtail_penalty=penalty)["LCOH"])

def lcoh_dcf(res: DispatchResult, stack: FixedStack, capex_ess: float = 0.0,
             elec_price: float | None = None) -> dict:
    price = ELEC_PRICE if elec_price is None else elec_price
    elec = (res.E_used if ELEC_ACCOUNTING == "consumed" else res.E_paid) * price
    capex_total = stack.capex_stack + stack.capex_bop + capex_ess
    om_base = capex_total if FIXED_OM_INCLUDES_ESS else (stack.capex_stack + stack.capex_bop)
    pv_cost, pv_h2, age_h, rows = capex_total, 0.0, 0.0, []
    for year in range(1, LIFETIME_YR + 1):
        repl = stack.capex_stack * REPL_STACK_FRAC if age_h >= STACK_LIFETIME_H else 0.0
        if repl:
            age_h = 0.0
        opex = (elec + om_base * FIXED_OM_FRAC + res.H2_total * VARIABLE_OM
                + res.H2_total * WATER_STOICH * WATER_PRICE + ANNUAL_BATT_REPL_USD + repl)
        dfac = (1.0 + INTEREST_RATE) ** year
        pv_cost += opex / dfac
        pv_h2 += res.H2_total / dfac
        rows.append({"year": year, "opex": opex, "replacement": repl,
                     "discounted_opex": opex / dfac})
        age_h += res.op_hours
    return {"LCOH_dcf": pv_cost / pv_h2 if pv_h2 > 0 else float("inf"),
            "cashflow": pd.DataFrame(rows)}

def report_lcoh(name: str, out: dict, dcf: dict | None = None) -> None:
    items = out["items_usd_per_kg"]
    print(f"\n[{name}] LCOH={out['LCOH']:.4f} USD/kg-H2 | "
          f"H2={out['annual_H2']:,.0f} kg/yr | accounting={out.get('accounting', ELEC_ACCOUNTING)}")
    print(f"  CAPEX stack/BOP/ESS = {out['capex_stack']:,.0f} / "
          f"{out['capex_bop']:,.0f} / {out['capex_ess']:,.0f} USD, "
          f"ESS annual share={out['ess_share_pct']:.2f} %")
    print("  USD/kg: " + ", ".join(f"{k}={v:.3f}" for k, v in items.items()))
    if dcf is not None:
        print(f"  DCF check={dcf['LCOH_dcf']:.4f} USD/kg-H2")


# ###################################################################################
# ###################################################################################
##
##   PART 10 — 그래프 공용 유틸                        [3케이스 공통]
##
# ###################################################################################
# ###################################################################################

CASE_COLORS = {"Case 1": "#d1495b", "Case 2": "#1f6f8b", "Case 3": "#2a9d8f"}
LEDGER_COLORS = {"curtail": "#e76f51", "idle": "#9c6644", "rte": "#e9c46a",
                 "bop": "#a8b8c8", "stack": "#2a9d8f"}
LEDGER_LABELS = {"curtail": "curtailment", "idle": "idle (below P_min)",
                 "rte": "ESS round-trip loss", "bop": "BOP (overhead)",
                 "stack": "stack (productive)"}

def setup_plots() -> None:
    if matplotlib is None:
        return
    matplotlib.rcParams.update({
        "figure.dpi": FIG_DPI, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.frameon": False})
    for cand in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"):
        try:
            matplotlib.font_manager.findfont(cand, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = cand
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

def _finish(fig, name: str) -> None:
    import os
    d = _ensure_fig_dir()
    if d:
        fig.savefig(os.path.join(d, f"{name}.png"), bbox_inches="tight", dpi=FIG_DPI)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

def duration_curve(P: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(P, dtype=float))[::-1]

def operating_heatmap_data(running: np.ndarray, n_days: int = 365) -> np.ndarray:
    arr = np.asarray(running, dtype=float)
    n = n_days * 24
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr[:n].reshape(n_days, 24).T

# %%

# ###################################################################################
# ###################################################################################
##
##   PART 11 — CASE 1 : 직결 (Direct coupling)
##
# ###################################################################################
# ###################################################################################

C1_NAME = "Case 1"

def dispatch_case1(P_in: np.ndarray, A_tot: float, j_min: float, j_max: float,
                   dt: float = 1.0) -> DispatchResult:
    P = np.asarray(P_in, dtype=float)
    P_min = float(P_of_j(j_min, A_tot)[0])
    P_max = float(P_of_j(j_max, A_tot)[0])
    inv = get_inverter(A_tot, j_min, j_max)     # inv = P -> j 계산한 거 찾기

    m_above = P > P_max
    m_idle = (P > 0.0) & (P < P_min)
    m_night = P <= 0.0
    m_in = ~(m_above | m_idle | m_night)

    j = np.zeros_like(P)
    j[m_above] = j_max
    if np.any(m_in):
        j[m_in] = inv(P[m_in])
    on = j > 0.0

    h2_rate = np.zeros_like(P)                      # [kg/h]
    h2_rate[on] = h2_area(j[on]) * A_tot
    H2_series = h2_rate * dt                        # [kg]  rate/amt 분리 (Δt 대비)

    P_stack = np.zeros_like(P)
    P_stack[on] = j[on] * V(j[on]) * A_tot / 1000.0
    P_bop = BOP_EFFICIENCY * h2_rate
    P_used = P_stack + P_bop

    E_paid = float(P.sum() * dt)                            # 배정 전량 지불 (R7)
    E_curtail = float((P[m_above] - P_max).sum() * dt)      # 초과분만 버림
    E_idle = float(P[m_idle].sum() * dt)                    # 하한 미만 -> 전량 손실
    E_stack = float(P_stack.sum() * dt)
    E_bop = float(P_bop.sum() * dt)

    cur = np.zeros_like(P)
    cur[m_above] = P[m_above] - P_max

    return DispatchResult(
        case=C1_NAME, dt=dt, P_in=P, j=j, P_used=P_used, H2_series=H2_series,
        curtail_series=cur,
        E_paid=E_paid, E_curtail=E_curtail, E_idle=E_idle, E_rte=0.0,
        E_bop=E_bop, E_stack=E_stack, E_direct=E_stack + E_bop, E_via_ess=0.0,
        H2_total=float(H2_series.sum()), op_hours=float(on.sum() * dt),
        hours_curtail=float(m_above.sum() * dt), hours_idle=float(m_idle.sum() * dt),
        hours_night=float(m_night.sum() * dt),
        meta={"P_min": P_min, "P_max": P_max, "j_min": j_min, "j_max": j_max})

def run_case1(profile=None, fixed_stack=None, j_star=None, verbose=True) -> dict:
    profile = profile or get_profile()
    fixed_stack = fixed_stack or build_fixed_stack(profile, verbose=verbose)
    win = operating_window(fixed_stack.A_tot)

    if verbose:
        print(f"\n[PART 11] {C1_NAME} — direct coupling")
        print(win.describe())
        get_inverter(fixed_stack.A_tot, win.j_min, win.j_max).validate(verbose=False)

    res = dispatch_case1(profile.P_in, fixed_stack.A_tot, win.j_min, win.j_max,
                         dt=profile.dt)
    lc = lcoh_compare(res, fixed_stack, 0.0)          # ESS = 0 : Case 1의 유일한 장점
    dcf = lcoh_dcf(res, fixed_stack, 0.0)

    if verbose:
        res.report(h2_area_at_rated=float(h2_area(win.j_max)[0]),
                   h2_area_at_jstar=float(h2_area(j_star)[0]) if j_star else None,
                   A_tot=fixed_stack.A_tot)
        _report_case1_extra(res, win)
        report_lcoh(C1_NAME, lc, dcf)

    return {"result": res, "lcoh": lc, "lcoh_dcf": dcf, "window": win,
            "stack": fixed_stack, "profile": profile, "capex_ess": 0.0}

def _report_case1_extra(res: DispatchResult, win: OperatingWindow) -> None:
    on = res.j > 0
    print("\n[Case 1 diagnostics]")
    print(f"  curtail={res.hours_curtail:,.0f} h ({res.E_curtail / res.E_paid * 100:.2f} %), "
          f"idle={res.hours_idle:,.0f} h ({res.E_idle / res.E_paid * 100:.2f} %), "
          f"night={res.hours_night:,.0f} h")
    if on.any():
        w = res.H2_series[on]
        jw = float((res.j[on] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        print(f"  j[min/med/mean/max]={res.j[on].min():.4f}/"
              f"{np.median(res.j[on]):.4f}/{res.j[on].mean():.4f}/"
              f"{res.j[on].max():.4f} A/cm2 | H2-weighted j={jw:.4f}")
    print(f"  SEC_stack_avg={res.SEC_stack_avg():.2f}, "
          f"rated SEC_stack={float(SEC_stack(win.j_max)[0]):.2f} kWh/kg")

def stack_rating_sweep_case1(profile=None, factors=None, base_nameplate=None,
                             verbose=True) -> pd.DataFrame:
    profile = profile or get_profile()
    factors = factors or CASE1["rating_sweep_factors"]
    if base_nameplate is None:
        base_nameplate = build_fixed_stack(profile, verbose=False).P_nameplate

    rows = []
    for f in factors:
        st = size_fixed_stack(base_nameplate * f)
        w = operating_window(st.A_tot)
        r = dispatch_case1(profile.P_in, st.A_tot, w.j_min, w.j_max, dt=profile.dt)
        lc = lcoh_compare(r, st, 0.0)
        rows.append({"nameplate [kW]": st.P_nameplate, "N_cells": st.N_cells,
                     "curtail [%]": r.E_curtail / r.E_paid * 100,
                     "idle [%]": r.E_idle / r.E_paid * 100,
                     "op_hours [h]": r.op_hours, "annual_H2 [kg]": r.H2_total,
                     "SEC_eff [kWh/kg]": r.SEC_eff(),
                     "CAPEX_fix [kUSD]": st.capex_fix / 1e3, "LCOH [$/kg]": lc["LCOH"]})
    df = pd.DataFrame(rows)
    if verbose:
        k = int(df["LCOH [$/kg]"].idxmin())
        print(f"\n[Case 1 rating sweep] best={df.loc[k, 'nameplate [kW]']:,.0f} kW, "
              f"LCOH={df.loc[k, 'LCOH [$/kg]']:.3f}, "
              f"curtail={df.loc[k, 'curtail [%]']:.2f} %, "
              f"inside={'YES' if 0 < k < len(df) - 1 else 'NO'}")
    return df

def plot_case1(res, win, profile, fixed_stack, rating_df=None) -> None:
    if plt is None:
        return
    setup_plots()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    dur = duration_curve(res.P_in)
    x = np.arange(len(dur))
    used = np.where(dur >= win.P_min, np.minimum(dur, win.P_max), 0.0)
    ax.fill_between(x, 0, used, color="#2a9d8f", alpha=0.35, label="used by stack+BOP")
    ax.fill_between(x, np.minimum(dur, win.P_max), dur, color="#e76f51", alpha=0.6,
                    label="curtailed")
    idle = np.where((dur > 0.0) & (dur < win.P_min), dur, 0.0)
    ax.fill_between(x, 0, idle, color="#9c6644", alpha=0.9,
                    label="idle loss (0 < P < P_min only)")
    ax.plot(x, dur, color="k", lw=1.2, label="P_in (sorted)")
    ax.axhline(win.P_max, color="#e76f51", ls="--", lw=1.2,
               label=f"P_max = {win.P_max:,.0f} kW")
    ax.axhline(win.P_min, color="#9c6644", ls=":", lw=1.4,
               label=f"P_min = {win.P_min:,.0f} kW")
    n_night = int((dur <= 0).sum())
    ax.axvspan(len(dur) - n_night, len(dur), color="#dddddd", alpha=0.5, zorder=0)
    ax.text(len(dur) - n_night / 2, win.P_max * 0.55,
            f"night / no generation\n{n_night:,} h  (NOT a loss)",
            ha="center", fontsize=8, color="#555")
    ax.set_xlabel("Hours of year (sorted, descending)")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim(0, len(dur))
    ax.set_title("Case 1 (G1) — Power duration curve with operating window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case1_G1_duration_curve")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    on = res.j > 0
    ax.hist(res.j[on], bins=60, color="#4a7c8c", alpha=0.75)
    ax.set_xlabel("Operating current density j [A/cm$^2$]")
    ax.set_ylabel("Hours per year")
    ax.set_title("Case 1 (G2) — Operating j distribution vs SEC(j)")
    ax2 = ax.twinx()
    jg = np.linspace(win.j_min, win.j_max, 200)
    ax2.plot(jg, SEC_stack(jg), color="#e76f51", lw=2, label="SEC_stack(j)")
    ax2.axhline(res.SEC_eff(), color="k", ls="--", lw=1.2,
                label=f"SEC_eff = {res.SEC_eff():.1f}")
    ax2.axhline(res.SEC_stack_avg(), color="#2a9d8f", ls=":", lw=1.6,
                label=f"SEC_stack_avg = {res.SEC_stack_avg():.1f}")
    ax2.set_ylabel("SEC [kWh/kg-H$_2$]")
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _finish(fig, "case1_G2_j_histogram_SEC")

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    bottom = 0.0
    for k in ("stack", "bop", "idle", "curtail"):
        v = res.ledger[k] / 1e3
        if v <= 0:
            continue
        ax.bar(["E_paid"], [v], bottom=[bottom], color=LEDGER_COLORS[k],
               label=f"{LEDGER_LABELS[k]}  ({v / (res.E_paid / 1e3) * 100:.1f}%)")
        bottom += v
    ax.set_ylabel("Energy [MWh/yr]")
    ax.set_title("Case 1 (G3) — Where the paid energy goes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case1_G3_loss_ledger")

    days = profile.representative_days()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (label, idx) in zip(axes, days.items()):
        if len(idx) == 0:
            continue
        h = np.arange(len(idx))
        ax.fill_between(h, 0, res.P_in[idx], color="#c8d8dd", label="P_in (paid)")
        ax.fill_between(h, 0, res.P_used[idx], color="#2a9d8f", alpha=0.85,
                        label="P_used (stack+BOP)")
        ax.axhline(win.P_max, color="#e76f51", ls="--", lw=1)
        ax.axhline(win.P_min, color="#9c6644", ls=":", lw=1.2)
        ax.set_xlabel("Hour of day")
        ax.set_title(f"Case 1 (G4) — representative day: {label}")
        ax2 = ax.twinx()
        ax2.plot(h, res.j[idx], color="#e76f51", lw=1.8)
        ax2.set_ylim(0, win.j_max * 1.15)
        ax2.grid(False)
        if ax is axes[-1]:
            ax2.set_ylabel("j [A/cm$^2$]")
    axes[0].set_ylabel("Power [kW]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _finish(fig, "case1_G4_representative_days")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    Pg = np.linspace(0, res.P_in.max() * 1.02, 600)
    h2 = np.zeros_like(Pg)
    inv = get_inverter(fixed_stack.A_tot, win.j_min, win.j_max)
    m_in = (Pg >= win.P_min) & (Pg <= win.P_max)
    h2[m_in] = h2_area(inv(Pg[m_in])) * fixed_stack.A_tot
    h2[Pg > win.P_max] = float(h2_area(win.j_max)[0]) * fixed_stack.A_tot
    ax.plot(Pg, h2, color="#1f6f8b", lw=2.2)
    ax.axvspan(0, win.P_min, color="#9c6644", alpha=0.18)
    ax.text(win.P_min * 0.5, h2.max() * 0.5, "dead\nzone", ha="center", fontsize=9)
    ax.axvline(win.P_max, color="#e76f51", ls="--", lw=1.2)
    ax.text(win.P_max, h2.max() * 0.15, "  saturation\n  (curtail)", fontsize=9)
    ax.set_xlabel("Instantaneous input power P_in [kW]")
    ax.set_ylabel("H$_2$ rate [kg/h]")
    ax.set_title("Case 1 (G5) — Transfer function P_in -> H$_2$ (concave)")
    fig.tight_layout()
    _finish(fig, "case1_G5_transfer_function")

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    s = pd.Series(res.H2_series).groupby(profile.month).sum()
    ax.bar(s.index, s.values / 1000.0, color="#d1495b", width=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("H$_2$ [t]")
    ax.set_title("Case 1 (G7) — Monthly hydrogen production")
    fig.tight_layout()
    _finish(fig, "case1_G7_monthly_H2")

    if rating_df is not None and len(rating_df):
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.plot(rating_df["nameplate [kW]"], rating_df["LCOH [$/kg]"], "o-",
                color="#1f6f8b", lw=2, label="LCOH")
        ax.set_xlabel("Stack nameplate power [kW]")
        ax.set_ylabel("LCOH [USD/kg-H$_2$]")
        ax.set_title("Case 1 (G8) — Stack rating sweep")
        ax2 = ax.twinx()
        ax2.plot(rating_df["nameplate [kW]"], rating_df["curtail [%]"], "s--",
                 color="#e76f51", label="curtail %")
        ax2.plot(rating_df["nameplate [kW]"], rating_df["idle [%]"], "^:",
                 color="#9c6644", label="idle %")
        ax2.set_ylabel("Loss share [%]")
        ax2.grid(False)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
        fig.tight_layout()
        _finish(fig, "case1_G8_rating_sweep")


# ###################################################################################
# ###################################################################################
##
##   PART 12 — CASE 2 : ESS 완충 (ESS-buffered, 항상 j*)
##
# ###################################################################################
# ###################################################################################

C2_NAME = "Case 2"

def c2_operating_point(j: float, A_tot: float) -> dict:
    P_stack, P_bop = split_power(j, A_tot)
    return {"j": float(j), "P_star": P_stack + P_bop, "P_stack": P_stack,
            "P_bop": P_bop, "h2_rate": float(h2_area(j)[0]) * A_tot}

def c2_charge_rating(profile, P_star: float) -> float:
    if _use_charge_pctl():
        return max(profile.pctl(CHARGE_RATING_PCTL), P_star)
    return profile.peak

def _c2_restart_level(E_rated, P_star, eta, dt, soc_floor, soc_restart, restart_hours):
    if restart_hours is not None:
        return min(E_rated * soc_floor + restart_hours * P_star / eta * dt,
                   E_rated * CASE2["soc_max"])
    return E_rated * soc_restart

def dispatch_case2(P_in, op, A_tot, E_rated, P_rated_chg,
                   soc_max=None, soc_floor=None, soc_restart=None,
                   restart_hours=None, eta_RTE=None, soc0=None, dt=1.0,
                   keep_series=True) -> DispatchResult:
    soc_max = CASE2["soc_max"] if soc_max is None else soc_max
    soc_floor = CASE2["soc_floor"] if soc_floor is None else soc_floor
    soc_restart = CASE2["soc_restart"] if soc_restart is None else soc_restart
    eta = ETA_RTE if eta_RTE is None else eta_RTE

    P = np.asarray(P_in, dtype=float)
    n = len(P)
    P_star, P_stack_star, P_bop_star = op["P_star"], op["P_stack"], op["P_bop"]
    h2_rate = op["h2_rate"]

    E_hi, E_lo = E_rated * soc_max, E_rated * soc_floor
    E_rs = _c2_restart_level(E_rated, P_star, eta, dt, soc_floor, soc_restart,
                             restart_hours)
    e_draw = P_star * dt / eta                       # 방전 인출량 (왕복손실 포함)
    e_chg_lim = P_rated_chg * dt                     # e_chg_lim -> P_rated_chg -> PCS 정격 (p95, 99, ...)

    E = E_lo if soc0 is None else float(np.clip(soc0, 0.0, soc_max)) * E_rated
    running = (E >= E_rs) and ((E - E_lo) >= e_draw)

    E_paid = E_curtail = E_rte = E_bop = E_stack = 0.0
    H2 = op_hours = eq_cycles = 0.0
    restarts = 0

    soc = np.empty(n) if keep_series else None                      # soc
    run = np.zeros(n, dtype=bool) if keep_series else None          # run
    chg = np.zeros(n) if keep_series else None                      # charge
    dis = np.zeros(n) if keep_series else None                      # discharge
    cur = np.zeros(n) if keep_series else None                      # curtail
    h2s = np.zeros(n) if keep_series else None                      # H2
    pus = np.zeros(n) if keep_series else None                      # P_used

    for i in range(n):
        e_in = P[i] * dt
        E_paid += e_in                                       

        e_chg = e_in
        e_empty = E_hi - E
        if e_chg > e_empty:
            e_chg = e_empty if e_empty > 0.0 else 0.0
        if e_chg > e_chg_lim:
            e_chg = e_chg_lim
        E += e_chg
        E_curtail += e_in - e_chg

        if running:
            if (E - E_lo) < e_draw:
                running = False
        else:
            if E >= E_rs and (E - E_lo) >= e_draw:
                running = True
                restarts += 1

        if running:
            E -= e_draw
            E_rte += P_star * dt * (1.0 / eta - 1.0)
            H2 += h2_rate * dt                               
            E_stack += P_stack_star * dt
            E_bop += P_bop_star * dt
            op_hours += dt
            eq_cycles += e_draw / E_rated if E_rated > 0 else 0.0
            if keep_series:
                dis[i] = e_draw / dt
                h2s[i] = h2_rate * dt
                pus[i] = P_star

        if keep_series:
            soc[i] = E / E_rated if E_rated > 0 else 0.0
            run[i] = running
            chg[i] = e_chg / dt
            cur[i] = (e_in - e_chg) / dt

    jj = np.where(run, op["j"], 0.0) if keep_series else np.zeros(0)
    return DispatchResult(
        case=C2_NAME, dt=dt, P_in=P, j=jj,
        P_used=pus if keep_series else np.zeros(0),
        H2_series=h2s if keep_series else np.zeros(0),
        soc=soc, running=run, charge=chg, discharge=dis, curtail_series=cur,
        E_paid=E_paid, E_curtail=E_curtail, E_idle=0.0, E_rte=E_rte,
        E_bop=E_bop, E_stack=E_stack,
        E_direct=0.0, E_via_ess=E_stack + E_bop,        # 전량 ESS 경유 (직렬)
        H2_total=H2, op_hours=op_hours,
        hours_curtail=float((cur > 1e-9).sum() * dt) if keep_series else 0.0,
        hours_idle=0.0, hours_night=float((P <= 0.0).sum() * dt),
        restarts=restarts, eq_cycles=eq_cycles, E_end=E,
        E_rated=E_rated, P_rated_chg=P_rated_chg, P_rated_dis=P_star,
        j_star=op["j"], P_star=P_star,
        meta={"E_hi": E_hi, "E_lo": E_lo, "E_restart": E_rs, "e_draw": e_draw,
              "eta_RTE": eta,
              "min_run_duration_h": (E_rs - E_lo) / e_draw * dt if e_draw > 0 else np.inf})

def c2_run_warmstart(P_in, op, A_tot, E_rated, P_rc, dt=1.0, **kw):
    if not WARM_START or E_rated <= 0:
        return dispatch_case2(P_in, op, A_tot, E_rated, P_rc, dt=dt, **kw)
    r1 = dispatch_case2(P_in, op, A_tot, E_rated, P_rc, dt=dt,
                        keep_series=False, **kw)
    return dispatch_case2(P_in, op, A_tot, E_rated, P_rc, dt=dt,
                        soc0=r1.E_end / E_rated, **kw)

def find_j_star_case2(profile, fixed_stack, cap_hours, n_j=None, verbose=True):
    n_j = n_j or _n_j_grid("case2")
    A_tot = fixed_stack.A_tot
    rows, best = [], None
    for j in np.linspace(J_MIN, J_MAX, n_j):
        op = c2_operating_point(j, A_tot)
        E_rated = cap_hours * op["P_star"]
        P_rc = c2_charge_rating(profile, op["P_star"])
        r = c2_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc,
                     dt=profile.dt,
                     restart_hours=CASE2["restart_hours_of_Pstar"])
        if r.H2_total <= 0:
            continue
        cx = ess_capex(E_rated, max(P_rc, op["P_star"]))
        li = lcoh_internal(r, fixed_stack, cx)
        rows.append({"j": j, "P_star": op["P_star"], "E_rated": E_rated,
                     "op_hours": r.op_hours,
                     "curtail_%": r.E_curtail / r.E_paid * 100,
                     "annual_H2_t": r.H2_total / 1000, "LCOH_internal": li,
                     "restarts": r.restarts})
        if best is None or li < best[2]:
            best = (j, op["P_star"], li, r)

    df = pd.DataFrame(rows)
    if verbose and len(df):
        k = int(df["LCOH_internal"].idxmin())
        print(f"\n[Case 2 j* search] cap={cap_hours:g} h, grid={n_j}, "
              f"j*={df.loc[k, 'j']:.4f} A/cm2, "
              f"P*={df.loc[k, 'P_star']:,.1f} kW, "
              f"LCOH_int={df.loc[k, 'LCOH_internal']:.3f}, "
              f"inside={'YES' if 0 < k < len(df) - 1 else 'NO'}")
    return best, df

def size_ess_case2(profile, fixed_stack, j, cap_hours_grid=None, verbose=True):
    cap_hours_grid = cap_hours_grid or _cap_grid("case2")
    A_tot = fixed_stack.A_tot
    op = c2_operating_point(j, A_tot)
    P_rc = c2_charge_rating(profile, op["P_star"])
    rippl = rippl_capacity(profile.P_in, op["P_star"], ETA_RTE, profile.dt)

    rows, best = [], None
    for h in cap_hours_grid:
        E_rated = h * op["P_star"]
        r = c2_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc,
                     dt=profile.dt,
                     restart_hours=CASE2["restart_hours_of_Pstar"])
        cx = ess_capex(E_rated, max(P_rc, op["P_star"]))
        li = lcoh_internal(r, fixed_stack, cx)
        lc = lcoh_compare(r, fixed_stack, cx)
        rows.append({"hours_of_P*": h, "E_rated [kWh]": E_rated, "op_hours": r.op_hours,
                     "annual_H2 [t]": r.H2_total / 1000,
                     "curtail [%]": r.E_curtail / r.E_paid * 100,
                     "RTE loss [%]": r.E_rte / r.E_paid * 100,
                     "restarts": r.restarts, "eq_cycles": r.eq_cycles,
                     "CAPEX_ESS [kUSD]": cx / 1e3,
                     "LCOH_internal": li, "LCOH_compare": lc["LCOH"]})
        if best is None or li < best[1]:
            best = (h, li, E_rated, r)

    df = pd.DataFrame(rows)
    if verbose:
        k = int(df["LCOH_internal"].idxmin())
        print(f"\n[Case 2 ESS sweep] j={j:.4f}, P*={op['P_star']:,.1f} kW, "
              f"E*={df.loc[k, 'E_rated [kWh]']:,.0f} kWh "
              f"({df.loc[k, 'hours_of_P*']:g} h), "
              f"LCOH_int={df.loc[k, 'LCOH_internal']:.3f}, "
              f"inside={'YES' if 0 < k < len(df) - 1 else 'NO'}, "
              f"Rippl={rippl / 1e3:,.1f} MWh")
    return best, df, rippl

def find_operating_point_case2(profile, fixed_stack, n_round=2, verbose=True) -> dict:
    j_cur = 0.9 * J_MAX
    cap_h = jdf = capdf = rippl = None
    for rd in range(n_round):
        if verbose:
            print(f"\n[Case 2 optimize] round {rd + 1}, anchor j={j_cur:.4f}")
        (cap_h, _, _, _), capdf, rippl = size_ess_case2(profile, fixed_stack, j_cur,
                                                        verbose=verbose)
        best, jdf = find_j_star_case2(profile, fixed_stack, cap_h, verbose=verbose)
        if best is None:
            raise RuntimeError("Case 2: 어떤 j에서도 운전이 성립하지 않았습니다.")
        if abs(best[0] - j_cur) / max(j_cur, 1e-9) < 1e-3:
            j_cur = best[0]
            break
        j_cur = best[0]
    return {"j_star": j_cur, "cap_hours": cap_h, "j_sweep": jdf,
            "cap_sweep": capdf, "rippl": rippl}

def hysteresis_sweep_case2(profile, fixed_stack, j, E_rated, restart_grid=None,
                           verbose=True) -> pd.DataFrame:
    restart_grid = restart_grid or CASE2["restart_sweep"]
    A_tot = fixed_stack.A_tot
    op = c2_operating_point(j, A_tot)
    P_rc = c2_charge_rating(profile, op["P_star"])
    rows = []
    for sr in restart_grid:
        r = c2_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc,
                             dt=profile.dt, soc_restart=sr, restart_hours=None)
        rows.append({"soc_restart": sr, "restarts": r.restarts, "op_hours": r.op_hours,
                     "annual_H2 [t]": r.H2_total / 1000,
                     "min_run_h": r.meta["min_run_duration_h"],
                     "eq_cycles": r.eq_cycles})
    df = pd.DataFrame(rows)
    if verbose:
        ok = int((df['min_run_h'] >= 1.0).sum())
        print(f"\n[Case 2 hysteresis sweep] rows={len(df)}, min_run>=1h: {ok}/{len(df)}")
    return df

def run_case2(profile=None, fixed_stack=None, j_star=None, cap_hours=None,
              optimize=True, verbose=True) -> dict:
    profile = profile or get_profile()
    fixed_stack = fixed_stack or build_fixed_stack(profile, verbose=verbose)
    A_tot = fixed_stack.A_tot

    if verbose:
        print(f"\n[PART 12] {C2_NAME} — ESS buffered, fixed j*")

    opt = None
    fixed_mode = j_is_fixed() and j_star is None
    if fixed_mode:
        j_star = fixed_j("case2", verbose=verbose)
        if verbose:
            print(f"\n[Case 2 fixed-j] j={j_star:.4f} A/cm2; skip j* search")
    if optimize and not fixed_mode and (j_star is None or cap_hours is None):
        opt = find_operating_point_case2(profile, fixed_stack, verbose=verbose)
        j_star = j_star if j_star is not None else opt["j_star"]
        cap_hours = cap_hours if cap_hours is not None else opt["cap_hours"]
    j_star = 0.9 * J_MAX if j_star is None else j_star

    (cap_hours, _, _, _), cap_df, rippl = size_ess_case2(profile, fixed_stack, j_star,
                                                          verbose=verbose)
    op = c2_operating_point(j_star, A_tot)
    E_rated = cap_hours * op["P_star"]
    P_rc = c2_charge_rating(profile, op["P_star"])
    P_rd = op["P_star"]  # Case 2 방전정격은 정의상 P_star

    res = c2_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc,
                        dt=profile.dt,
                        restart_hours=CASE2["restart_hours_of_Pstar"])
    cx = ess_capex(E_rated, max(P_rc, P_rd))
    lc = lcoh_compare(res, fixed_stack, cx)
    dcf = lcoh_dcf(res, fixed_stack, cx)

    if verbose:
        mrd = res.meta["min_run_duration_h"]
        print("\n[Case 2 operating point]")
        print(f"  j*={j_star:.4f} A/cm2, V={float(V(j_star)[0]):.4f} V, "
              f"SEC_stack={float(SEC_stack(j_star)[0]):.2f} kWh/kg")
        print(f"  P*={op['P_star']:,.1f} kW (stack={op['P_stack']:,.1f}, "
              f"BOP={op['P_bop']:,.1f}), P*/mean={op['P_star'] / profile.mean:.2f}")
        print(f"  E*={E_rated:,.0f} kWh ({E_rated / 1e3:,.2f} MWh, {cap_hours:g} h), "
              f"Pchg/Pdis={P_rc:,.1f}/{P_rd:,.1f} kW, CAPEX_ESS={cx:,.0f} USD")
        print(f"  SOC restart={res.meta['E_restart'] / E_rated:.3f}, "
              f"min_run={mrd:.2f} h, Rippl={rippl / 1e3:,.1f} MWh")
        res.report(h2_area_at_rated=float(h2_area(J_MAX)[0]),
                   h2_area_at_jstar=float(h2_area(j_star)[0]), A_tot=A_tot)
        report_lcoh(C2_NAME, lc, dcf)

    return {"result": res, "lcoh": lc, "lcoh_dcf": dcf, "op": op, "j_star": j_star,
            "cap_hours": cap_hours, "E_rated": E_rated, "capex_ess": cx,
            "cap_sweep": cap_df, "rippl": rippl, "opt": opt,
            "stack": fixed_stack, "profile": profile}

def plot_case2(out: dict, hyst_df=None) -> None:
    if plt is None:
        return
    setup_plots()
    res, profile, op, E_rated = out["result"], out["profile"], out["op"], out["E_rated"]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    t = np.arange(len(res.soc)) / 24.0
    ax.plot(t, res.soc * 100, lw=0.6, color="#1f6f8b")
    ax.axhline(CASE2["soc_max"] * 100, color="#e76f51", ls="--", lw=1,
               label="SOC_max (full -> curtail)")
    ax.axhline(res.meta["E_restart"] / E_rated * 100, color="#2a9d8f", ls="-.", lw=1,
               label="SOC_restart")
    ax.axhline(CASE2["soc_floor"] * 100, color="#9c6644", ls=":", lw=1.4,
               label="SOC_floor (stop)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("SOC [%]")
    ax.set_xlim(0, 365)
    ax.set_title("Case 2 (G1) — Annual SOC: full band = curtail, bottom = shutdown")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    _finish(fig, "case2_G1_soc_timeseries")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    dur = duration_curve(res.P_in)
    x = np.arange(len(dur))
    ax.plot(x, dur, color="k", lw=1.2, label="P_in (sorted)")
    ax.fill_between(x, op["P_star"], dur, where=dur > op["P_star"], color="#e9c46a",
                    alpha=0.6, label="above P* -> charge")
    ax.fill_between(x, dur, op["P_star"], where=dur < op["P_star"], color="#1f6f8b",
                    alpha=0.35, label="below P* -> discharge e_draw")
    ax.axhline(op["P_star"], color="#e76f51", ls="--", lw=1.4,
               label=f"P_star = {op['P_star']:,.0f} kW")
    ax.axhline(profile.mean, color="#2a9d8f", ls=":", lw=1.4,
               label=f"RE mean = {profile.mean:,.0f} kW")
    ax.set_xlabel("Hours of year (sorted, descending)")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim(0, len(dur))
    ax.set_title("Case 2 (G3) — Duration curve vs P_star: the ESS sizing argument")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case2_G3_duration_vs_Pstar")

    df = out["cap_sweep"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(df["hours_of_P*"], df["annual_H2 [t]"], "o-", color="#2a9d8f", lw=2,
            label="annual H$_2$")
    ax.set_xlabel("ESS energy rating [hours of P_star]")
    ax.set_ylabel("Annual H$_2$ [t]")
    ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(df["hours_of_P*"], df["LCOH_compare"], "s--", color="#1f6f8b",
             label="LCOH (compare)")
    ax2.plot(df["hours_of_P*"], df["curtail [%]"], "^:", color="#e76f51",
             label="curtail %")
    ax2.set_ylabel("LCOH [USD/kg] / curtail [%]")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.set_title("Case 2 (G5) — ESS capacity sweep: diminishing returns")
    fig.tight_layout()
    _finish(fig, "case2_G5_capacity_sweep")

    if hyst_df is not None and len(hyst_df):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(hyst_df["soc_restart"] * 100, hyst_df["restarts"], "o-",
                color="#e76f51", lw=2, label="restarts / yr")
        ax.set_xlabel("SOC_restart [%]")
        ax.set_ylabel("Electrolyser restarts per year")
        ax2 = ax.twinx()
        ax2.plot(hyst_df["soc_restart"] * 100, hyst_df["annual_H2 [t]"], "s--",
                 color="#2a9d8f", lw=2, label="annual H$_2$")
        ax2.set_ylabel("Annual H$_2$ [t]")
        ax2.grid(False)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
        ax.set_title("Case 2 (G6) — Hysteresis sweep: pick the knee")
        fig.tight_layout()
        _finish(fig, "case2_G6_hysteresis_sweep")

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.imshow(operating_heatmap_data(res.running), aspect="auto", origin="lower",
              cmap="YlGnBu", extent=[0, 365, 0, 24], vmin=0, vmax=1)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_title("Case 2 (G7) — Operating state map: dark = running "
                 "(note night-time operation)")
    fig.tight_layout()
    _finish(fig, "case2_G7_operating_heatmap")

    opt = out.get("opt")
    if opt and opt.get("j_sweep") is not None and len(opt["j_sweep"]):
        d = opt["j_sweep"]
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.plot(d["j"], d["LCOH_internal"], "o-", color="#1f6f8b", lw=2)
        k = int(d["LCOH_internal"].idxmin())
        ax.plot(d.loc[k, "j"], d.loc[k, "LCOH_internal"], "*", ms=16, color="#e76f51",
                label=f"j* = {d.loc[k, 'j']:.3f} A/cm$^2$")
        ax.set_xlabel("Current density j [A/cm$^2$]")
        ax.set_ylabel("Internal LCOH [USD/kg] (actual-consumption basis)")
        ax.set_title("Case 2 — j* search: dispatch nested inside the sweep")
        ax2 = ax.twinx()
        ax2.plot(d["j"], d["curtail_%"], "^:", color="#e76f51", label="curtail %")
        ax2.plot(d["j"], d["op_hours"] / 100, "v:", color="#9c6644",
                 label="op_hours / 100")
        ax2.set_ylabel("curtail [%] / op_hours[h]/100")
        ax2.grid(False)
        lines = ax.get_lines()[:2] + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
        fig.tight_layout()
        _finish(fig, "case2_jstar_Ucurve")

    days = profile.representative_days()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (label, idx) in zip(axes, days.items()):
        if len(idx) == 0:
            continue
        h = np.arange(len(idx))
        ax.fill_between(h, 0, res.P_in[idx], color="#c8d8dd", label="P_in")
        ax.plot(h, res.P_used[idx], color="#2a9d8f", lw=2, label="P_used (= P*)")
        ax.axhline(op["P_star"], color="#e76f51", ls="--", lw=1)
        ax.set_xlabel("Hour of day")
        ax.set_title(f"Case 2 — representative day: {label}")
        ax2 = ax.twinx()
        ax2.plot(h, res.soc[idx] * 100, color="#1f6f8b", lw=1.6)
        ax2.set_ylim(0, 100)
        ax2.grid(False)
        if ax is axes[-1]:
            ax2.set_ylabel("SOC [%]")
    axes[0].set_ylabel("Power [kW]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _finish(fig, "case2_representative_days")

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    bottom = 0.0
    for k in ("stack", "bop", "rte", "curtail"):
        v = res.ledger[k] / 1e3
        if v <= 0:
            continue
        ax.bar(["E_paid"], [v], bottom=[bottom], color=LEDGER_COLORS[k],
               label=f"{LEDGER_LABELS[k]}  ({v / (res.E_paid / 1e3) * 100:.1f}%)")
        bottom += v
    ax.set_ylabel("Energy [MWh/yr]")
    ax.set_title("Case 2 — loss ledger (E_idle vanishes, E_rte appears)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case2_loss_ledger")

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    s = pd.Series(res.H2_series).groupby(profile.month).sum()
    ax.bar(s.index, s.values / 1000.0, color="#1f6f8b", width=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("H$_2$ [t]")
    ax.set_title("Case 2 — Monthly H$_2$ (flat = healthy; if not, battery undersized)")
    fig.tight_layout()
    _finish(fig, "case2_monthly_H2")


# ###################################################################################
# ###################################################################################
##
##   PART 13 — CASE 3 : 하이브리드 (제안, ESS 최소화)
##
# ###################################################################################
# ###################################################################################

C3_NAME = "Case 3"


def c3_operating_point(j: float, A_tot: float) -> dict:
    """split_power: P_stack + P_bop == P_star 를 assert로 보증."""
    P_stack, P_bop = split_power(j, A_tot)
    return {"j": float(j), "P_star": P_stack + P_bop, "P_stack": P_stack,
            "P_bop": P_bop, "h2_rate": float(h2_area(j)[0]) * A_tot}


def c3_charge_rating(profile, P_star: float) -> float:
    surplus = np.clip(profile.P_in - P_star, 0.0, None)
    if not _use_charge_pctl():
        return max(float(surplus.max()), 1e-6)
    pos = surplus[surplus > 0.0]
    if pos.size == 0:
        return 1e-6
    return max(float(np.percentile(pos, CHARGE_RATING_PCTL)), 1e-6)


def _c3_restart_level(E_rated, P_star, eta, dt, restart_hours):
    if restart_hours is not None:
        return min(E_rated * CASE3["soc_floor"] + restart_hours * P_star / eta * dt,
                   E_rated * CASE3["soc_max"])
    return E_rated * CASE3["soc_restart"]


def dispatch_case3(P_in, op, A_tot, E_rated, P_rated_chg, P_rated_dis,
                   soc_max=None, soc_floor=None, soc_stop=None, soc_restart=None,
                   restart_hours=None, eta_RTE=None, soc0=None, dt=1.0,
                   keep_series=True) -> DispatchResult:
    soc_max = CASE3["soc_max"] if soc_max is None else soc_max
    soc_floor = CASE3["soc_floor"] if soc_floor is None else soc_floor
    soc_stop = CASE3["soc_stop"] if soc_stop is None else soc_stop
    soc_restart = CASE3["soc_restart"] if soc_restart is None else soc_restart
    eta = ETA_RTE if eta_RTE is None else eta_RTE

    P = np.asarray(P_in, dtype=float)
    n = len(P)
    P_star, P_stack_star, P_bop_star = op["P_star"], op["P_stack"], op["P_bop"]
    h2_rate = op["h2_rate"]

    E_hi = E_rated * soc_max
    E_lo = E_rated * soc_floor
    E_stp = E_rated * soc_stop
    E_rs = _c3_restart_level(E_rated, P_star, eta, dt, restart_hours)
    e_chg_lim = P_rated_chg * dt
    dis_cap = P_rated_dis * dt / eta

    E = E_lo if soc0 is None else float(np.clip(soc0, 0.0, soc_max)) * E_rated
    running = E >= E_rs

    E_paid = E_curtail = E_rte = E_bop = E_stack = 0.0
    E_direct = E_via = 0.0
    H2 = op_hours = eq_cycles = 0.0
    restarts = 0

    soc = np.empty(n) if keep_series else None                      # soc
    run = np.zeros(n, dtype=bool) if keep_series else None          # run
    chg = np.zeros(n) if keep_series else None                      # charge
    dis = np.zeros(n) if keep_series else None                      # discharge
    cur = np.zeros(n) if keep_series else None                      # curtail
    h2s = np.zeros(n) if keep_series else None                      # H2
    pus = np.zeros(n) if keep_series else None                      # P_used

    for i in range(n):
        p = P[i]
        E_paid += p * dt                                     # 배정 전량 지불 (R7)

        surplus = p - P_star
        if surplus < 0.0:
            surplus = 0.0
        deficit = P_star - p
        if deficit < 0.0:
            deficit = 0.0
        draw_need = deficit * dt / eta                       
        if running:
            if (E - E_stp) < draw_need or draw_need > dis_cap:
                if not (CASE3.get("direct_restart", True) and p >= P_star):
                    running = False
        else:
            can_run_direct = CASE3.get("direct_restart", True) and p >= P_star
            can_run_from_batt = (E >= E_rs and
                                 (E - E_stp) >= draw_need and
                                 draw_need <= dis_cap)

            if can_run_direct or can_run_from_batt:
                running = True
                restarts += 1

        chg_want = (surplus if running else p) * dt
        e_chg = chg_want
        e_empty = E_hi - E
        if e_chg > e_empty:
            e_chg = e_empty if e_empty > 0.0 else 0.0
        if e_chg > e_chg_lim:
            e_chg = e_chg_lim
        E += e_chg
        E_curtail += chg_want - e_chg

        if running:
            E -= draw_need
            E_rte += deficit * dt * (1.0 / eta - 1.0)         # ★ 부족분에만 (C3-2)
            H2 += h2_rate * dt                                # Faraday (R3)
            E_stack += P_stack_star * dt                      # 직결분+방전분 = P_star
            E_bop += P_bop_star * dt
            op_hours += dt
            if E_rated > 0:
                eq_cycles += draw_need / E_rated
            E_direct += (p if p < P_star else P_star) * dt
            E_via += deficit * dt
            if keep_series:
                dis[i] = draw_need / dt
                h2s[i] = h2_rate * dt
                pus[i] = P_star

        if keep_series:
            soc[i] = E / E_rated if E_rated > 0 else 0.0
            run[i] = running
            chg[i] = e_chg / dt
            cur[i] = (chg_want - e_chg) / dt

    jj = np.where(run, op["j"], 0.0) if keep_series else np.zeros(0)
    return DispatchResult(
        case=C3_NAME, dt=dt, P_in=P, j=jj,
        P_used=pus if keep_series else np.zeros(0),
        H2_series=h2s if keep_series else np.zeros(0),
        soc=soc, running=run, charge=chg, discharge=dis, curtail_series=cur,
        E_paid=E_paid, E_curtail=E_curtail, E_idle=0.0, E_rte=E_rte,
        E_bop=E_bop, E_stack=E_stack, E_direct=E_direct, E_via_ess=E_via,
        H2_total=H2, op_hours=op_hours,
        hours_curtail=float((cur > 1e-9).sum() * dt) if keep_series else 0.0,
        hours_idle=0.0, hours_night=float((P <= 0.0).sum() * dt),
        restarts=restarts, eq_cycles=eq_cycles, E_end=E,
        E_rated=E_rated, P_rated_chg=P_rated_chg, P_rated_dis=P_rated_dis,
        j_star=op["j"], P_star=P_star,
        meta={"E_hi": E_hi, "E_lo": E_lo, "E_stop": E_stp, "E_restart": E_rs,
              "eta_RTE": eta,
              "min_run_duration_h": ((E_rs - E_stp) / (P_star / eta) * dt
                                     if P_star > 0 else np.inf)})


def c3_run_warmstart(P_in, op, A_tot, E_rated, P_rc, P_rd, dt=1.0, **kw):
    if not WARM_START or E_rated <= 0:
        return dispatch_case3(P_in, op, A_tot, E_rated, P_rc, P_rd, dt=dt, **kw)
    r1 = dispatch_case3(P_in, op, A_tot, E_rated, P_rc, P_rd, dt=dt,
                        keep_series=False, **kw)
    return dispatch_case3(P_in, op, A_tot, E_rated, P_rc, P_rd, dt=dt,
                          soc0=r1.E_end / E_rated, **kw)


def find_operating_point_case3(profile, fixed_stack, cap_hours_grid=None, n_j=None,
                               q_grid=None, mode=None, verbose=True) -> dict:
    mode = ESS_SIZING_MODE if mode is None else mode
    n_j = n_j or _n_j_grid("case3")
    A_tot = fixed_stack.A_tot
    j_grid = np.linspace(J_MIN, J_MAX, n_j)

    if mode == "surplus_q":
        key_grid = tuple(q_grid or _q_grid("case3"))
        key_label = "q [%] of daily-surplus distribution"
    elif mode == "hours":
        key_grid = tuple(cap_hours_grid or _cap_grid("case3"))
        key_label = "E_rated in hours of P*"
    else:
        raise ValueError(f"ESS_SIZING_MODE 값이 잘못되었습니다 -> {mode!r} "
                         f"('hours' | 'surplus_q')")

    rows, best, n_skip = [], None, 0
    for j in j_grid:
        op = c3_operating_point(j, A_tot)
        P_star = op["P_star"]
        P_rc, P_rd = c3_charge_rating(profile, P_star), P_star
        for key in key_grid:
            if mode == "surplus_q":
                E_rated = size_ess_from_surplus(profile.P_in, P_star, P_rc, key,
                                                dt=profile.dt, case="case3")
            else:
                E_rated = key * P_star          
            if E_rated <= 1e-9:                 
                n_skip += 1
                continue
            r = c3_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc, P_rd,
                                 dt=profile.dt,
                                 restart_hours=CASE3["restart_hours_of_Pstar"])
            if r.H2_total <= 0:
                continue
            cx = ess_capex(E_rated, max(P_rc, P_rd))
            li = lcoh_internal(r, fixed_stack, cx)  
            rows.append({"j": j, "key": key, "sizing_mode": mode,
                         "hours_of_P*": E_rated / P_star,
                         "P_star": P_star,
                         "E_rated": E_rated, "op_hours": r.op_hours,
                         "annual_H2_t": r.H2_total / 1000,
                         "curtail_%": r.E_curtail / r.E_paid * 100,
                         "rte_%": r.E_rte / r.E_paid * 100, "restarts": r.restarts,
                         "CAPEX_ESS_kUSD": cx / 1e3, "LCOH_internal": li})
            if best is None or li < best["LCOH_internal"]:
                best = {"j": j, "key": key, "mode": mode,
                        "hours": E_rated / P_star,     
                        "E_rated": E_rated, "LCOH_internal": li,
                        "result": r, "P_rc": P_rc, "P_rd": P_rd, "op": op}

    df = pd.DataFrame(rows)
    if verbose and len(df):
        ji = list(j_grid).index(best["j"])
        ki = list(key_grid).index(best["key"])
        key_txt = (f"q={best['key']:g}%" if mode == "surplus_q"
                   else f"{best['key']:g} h of P*")
        print(f"\n[Case 3 2D search] mode={mode}, grid={n_j}x{len(key_grid)}, "
              f"best j={best['j']:.4f} A/cm2, E={best['E_rated']:,.0f} kWh "
              f"({key_txt}, {best['hours']:.2f} h), "
              f"LCOH_int={best['LCOH_internal']:.3f}, "
              f"inside_j={'YES' if 0 < ji < n_j - 1 else 'NO'}, "
              f"inside_E={'YES' if 0 < ki < len(key_grid) - 1 else 'NO'}, "
              f"skipped={n_skip}")
    return {"best": best, "records": df, "mode": mode,
            "key_grid": list(key_grid), "key_label": key_label,
            "cap_grid": list(key_grid),          
            "j_grid": j_grid, "n_skipped": n_skip}


def size_ess_case3(profile, fixed_stack, j, cap_hours_grid=None, q_grid=None,
                   mode=None, verbose=True):
    mode = ESS_SIZING_MODE if mode is None else mode
    A_tot = fixed_stack.A_tot
    op = c3_operating_point(j, A_tot)
    P_star = op["P_star"]
    P_rc, P_rd = c3_charge_rating(profile, P_star), P_star
    rippl = rippl_capacity(profile.P_in, P_star, ETA_RTE, profile.dt)

    if mode == "surplus_q":
        key_grid = tuple(q_grid or _q_grid("case3"))
    elif mode == "hours":
        key_grid = tuple(cap_hours_grid or _cap_grid("case3"))
    else:
        raise ValueError(f"ESS_SIZING_MODE 값이 잘못되었습니다 -> {mode!r}")

    rows, best = [], None
    for key in key_grid:
        if mode == "surplus_q":
            E_rated = size_ess_from_surplus(profile.P_in, P_star, P_rc, key,
                                            dt=profile.dt, case="case3")
        else:
            E_rated = key * P_star
        if E_rated <= 1e-9:
            continue
        h_eq = E_rated / P_star
        r = c3_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc, P_rd,
                             dt=profile.dt,
                             restart_hours=CASE3["restart_hours_of_Pstar"])
        cx = ess_capex(E_rated, max(P_rc, P_rd))
        li = lcoh_internal(r, fixed_stack, cx)
        lc = lcoh_compare(r, fixed_stack, cx)
        row = {"hours_of_P*": h_eq, "E_rated [kWh]": E_rated, "op_hours": r.op_hours,
               "annual_H2 [t]": r.H2_total / 1000,
               "curtail [%]": r.E_curtail / r.E_paid * 100,
               "RTE loss [%]": r.E_rte / r.E_paid * 100,
               "direct [%]": r.E_direct / r.E_paid * 100,
               "restarts": r.restarts, "eq_cycles": r.eq_cycles,
               "CAPEX_ESS [kUSD]": cx / 1e3,
               "LCOH_internal": li, "LCOH_compare": lc["LCOH"]}
        if mode == "surplus_q":
            row = {"q [%]": key, **row}          # 표 맨 앞에 sweep 축을 세운다
        rows.append(row)
        if best is None or li < best[1]:
            best = (h_eq, li, E_rated, r)

    df = pd.DataFrame(rows)
    if best is None:
        raise RuntimeError(f"Case 3: j = {j:.4f} 에서 유효한 ESS 용량 후보가 없습니다 "
                           f"(모드 '{mode}'). 잉여가 전혀 없는 j 일 수 있습니다.")
    if mode == "surplus_q" and len(df) > 1 and _ESS_AUTO_EXTEND_Q:
        for _ in range(3):
            k = int(df["LCOH_internal"].idxmin())
            if k != 0:                       # 내부에 최적점이 생겼다 -> 끝
                break
            q_lo = float(df.iloc[0]["q [%]"])
            if q_lo <= 0.5:                  # 더 내려갈 곳이 없다
                break
            add = tuple(round(q_lo * f, 3) for f in (0.25, 0.5, 0.75))
            if verbose:
                print(f"  [Case 3 q-autoextend] add q={', '.join(f'{a:g}' for a in add)}")
            for key in add:
                E_rated = size_ess_from_surplus(profile.P_in, P_star, P_rc, key,
                                                dt=profile.dt, case="case3")
                if E_rated <= 1e-9:
                    continue
                h_eq = E_rated / P_star
                r = c3_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc, P_rd,
                                     dt=profile.dt,
                                     restart_hours=CASE3["restart_hours_of_Pstar"])
                cx = ess_capex(E_rated, max(P_rc, P_rd))
                li = lcoh_internal(r, fixed_stack, cx)
                lc = lcoh_compare(r, fixed_stack, cx)
                df = pd.concat([df, pd.DataFrame([{
                    "q [%]": key, "hours_of_P*": h_eq, "E_rated [kWh]": E_rated,
                    "op_hours": r.op_hours, "annual_H2 [t]": r.H2_total / 1000,
                    "curtail [%]": r.E_curtail / r.E_paid * 100,
                    "RTE loss [%]": r.E_rte / r.E_paid * 100,
                    "direct [%]": r.E_direct / r.E_paid * 100,
                    "restarts": r.restarts, "eq_cycles": r.eq_cycles,
                    "CAPEX_ESS [kUSD]": cx / 1e3,
                    "LCOH_internal": li, "LCOH_compare": lc["LCOH"]}])],
                    ignore_index=True)
                if li < best[1]:
                    best = (h_eq, li, E_rated, r)
            df = df.sort_values("q [%]").reset_index(drop=True)

    if verbose:
        k = int(df["LCOH_internal"].idxmin())
        edge = k == 0 or k == len(df) - 1
        q_txt = f", q={df.loc[k, 'q [%]']:g}%" if "q [%]" in df.columns else ""
        print(f"\n[Case 3 ESS sweep] mode={mode}, j={j:.4f}, P*={P_star:,.1f} kW, "
              f"E*={df.loc[k, 'E_rated [kWh]']:,.0f} kWh "
              f"({df.loc[k, 'hours_of_P*']:.2f} h{q_txt}), "
              f"LCOH_int={df.loc[k, 'LCOH_internal']:.3f}, "
              f"inside={'NO' if edge else 'YES'}, Rippl={rippl / 1e3:,.1f} MWh")
    return best, df, rippl


def run_case3(profile=None, fixed_stack=None, j_star=None, cap_hours=None,
              optimize=True, verbose=True) -> dict:
    profile = profile or get_profile()
    fixed_stack = fixed_stack or build_fixed_stack(profile, verbose=verbose)
    A_tot = fixed_stack.A_tot

    if verbose:
        print(f"\n[PART 13] {C3_NAME} — hybrid direct + ESS deficit buffer")

    opt = None
    fixed_mode = j_is_fixed() and j_star is None
    if fixed_mode:
        j_star = fixed_j("case3", verbose=verbose)
        if verbose:
            print(f"\n[Case 3 fixed-j] j={j_star:.4f} A/cm2; skip 2D search")
    if optimize and not fixed_mode and (j_star is None or cap_hours is None):
        opt = find_operating_point_case3(profile, fixed_stack, verbose=verbose)
        if opt["best"] is None:
            raise RuntimeError("Case 3: 어떤 조합에서도 운전이 성립하지 않았습니다.")
        j_star = j_star if j_star is not None else opt["best"]["j"]
        cap_hours = cap_hours if cap_hours is not None else opt["best"]["hours"]
    j_star = 0.9 * J_MAX if j_star is None else j_star

    (cap_hours, _, _, _), cap_df, rippl = size_ess_case3(profile, fixed_stack, j_star,
                                                          verbose=verbose)
    op = c3_operating_point(j_star, A_tot)
    E_rated = cap_hours * op["P_star"]
    P_rc, P_rd = c3_charge_rating(profile, op["P_star"]), op["P_star"]

    res = c3_run_warmstart(profile.P_in, op, A_tot, E_rated, P_rc, P_rd, dt=profile.dt,
                           restart_hours=CASE3["restart_hours_of_Pstar"])
    cx = ess_capex(E_rated, max(P_rc, P_rd))
    lc = lcoh_compare(res, fixed_stack, cx)
    dcf = lcoh_dcf(res, fixed_stack, cx)

    if verbose:
        mrd = res.meta["min_run_duration_h"]
        print("\n[Case 3 operating point]")
        print(f"  j*={j_star:.4f} A/cm2, V={float(V(j_star)[0]):.4f} V, "
              f"SEC_stack={float(SEC_stack(j_star)[0]):.2f} kWh/kg")
        print(f"  P*={op['P_star']:,.1f} kW (stack={op['P_stack']:,.1f}, "
              f"BOP={op['P_bop']:,.1f}), P*/mean={op['P_star'] / profile.mean:.2f}")
        print(f"  E*={E_rated:,.0f} kWh ({E_rated / 1e3:,.2f} MWh, {cap_hours:.2f} h), "
              f"Pchg/Pdis={P_rc:,.1f}/{P_rd:,.1f} kW, CAPEX_ESS={cx:,.0f} USD")
        print(f"  SOC restart/stop/floor={res.meta['E_restart'] / E_rated:.3f}/"
              f"{CASE3['soc_stop']:.2f}/{CASE3['soc_floor']:.2f}, "
              f"min_run={mrd:.2f} h, direct={res.E_direct / 1e3:,.1f} MWh, "
              f"RTE={res.E_rte / 1e3:,.1f} MWh")
        if mrd < 1.0:
            print("  [CHECK] hysteresis band < 1 h; consider restart_hours_of_Pstar")
        res.report(h2_area_at_rated=float(h2_area(J_MAX)[0]),
                   h2_area_at_jstar=float(h2_area(j_star)[0]), A_tot=A_tot)
        report_lcoh(C3_NAME, lc, dcf)

    return {"result": res, "lcoh": lc, "lcoh_dcf": dcf, "op": op, "j_star": j_star,
            "cap_hours": cap_hours, "E_rated": E_rated, "capex_ess": cx,
            "cap_sweep": cap_df, "rippl": rippl, "opt": opt,
            "stack": fixed_stack, "profile": profile}


def plot_case3(out: dict) -> None:
    if plt is None:
        return
    setup_plots()
    res, profile, op, E_rated = out["result"], out["profile"], out["op"], out["E_rated"]

    opt = out.get("opt")
    if opt and len(opt["records"]):
        mode = opt.get("mode", "hours")
        piv = opt["records"].pivot_table(index="j", columns="key",
                                         values="LCOH_internal")
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis_r",
                       extent=[-0.5, piv.shape[1] - 0.5, piv.index.min(), piv.index.max()])
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{col:g}" for col in piv.columns])
        ax.set_xlabel("ESS sizing key: q [%] of daily-surplus distribution   "
                      "(left -> curtail more, right -> store all)"
                      if mode == "surplus_q" else
                      "ESS energy rating [hours of P_star]   "
                      "(left edge -> Case 1-like, right edge -> Case 2-like)")
        ax.set_ylabel("Current density j [A/cm$^2$]")
        ax.set_title("Case 3 (N1) — Internal LCOH surface over "
                     f"(j, E_rated)   [sizing = '{mode}']")
        b = opt["best"]
        ktxt = (f"q={b['key']:g}%" if mode == "surplus_q" else f"{b['key']:g} h")
        ax.plot(list(piv.columns).index(b["key"]), b["j"], "*", ms=20, color="w",
                markeredgecolor="k",
                label=f"optimum j*={b['j']:.3f}, {ktxt} ({b['hours']:.2f} h eq.)")
        ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=ax, label="LCOH [USD/kg]")
        fig.tight_layout()
        _finish(fig, "case3_N1_lcoh_heatmap")

    if ESS_SIZING_MODE == "surplus_q":
        P_rc_ = c3_charge_rating(profile, op["P_star"])
        e_draw = daily_storage_need(profile.P_in, op["P_star"], P_rc_, dt=profile.dt)
        usable = CASE3["soc_max"] - CASE3["soc_stop"]
        q_sel = (opt or {}).get("best", {}).get("key") if opt else None
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.hist(e_draw / 1e3, bins=60, color="#2a9d8f", alpha=0.75)
        ax.axvline(E_rated * usable / 1e3, color="#e76f51", ls="--", lw=1.8,
                   label=f"selected E_rated x SOC window = {E_rated * usable / 1e3:,.1f} MWh"
                         + (f"  (q = {q_sel:g} %)" if q_sel is not None else ""))
        ax.axvline(e_draw.max() / 1e3, color="#9c6644", ls=":", lw=1.4,
                   label=f"worst day = {e_draw.max() / 1e3:,.1f} MWh (q = 100)")
        ax.set_xlabel("Daily energy the battery must absorb [MWh]   "
                      "(= green area above P*, charge-rating clipped; RTE is applied on discharge in dispatch)")
        ax.set_ylabel("Days per year")
        ax.set_title("Case 3 (N1b) — Daily surplus distribution drives ESS sizing")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _finish(fig, "case3_N1b_daily_surplus")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    parts = [("direct to PEM\n(no RTE loss)", res.E_direct, "#2a9d8f"),
             ("via ESS to PEM\n(RTE applies here only)", res.E_via_ess, "#e9c46a"),
             ("RTE loss", res.E_rte, "#e76f51"),
             ("curtailed", res.E_curtail, "#9c6644")]
    vals = [p[1] / 1e3 for p in parts]
    ax.bar([p[0] for p in parts], vals, color=[p[2] for p in parts], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}\n({v / (res.E_paid / 1e3) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Energy [MWh/yr]")
    ax.set_title("Case 3 (N2) — Energy path split: RTE is charged only on stored part")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    _finish(fig, "case3_N2_energy_path")

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.plot(np.arange(len(res.soc)) / 24.0, res.soc * 100, lw=0.6, color="#2a9d8f")
    ax.axhline(CASE3["soc_max"] * 100, color="#e76f51", ls="--", lw=1, label="SOC_max 90%")
    ax.axhline(res.meta["E_restart"] / E_rated * 100, color="#1f6f8b", ls="-.", lw=1,
               label="SOC_restart")
    ax.axhline(CASE3["soc_stop"] * 100, color="#9c6644", ls=":", lw=1.4, label="SOC_stop 15%")
    ax.axhline(CASE3["soc_floor"] * 100, color="#555", ls=":", lw=1, label="SOC_floor 10%")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("SOC [%]")
    ax.set_xlim(0, 365)
    ax.set_title("Case 3 — Annual SOC (same shape as Case 2, much smaller battery)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    _finish(fig, "case3_soc_timeseries")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    dur = duration_curve(res.P_in)
    x = np.arange(len(dur))
    ax.plot(x, dur, color="k", lw=1.2, label="P_in (sorted)")
    ax.fill_between(x, op["P_star"], dur, where=dur > op["P_star"], color="#e9c46a",
                    alpha=0.6, label="surplus -> ESS only")
    ax.fill_between(x, 0, np.minimum(dur, op["P_star"]), color="#2a9d8f", alpha=0.3,
                    label="direct to PEM (no RTE)")
    ax.axhline(op["P_star"], color="#e76f51", ls="--", lw=1.4,
               label=f"P_star = {op['P_star']:,.0f} kW")
    ax.set_xlabel("Hours of year (sorted, descending)")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim(0, len(dur))
    ax.set_title("Case 3 — Duration curve: only the yellow area touches the battery")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case3_duration_vs_Pstar")

    df = out["cap_sweep"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(df["hours_of_P*"], df["annual_H2 [t]"], "o-", color="#2a9d8f", lw=2,
            label="annual H$_2$")
    ax.set_xlabel("ESS energy rating [hours of P_star]")
    ax.set_ylabel("Annual H$_2$ [t]")
    ax2 = ax.twinx()
    ax2.plot(df["hours_of_P*"], df["LCOH_compare"], "s--", color="#1f6f8b",
             label="LCOH (compare)")
    ax2.plot(df["hours_of_P*"], df["RTE loss [%]"], "^:", color="#e76f51",
             label="RTE loss %")
    ax2.set_ylabel("LCOH [USD/kg] / RTE loss [%]")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.set_title("Case 3 — ESS capacity sweep (optimum lands small)")
    fig.tight_layout()
    _finish(fig, "case3_capacity_sweep")

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.imshow(operating_heatmap_data(res.running), aspect="auto", origin="lower",
              cmap="YlGn", extent=[0, 365, 0, 24], vmin=0, vmax=1)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_title("Case 3 — Operating state map")
    fig.tight_layout()
    _finish(fig, "case3_operating_heatmap")

    days = profile.representative_days()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (label, idx) in zip(axes, days.items()):
        if len(idx) == 0:
            continue
        h = np.arange(len(idx))
        ax.fill_between(h, 0, res.P_in[idx], color="#c8d8dd", label="P_in")
        ax.plot(h, res.P_used[idx], color="#2a9d8f", lw=2, label="P_used (= P*)")
        ax.axhline(op["P_star"], color="#e76f51", ls="--", lw=1)
        ax.set_xlabel("Hour of day")
        ax.set_title(f"Case 3 — representative day: {label}")
        ax2 = ax.twinx()
        ax2.plot(h, res.soc[idx] * 100, color="#1f6f8b", lw=1.6)
        ax2.set_ylim(0, 100)
        ax2.grid(False)
        if ax is axes[-1]:
            ax2.set_ylabel("SOC [%]")
    axes[0].set_ylabel("Power [kW]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _finish(fig, "case3_representative_days")

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    bottom = 0.0
    for k in ("stack", "bop", "rte", "curtail"):
        v = res.ledger[k] / 1e3
        if v <= 0:
            continue
        ax.bar(["E_paid"], [v], bottom=[bottom], color=LEDGER_COLORS[k],
               label=f"{LEDGER_LABELS[k]}  ({v / (res.E_paid / 1e3) * 100:.1f}%)")
        bottom += v
    ax.set_ylabel("Energy [MWh/yr]")
    ax.set_title("Case 3 — loss ledger (same buckets as Case 2, smaller E_rte)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, "case3_loss_ledger")

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    s = pd.Series(res.H2_series).groupby(profile.month).sum()
    ax.bar(s.index, s.values / 1000.0, color="#2a9d8f", width=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("H$_2$ [t]")
    ax.set_title("Case 3 — Monthly H$_2$")
    fig.tight_layout()
    _finish(fig, "case3_monthly_H2")

# %%

# ###################################################################################
# ###################################################################################
##
##   PART 14 — 3케이스 비교 그래프                     [결론 그림]
##
##   출력 순서 (총 16장)
##     G01                입력 : 월별 재생에너지 발전량
##     G02 ~ G04          케이스별 수소 생산 시간   (Case 1 / 2 / 3)
##     G05 ~ G07          케이스별 수소 생산량      (Case 1 / 2 / 3)
##     G08 ~ G10          케이스별 전류밀도 vs LCOH (Case 1 / 2 / 3)
##     G11                통합 : 케이스별 총 수소 생산량 (막대)
##     G12                통합 : 월별 수소 생산량
##     G13                통합 : 수소 생산 시간
##     G14                LCOH + 손실 원장
##     G15                LCOH breakdown (항목별 $/kg)
##     G16                지불 에너지 breakdown (항목별 MWh)
##
# ###################################################################################
# ###################################################################################

_CASE_ORDER = ["Case 1", "Case 2", "Case 3"]


def _production_hours(res, dt: float = 1.0) -> dict:
    P_in = np.asarray(res.P_in, dtype=float)
    j = np.asarray(res.j, dtype=float)
    producing = j > 1e-9
    gen = P_in > 1e-9
    return {"producing": float(producing.sum() * dt),
            "idle_with_gen": float((gen & ~producing).sum() * dt),
            "no_gen": float((~gen).sum() * dt),
            "mask": producing}


def _monthly(series, months, agg="sum"):
    x = np.arange(1, 13)
    s = pd.Series(np.asarray(series, dtype=float)).groupby(months)
    s = s.sum() if agg == "sum" else s.mean()
    return x, s.reindex(x, fill_value=0.0).values


def plot_g01_re_generation(profile, name="G01_RE_generation_monthly"):
    if plt is None:
        return
    x, mwh = _monthly(profile.P_in, profile.month)
    mwh = mwh / 1e3
    _, hrs = _monthly(np.ones(len(profile.P_in)), profile.month)
    try:
        nameplate_kw = float(profile.meta.get("re_capacity_MW")) * 1e3
        if not np.isfinite(nameplate_kw) or nameplate_kw <= 0:
            raise ValueError
    except (TypeError, ValueError):
        nameplate_kw = float(np.max(profile.P_in)) if profile.P_in.size else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(hrs > 0, mwh * 1e3 / (nameplate_kw * hrs) * 100.0, np.nan)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x, mwh, width=0.62, color="#f4a261", label="Monthly generation")
    for xi, v in zip(x, mwh):
        ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Month")
    ax.set_ylabel("RE generation [MWh]")
    ax.set_ylim(0, mwh.max() * 1.18 if mwh.max() > 0 else 1)

    ax2 = ax.twinx()
    ax2.plot(x, cf, color="#264653", lw=1.8, marker="o", ms=4, label="Capacity factor")
    ax2.set_ylabel("Capacity factor [%]")
    ax2.set_ylim(0, np.nanmax(cf) * 1.6 if np.isfinite(np.nanmax(cf)) else 1)
    ax2.grid(False)

    ax.set_title(f"Monthly renewable generation — {profile.region} {profile.year}"
                 f"  ({nameplate_kw / 1e3:,.2f} MW, {mwh.sum():,.0f} MWh/yr)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    _finish(fig, name)


def plot_case_operating_hours(case: str, res, months, idx: int):
    if plt is None:
        return
    h = _production_hours(res, getattr(res, "dt", 1.0) or 1.0)
    color = CASE_COLORS.get(case, "#333333")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    vals = [h["producing"], h["idle_with_gen"], h["no_gen"]]
    labels = ["producing H$_2$", "generation available,\nstopped", "no generation\n(night etc.)"]
    colors = [color, "#e9c46a", "#adb5bd"]
    w, _tx, _at = ax.pie(vals, labels=labels, colors=colors, startangle=90,
                         autopct=lambda p: f"{p:.1f}%\n({p / 100 * sum(vals):,.0f} h)",
                         pctdistance=0.72, textprops={"fontsize": 8},
                         wedgeprops={"width": 0.45, "edgecolor": "w"})
    ax.set_title(f"Annual hour ledger (total {sum(vals):,.0f} h)")

    ax = axes[1]
    x, mh = _monthly(h["mask"].astype(float), months)
    ax.bar(x, mh, width=0.62, color=color)
    for xi, v in zip(x, mh):
        ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xlabel("Month")
    ax.set_ylabel("Producing hours [h]")
    ax.set_ylim(0, mh.max() * 1.16 if mh.max() > 0 else 1)
    ax.set_title("Monthly hydrogen-producing hours")

    fig.suptitle(f"{case} — hydrogen production hours "
                 f"({h['producing']:,.0f} h/yr)", fontsize=12)
    fig.tight_layout()
    _finish(fig, f"G{idx:02d}_{case.replace(' ', '').lower()}_operating_hours")


def plot_case_monthly_h2(case: str, res, months, idx: int):
    if plt is None:
        return
    x, kg = _monthly(res.H2_series, months)
    t = kg / 1e3
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x, t, width=0.62, color=CASE_COLORS.get(case, "#333333"))
    for xi, v in zip(x, t):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Month")
    ax.set_ylabel("H$_2$ [t]")
    ax.set_ylim(0, t.max() * 1.16 if t.max() > 0 else 1)
    ax.set_title(f"{case} — monthly hydrogen production "
                 f"({res.H2_total / 1e3:,.2f} t/yr)")
    fig.tight_layout()
    _finish(fig, f"G{idx:02d}_{case.replace(' ', '').lower()}_monthly_H2")


def lcoh_vs_j_case1(profile, fixed_stack, n: int = 25) -> pd.DataFrame:
    j_lo, j_hi = max(2.0 * J_MIN, 0.4), 0.92 * J_LIM
    rows = []
    for jr in np.linspace(j_lo, j_hi, n):
        st = size_fixed_stack(fixed_stack.P_nameplate, j_rated=float(jr))
        res = dispatch_case1(profile.P_in, st.A_tot, J_MIN, float(jr), dt=profile.dt)
        lc = lcoh_compare(res, st, capex_ess=0.0)
        rows.append({"j": float(jr), "LCOH": lc["LCOH"], "H2_t": res.H2_total / 1e3,
                     "curtail_%": res.E_curtail / res.E_paid * 100 if res.E_paid else np.nan})
    return pd.DataFrame(rows)


def lcoh_vs_j_case23(profile, fixed_stack, cap_hours: float, case: str,
                     n: int = 17) -> pd.DataFrame:
    A = fixed_stack.A_tot
    op_fn = c2_operating_point if case == "Case 2" else c3_operating_point
    chg_fn = c2_charge_rating if case == "Case 2" else c3_charge_rating
    restart = (CASE2 if case == "Case 2" else CASE3)["restart_hours_of_Pstar"]

    rows = []
    for j in np.linspace(max(2.0 * J_MIN, 0.4), J_MAX, n):
        op = op_fn(float(j), A)
        E_rated = float(cap_hours) * op["P_star"]
        P_rc = chg_fn(profile, op["P_star"])
        P_rd = op["P_star"]  # ESS CAPEX 계산용. Case 2도 방전정격은 P_star로 간주.

        try:
            if case == "Case 2":
                res = c2_run_warmstart(profile.P_in, op, A, E_rated, P_rc,
                                       dt=profile.dt, restart_hours=restart)
            else:
                res = c3_run_warmstart(profile.P_in, op, A, E_rated, P_rc, P_rd,
                                       dt=profile.dt, restart_hours=restart)
        except Exception:
            continue

        if res.H2_total <= 0:
            continue

        cx = ess_capex(E_rated, max(P_rc, P_rd))
        lc = lcoh_compare(res, fixed_stack, cx)

        rows.append({
            "j": float(j),
            "LCOH": lc["LCOH"],
            "H2_t": res.H2_total / 1e3,
            "E_rated": E_rated,
            "op_hours": res.op_hours,
        })

    return pd.DataFrame(rows)


def plot_case_lcoh_vs_j(case: str, df: pd.DataFrame, j_used: float | None,
                        lcoh_used: float | None, idx: int, xlabel: str):
    if plt is None or df is None or not len(df):
        return
    color = CASE_COLORS.get(case, "#333333")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(df["j"], df["LCOH"], color=color, lw=2.2, marker="o", ms=4,
            label="LCOH(j)")

    k = int(df["LCOH"].idxmin())
    j_opt, l_opt = df.loc[k, "j"], df.loc[k, "LCOH"]
    ax.axvline(j_opt, color=color, ls="--", lw=1.1, alpha=0.7)
    ax.plot([j_opt], [l_opt], marker="*", ms=15, color="#e76f51", zorder=5,
            label=f"minimum  j = {j_opt:.2f}, LCOH = {l_opt:.2f}")
    edge = (k == 0) or (k == len(df) - 1)
    if edge:
        ax.text(0.02, 0.06, "※ 최소점이 탐색 구간의 경계입니다 (내부 최적 아님)",
                transform=ax.transAxes, fontsize=8, color="#e76f51")

    if j_used is not None and lcoh_used is not None:
        ax.plot([j_used], [lcoh_used], marker="D", ms=8, mfc="none",
                mec="#264653", mew=1.8, zorder=5,
                label=f"model used  j = {j_used:.2f}, LCOH = {lcoh_used:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("LCOH [USD/kg-H$_2$]")
    ax.set_title(f"{case} — LCOH vs current density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, f"G{idx:02d}_{case.replace(' ', '').lower()}_LCOH_vs_j")


def plot_g11_total_h2(results: dict, name="G11_compare_total_H2"):
    if plt is None:
        return
    cases = [case for case in _CASE_ORDER if results.get(case) is not None]
    if not cases:
        return
    tons = np.array([results[case].H2_total / 1e3 for case in cases], dtype=float)
    colors = [CASE_COLORS.get(case, "#333333") for case in cases]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(cases, tons, width=0.55, color=colors, edgecolor="white", zorder=3)

    base = float(tons[0]) if tons.size and tons[0] > 0 else float("nan")
    top = float(tons.max()) if tons.size else 1.0
    for i, (b, v) in enumerate(zip(bars, tons)):
        label = f"{v:,.2f} t"
        if i > 0 and np.isfinite(base):
            label += f"\n({(v / base - 1) * 100:+.1f}% vs {cases[0]})"
        ax.text(b.get_x() + b.get_width() / 2, v, label,
                ha="center", va="bottom", fontsize=9, zorder=4)

    k = int(np.argmax(tons))
    ax.set_ylabel("Annual H$_2$ production [t/yr]")
    ax.set_ylim(0, top * 1.24 if top > 0 else 1.0)
    ax.set_title("Total annual hydrogen production — 3 connection cases\n"
                 f"(max : {cases[k]}, {tons[k]:,.2f} t/yr)")
    ax.xaxis.grid(False)
    fig.tight_layout()
    _finish(fig, name)


plot_g11_cumulative_h2 = plot_g11_total_h2


def plot_g12_monthly_h2(results: dict, months, name="G12_compare_monthly_H2"):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    w = 0.26
    for i, case in enumerate(_CASE_ORDER):
        r = results.get(case)
        if r is None:
            continue
        x, kg = _monthly(r.H2_series, months)
        ax.bar(x + (i - 1) * w, kg / 1e3, width=w, color=CASE_COLORS.get(case),
               label=f"{case}  ({r.H2_total / 1e3:,.1f} t/yr)")
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("H$_2$ [t]")
    ax.set_title("Monthly hydrogen production — 3 connection cases")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _finish(fig, name)


def plot_g13_operating_hours(results: dict, months, name="G13_compare_operating_hours"):
    if plt is None:
        return
    cases = [case for case in _CASE_ORDER if case in results]
    hrs = {case: _production_hours(results[case]) for case in cases}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    segs = [("producing", "producing H$_2$", "#2a9d8f"),
            ("idle_with_gen", "generation available, stopped", "#e9c46a"),
            ("no_gen", "no generation (night etc.)", "#adb5bd")]
    lefts = np.zeros(len(cases))
    for key, label, color in segs:
        vals = np.array([hrs[case][key] for case in cases])
        ax.barh(cases, vals, left=lefts, color=color, label=label, height=0.5)
        for i, v in enumerate(vals):
            if v > 350:
                ax.text(lefts[i] + v / 2, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=9)
        lefts += vals
    ax.set_xlabel("Hours per year [h]")
    ax.set_xlim(0, lefts.max() * 1.02)
    ax.set_title("Annual hour ledger")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.invert_yaxis()

    ax = axes[1]
    w = 0.26
    for i, case in enumerate(cases):
        x, mh = _monthly(hrs[case]["mask"].astype(float), months)
        ax.bar(x + (i - 1) * w, mh, width=w, color=CASE_COLORS.get(case),
               label=f"{case}  ({hrs[case]['producing']:,.0f} h/yr)")
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Producing hours [h]")
    ax.set_title("Monthly hydrogen-producing hours")
    ax.legend(fontsize=8)

    fig.suptitle("Hydrogen production hours — 3 connection cases", fontsize=12)
    fig.tight_layout()
    _finish(fig, name)


def plot_g14_lcoh_and_ledger(results: dict, lcohs: dict, name="G14_LCOH_and_ledger"):
    if plt is None:
        return
    cases = [case for case in _CASE_ORDER if case in results]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    bars = ax.bar(cases, [lcohs[case]["LCOH"] for case in cases],
                  color=[CASE_COLORS.get(case) for case in cases], width=0.55)
    for b, case in zip(bars, cases):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=10)
        share = lcohs[case].get("ess_share_pct", 0.0)
        if share and share > 0.05:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5,
                    f"ESS\n{share:.1f}%", ha="center", va="center", fontsize=9, color="w")
    ax.set_ylabel("LCOH [USD/kg-H$_2$]")
    ax.set_title("Levelized cost of hydrogen")

    ax = axes[1]
    bottoms = np.zeros(len(cases))
    for k in ("stack", "bop", "rte", "idle", "curtail"):
        vals = np.array([results[case].ledger[k] / 1e3 for case in cases])
        if vals.sum() <= 0:
            continue
        ax.bar(cases, vals, bottom=bottoms, color=LEDGER_COLORS[k],
               label=LEDGER_LABELS[k], width=0.55)
        for i, v in enumerate(vals):
            tot = results[cases[i]].E_paid / 1e3
            if tot > 0 and v / tot > 0.04:
                ax.text(i, bottoms[i] + v / 2, f"{v / tot * 100:.1f}%",
                        ha="center", va="center", fontsize=8)
        bottoms += vals
    ax.set_ylabel("Paid energy [MWh/yr]")
    ax.set_ylim(0, bottoms.max() * 1.28)
    ax.set_title("Where the paid energy goes (loss ledger)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("LCOH and loss ledger", fontsize=12)
    fig.tight_layout()
    _finish(fig, name)


_LCOH_ITEM_LABELS = {
    "annualized_capex": "annualized CAPEX",
    "fixed_OM": "fixed O&M",
    "stack_replacement": "stack replacement",
    "battery_replacement": "battery replacement",
    "electricity": "electricity",
    "curtailment_penalty": "curtailment penalty",
    "variable_OM": "variable O&M",
    "water": "water",
}
_LCOH_ITEM_COLORS = {
    "annualized_capex": "#264653", "fixed_OM": "#2a9d8f",
    "stack_replacement": "#8ab17d", "battery_replacement": "#b07d62",
    "electricity": "#e9c46a", "curtailment_penalty": "#e76f51",
    "variable_OM": "#f4a261", "water": "#a8dadc",
}


def plot_g15_lcoh_breakdown(lcohs: dict, name="G15_LCOH_breakdown"):
    if plt is None:
        return
    cases = [case for case in _CASE_ORDER if case in lcohs]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bottoms = np.zeros(len(cases))
    for key, label in _LCOH_ITEM_LABELS.items():
        vals = np.array([lcohs[case]["items_usd_per_kg"].get(key, 0.0) for case in cases])
        if vals.sum() <= 1e-9:
            continue
        ax.bar(cases, vals, bottom=bottoms, width=0.55,
               color=_LCOH_ITEM_COLORS[key], label=label)
        for i, v in enumerate(vals):
            if v > 0.35:
                ax.text(i, bottoms[i] + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="w")
        bottoms += vals
    for i, case in enumerate(cases):
        ax.text(i, bottoms[i], f"{lcohs[case]['LCOH']:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("LCOH [USD/kg-H$_2$]")
    ax.set_ylim(0, bottoms.max() * 1.30)
    ax.set_title("LCOH breakdown by cost item")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    _finish(fig, name)


def plot_g16_energy_breakdown(results: dict, name="G16_energy_paid_breakdown"):
    if plt is None:
        return
    cases = [case for case in _CASE_ORDER if case in results]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    bottoms = np.zeros(len(cases))
    for k in ("stack", "bop", "rte", "idle", "curtail"):
        vals = np.array([results[case].ledger[k] / 1e3 for case in cases])
        if vals.sum() <= 0:
            continue
        ax.bar(cases, vals, bottom=bottoms, color=LEDGER_COLORS[k],
               label=LEDGER_LABELS[k], width=0.55)
        for i, v in enumerate(vals):
            if v / (results[cases[i]].E_paid / 1e3) > 0.04:
                ax.text(i, bottoms[i] + v / 2, f"{v:,.0f}", ha="center", va="center",
                        fontsize=8)
        bottoms += vals
    ax.set_ylabel("Paid energy [MWh/yr]")
    ax.set_ylim(0, bottoms.max() * 1.30)
    ax.set_title("Absolute energy [MWh/yr]")
    ax.legend(fontsize=8, ncol=2, loc="upper center")

    ax = axes[1]
    bottoms = np.zeros(len(cases))
    for k in ("stack", "bop", "rte", "idle", "curtail"):
        vals = np.array([results[case].ledger[k] / max(results[case].E_paid, 1e-9) * 100
                         for case in cases])
        if vals.sum() <= 0:
            continue
        ax.bar(cases, vals, bottom=bottoms, color=LEDGER_COLORS[k], width=0.55)
        for i, v in enumerate(vals):
            if v > 3.0:
                ax.text(i, bottoms[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8)
        bottoms += vals
    ax.set_ylabel("Share of paid energy [%]")
    ax.set_ylim(0, 118)
    ax.set_title("Share [%]")
    fig.suptitle("Paid-energy breakdown", fontsize=12)
    fig.tight_layout()
    _finish(fig, name)


def make_all_figures(profile, results: dict, lcohs: dict, fixed_stack,
                     out1: dict, out2: dict, out3: dict, verbose: bool = True):
    if plt is None:
        return {}
    setup_plots()
    sweeps = {}

    plot_g01_re_generation(profile)                                       # G01

    for i, case in enumerate(_CASE_ORDER):                                # G02~G04
        if case in results:
            plot_case_operating_hours(case, results[case], profile.month, 2 + i)

    for i, case in enumerate(_CASE_ORDER):                                # G05~G07
        if case in results:
            plot_case_monthly_h2(case, results[case], profile.month, 5 + i)

    if verbose:
        print("  전류밀도 sweep 계산 중 ... (G08~G10)")
    d1 = lcoh_vs_j_case1(profile, fixed_stack)                            # G08
    sweeps["Case 1"] = d1
    plot_case_lcoh_vs_j("Case 1", d1, J_MAX, lcohs["Case 1"]["LCOH"], 8,
                        "Rated current density  j$_{rated}$ [A/cm$^2$]  (stack resized)")

    for i, (case, out) in enumerate([("Case 2", out2), ("Case 3", out3)]):  # G09~G10
        if case not in results:
            continue
        cap_h = out.get("cap_hours")
        if cap_h is None and results[case].P_star:
            cap_h = results[case].E_rated / max(results[case].P_star, 1e-9)
        d = lcoh_vs_j_case23(profile, fixed_stack, float(cap_h or 1.0), case)
        sweeps[case] = d
        plot_case_lcoh_vs_j(case, d, out.get("j_star"), lcohs[case]["LCOH"], 9 + i,
                            "Operating current density  j* [A/cm$^2$]  (stack fixed)")

    plot_g11_total_h2(results)                                            # G11
    plot_g12_monthly_h2(results, profile.month)                           # G12
    plot_g13_operating_hours(results, profile.month)                      # G13
    plot_g14_lcoh_and_ledger(results, lcohs)                              # G14
    plot_g15_lcoh_breakdown(lcohs)                                        # G15
    plot_g16_energy_breakdown(results)                                    # G16
    return sweeps


def _ess_sizing_key(out: dict | None) -> str:
    if not out:
        return "-"
    df = out.get("cap_sweep")
    if isinstance(df, pd.DataFrame) and len(df) and "LCOH_internal" in df.columns:
        k = int(df["LCOH_internal"].idxmin())
        if "q [%]" in df.columns:
            return f"q = {df.loc[k, 'q [%]']:g} % (일별 잉여 분위수)"
        if "hours_of_P*" in df.columns:
            return f"{df.loc[k, 'hours_of_P*']:.2f} h of P*"
    ch = out.get("cap_hours")
    return f"{float(ch):.2f} h of P*" if ch is not None else "-"


def ess_summary_table(out2: dict | None, out3: dict | None) -> pd.DataFrame:
    rows = [{"case": "Case 1",
             "ESS 용량 E_rated [kWh]": 0.0,
             "ESS 용량 [MWh]": 0.0,
             "등가 저장시간 [h of P*]": 0.0,
             "ESS 출력 P_rated [kW]": 0.0,
             "ESS CAPEX [USD]": 0.0,
             "등가 사이클 [cyc/yr]": 0.0,
             "운전 j* [A/cm2]": float("nan"),
             "사이징 근거": "ESS 없음 (직결)"}]

    for case, out in (("Case 2", out2), ("Case 3", out3)):
        if not out:
            continue
        r = out.get("result")
        E = float(out.get("E_rated", getattr(r, "E_rated", 0.0)) or 0.0)
        P_star = float((out.get("op") or {}).get("P_star", 0.0) or 0.0)
        if P_star <= 0:
            P_star = float(getattr(r, "P_star", 0.0) or 0.0)
        P_rated = max(float(getattr(r, "P_rated_chg", 0.0) or 0.0),
                      float(getattr(r, "P_rated_dis", 0.0) or 0.0))
        rows.append({
            "case": case,
            "ESS 용량 E_rated [kWh]": E,
            "ESS 용량 [MWh]": E / 1e3,
            "등가 저장시간 [h of P*]": E / P_star if P_star > 0 else float("nan"),
            "ESS 출력 P_rated [kW]": P_rated,
            "ESS CAPEX [USD]": float(out.get("capex_ess", 0.0) or 0.0),
            "등가 사이클 [cyc/yr]": float(getattr(r, "eq_cycles", 0.0) or 0.0),
            "운전 j* [A/cm2]": float(out.get("j_star", float("nan"))),
            "사이징 근거": _ess_sizing_key(out),
        })
    return pd.DataFrame(rows).set_index("case")


def print_ess_summary(out2: dict | None, out3: dict | None,
                      title: str = "ESS 용량 최적화 결과 (E_rated*)",
                      save_csv: bool | str = False) -> pd.DataFrame:
    df = ess_summary_table(out2, out3)
    _banner(title, "-")
    print(f"  mode={ESS_SIZING_MODE}, c_E={C_E_USD_KWH} $/kWh, "
          f"c_P={C_P_USD_KW} $/kW, RTE={ETA_RTE:.2f}")
    cols = ["ESS 용량 E_rated [kWh]", "ESS 출력 P_rated [kW]",
            "ESS CAPEX [USD]", "등가 사이클 [cyc/yr]", "운전 j* [A/cm2]", "사이징 근거"]
    print(df[cols].to_string(float_format=lambda x: f"{x:,.2f}"))
    e2 = df.loc["Case 2", "ESS 용량 E_rated [kWh]"] if "Case 2" in df.index else np.nan
    e3 = df.loc["Case 3", "ESS 용량 E_rated [kWh]"] if "Case 3" in df.index else np.nan
    if np.isfinite(e2) and np.isfinite(e3) and e2 > 0:
        print(f"  Case3/Case2 E_rated={e3 / e2 * 100:,.1f} % "
              f"(reduction={100 - e3 / e2 * 100:,.1f} %)")
        if e3 >= e2:
            print("  [CHECK] Case 3 E_rated >= Case 2; q/cap grid boundary 확인 필요")
    if save_csv:
        try:
            if isinstance(save_csv, str):
                path = save_csv
            else:
                d = _ensure_fig_dir()
                path = f"{d}/ESS_sizing_summary.csv" if d else "ESS_sizing_summary.csv"
            df.to_csv(path, encoding="utf-8-sig")
            print(f"  saved: {path}")
        except Exception as e:
            print(f"  [WARN] ESS summary CSV save failed: {e}")
    return df


def print_final_summary(profile, fixed_stack, results: dict, lcohs: dict,
                        out1: dict, out2: dict, out3: dict, sweeps: dict | None = None):
    _banner("최종 요약")
    cap_mw = profile.meta.get("re_capacity_MW", np.nan)
    cf_txt = (f", CF={profile.meta['unit_CF'] * 100:,.2f} %"
              if "unit_CF" in profile.meta else "")
    print(f"  input={profile.region} {profile.year}, "
          f"RE={cap_mw:,.2f} MW" if np.isfinite(cap_mw) else
          f"  input={profile.region} {profile.year}, RE=(unspecified)")
    print(f"  E_paid={profile.P_in.sum() / 1e3:,.1f} MWh{cf_txt}, "
          f"PEM={fixed_stack.P_nameplate / 1000:,.3f} MW, "
          f"j_window={J_MIN:.3f}-{J_MAX:.2f} A/cm2, "
          f"elec={ELEC_PRICE:,.4f} $/kWh")

    rows = []
    for case, out in [("Case 1", out1), ("Case 2", out2), ("Case 3", out3)]:
        if case not in results:
            continue
        r = results[case]
        h = _production_hours(r, r.dt or 1.0)
        lc = lcohs[case]
        j_star = out.get("j_star")
        sw = (sweeps or {}).get(case)
        j_opt = float(sw.loc[sw["LCOH"].idxmin(), "j"]) if sw is not None and len(sw) else np.nan
        rows.append({
            "case": case,
            "수소 생산시간 [h]": h["producing"],
            "정지(발전有) [h]": h["idle_with_gen"],
            "연간 H2 [t]": r.H2_total / 1e3,
            "평균 생산율 [kg/h]": r.H2_total / h["producing"] if h["producing"] else np.nan,
            "운전 j* [A/cm2]": j_star if j_star is not None else J_MAX,
            "LCOH 최소 j [A/cm2]": j_opt,
            "ESS 용량 [kWh]": r.E_rated,
            "ESS 출력 [kW]": max(r.P_rated_chg, r.P_rated_dis),
            "ESS CAPEX [k$]": lc["capex_ess"] / 1e3,
            "SEC_eff [kWh/kg]": r.SEC_eff(),
            "curtail [%]": r.E_curtail / max(r.E_paid, 1e-9) * 100,
            "LCOH [$/kg]": lc["LCOH"],
        })
    df = pd.DataFrame(rows).set_index("case").T
    print("\n[case metrics]")
    print(df.to_string(float_format=lambda x: f"{x:,.2f}"))

    items = pd.DataFrame({case: lcohs[case]["items_usd_per_kg"] for case in results}).round(4)
    items.index = [_LCOH_ITEM_LABELS.get(i, i) for i in items.index]
    items.loc["TOTAL"] = [lcohs[case]["LCOH"] for case in results]
    print("\n[LCOH breakdown, $/kg]")
    print(items.to_string(float_format=lambda x: f"{x:,.3f}"))

    ess = ess_summary_table(out2, out3)
    ecols = ["ESS 용량 E_rated [kWh]", "ESS 출력 P_rated [kW]",
             "ESS CAPEX [USD]", "등가 사이클 [cyc/yr]", "사이징 근거"]
    print(f"\n[ESS sizing] mode={ESS_SIZING_MODE}")
    print(ess[ecols].to_string(float_format=lambda x: f"{x:,.2f}"))

    win = min(((case, lcohs[case]["LCOH"]) for case in results), key=lambda kv: kv[1])
    best_h2 = max(((case, results[case].H2_total) for case in results), key=lambda kv: kv[1])
    print(f"\n[결론] LCOH 최저={win[0]} ({win[1]:,.3f} $/kg), "
          f"생산량 최대={best_h2[0]} ({best_h2[1] / 1e3:,.2f} t/yr)")
    return df


def _banner(title: str, ch: str = "=") -> None:
    print("\n" + ch * 78)
    print(f" {title}")
    print(ch * 78)


def echo_config(profile, fixed_stack, win) -> None:
    _banner("공유 파라미터 요약 (PART 1)", "-")
    print(f"  catalyst={CATALYST_NAME}, T={T_OPER - 273.15:.0f} C, "
          f"membrane={MEMB_T_UM:.0f} um, BOP={BOP_EFFICIENCY:.2f} kWh/kg")
    print(f"  stack={fixed_stack.P_nameplate:,.1f} kW, "
          f"A_tot={fixed_stack.A_tot:,.0f} cm2, j={J_MIN:.4f}-{J_MAX:.2f} A/cm2, "
          f"P={win.P_min:,.0f}-{win.P_max:,.0f} kW")
    print(f"  RE={profile.region} {profile.year}, E_paid={profile.E_paid / 1e3:,.1f} MWh, "
          f"n={profile.allocation_n * 100:.0f} %, elec={ELEC_PRICE} $/kWh, "
          f"ESS(c_E/c_P/RTE)={C_E_USD_KWH}/{C_P_USD_KW}/{ETA_RTE:.2f}")


def setup_common(verbose: bool = True):
    _banner("공통 셋업 — 3케이스가 함께 쓰는 부분 (여기서 갈리는 건 없다)")
    profile = get_profile()
    if verbose:
        profile.report()
        report_catalyst()
        report_cell_voltage()
        report_kernel()
    fixed_stack = build_fixed_stack(profile, verbose=verbose)
    win = operating_window(fixed_stack.A_tot)
    if verbose:
        print("\n[운전 창]")
        print(j_min_derivation())
        print(win.describe())
        get_inverter(fixed_stack.A_tot, win.j_min, win.j_max).validate()
        echo_config(profile, fixed_stack, win)
    return profile, fixed_stack, win


def _interior_j_case2(out2) -> bool:
    opt = out2.get("opt")
    if not opt or opt.get("j_sweep") is None or not len(opt["j_sweep"]):
        return False
    d = opt["j_sweep"]
    k = int(d["LCOH_internal"].idxmin())
    return 0 < k < len(d) - 1


def _interior_j_case3(out3) -> bool:
    opt = out3.get("opt")
    if not opt or opt.get("best") is None:
        return False
    jg = opt["j_grid"]
    return jg.min() < opt["best"]["j"] < jg.max()


def completion_checks(res1, out2, out3, inv_err: float) -> pd.DataFrame:
    r2, r3 = out2["result"], out3["result"]
    checks = [
        ("P1 P<->j 역산 왕복오차 < 1e-6", inv_err < INVERSION_TOL, f"{inv_err:.2e}"),
        ("P3 Case 1 에너지수지 < 1 %", res1.balance_error() < BALANCE_TOL,
         f"{res1.balance_error() * 100:.4f} %"),
        ("P3 Case 2 에너지수지 < 1 %", r2.balance_error() < BALANCE_TOL,
         f"{r2.balance_error() * 100:.4f} %"),
        ("P3 Case 3 에너지수지 < 1 %", r3.balance_error() < BALANCE_TOL,
         f"{r3.balance_error() * 100:.4f} %"),
    ]
    j2v = float(out2.get("j_star_own", out2["j_star"]))
    j3v = float(out3.get("j_star_own", out3["j_star"]))
    if j_is_fixed():
        checks += [
            ("C2 지정 j가 운전 창 내부", J_MIN <= j2v <= J_MAX, f"j={j2v:.4f} (고정 운전)"),
            ("C3 지정 j가 운전 창 내부", J_MIN <= j3v <= J_MAX, f"j={j3v:.4f} (고정 운전)"),
        ]
    else:
        checks += [
            ("C2 j*가 경계가 아닌 내부에 형성", _interior_j_case2(out2), f"j*={j2v:.4f}"),
            ("C3 j*가 경계가 아닌 내부에 형성", _interior_j_case3(out3), f"j*={j3v:.4f}"),
        ]
    checks += [
        ("C3 E_rated* < C2 E_rated*  ★핵심명제", out3["E_rated"] < out2["E_rated"],
         f"{out3['E_rated']:,.0f} vs {out2['E_rated']:,.0f} kWh"),
        ("C3 E_rte < C2 E_rte", r3.E_rte < r2.E_rte,
         f"{r3.E_rte / 1e3:,.1f} vs {r2.E_rte / 1e3:,.1f} MWh"),
        ("C3 CAPEX_ESS < C2 CAPEX_ESS", out3["capex_ess"] < out2["capex_ess"],
         f"{out3['capex_ess']:,.0f} vs {out2['capex_ess']:,.0f} USD"),
        ("C3 eq_cycles < C2 eq_cycles", r3.eq_cycles < r2.eq_cycles,
         f"{r3.eq_cycles:,.0f} vs {r2.eq_cycles:,.0f} cyc/yr"),
        ("C2 히스테리시스 밴드 >= 1 h", r2.meta["min_run_duration_h"] >= 1.0,
         f"{r2.meta['min_run_duration_h']:.2f} h"),
        ("C3 히스테리시스 밴드 >= 1 h", r3.meta["min_run_duration_h"] >= 1.0,
         f"{r3.meta['min_run_duration_h']:.2f} h"),
    ]
    return pd.DataFrame([{"완료조건": label, "판정": "PASS" if ok else "CHECK", "값": v}
                         for label, ok, v in checks])


def main(make_plots: bool = True, run_case1_sweep: bool = True,
         run_hysteresis: bool = True) -> dict:
    profile, fixed_stack, win = setup_common()
    inv_err = get_inverter(fixed_stack.A_tot, win.j_min, win.j_max).max_roundtrip_error

    _banner("케이스별 독립 실행 — 각 케이스를 '자기 최적 운전점'에서 돌린다")
    out1 = run_case1(profile, fixed_stack)
    rating_df = (stack_rating_sweep_case1(profile, base_nameplate=fixed_stack.P_nameplate)
                 if run_case1_sweep else None)
    out2 = run_case2(profile, fixed_stack)
    hyst_df = (hysteresis_sweep_case2(profile, fixed_stack, out2["j_star"],
                                      out2["E_rated"]) if run_hysteresis else None)
    out3 = run_case3(profile, fixed_stack)

    j_manual = None
    if str(HEADLINE_J_MODE).strip().lower() == "manual":
        if J_STAR_MANUAL is None:
            raise ValueError("HEADLINE_J_MODE='manual' 인데 J_STAR_MANUAL 이 None 입니다. "
                             "숫자를 넣거나 HEADLINE_J_MODE='own' 으로 두세요.")
        j_manual = float(min(max(float(J_STAR_MANUAL), J_MIN), J_MAX))
        if abs(j_manual - float(J_STAR_MANUAL)) > 1e-12:
            print(f"  [경고] J_STAR_MANUAL = {float(J_STAR_MANUAL):.4f} 가 운전 창 "
                  f"[{J_MIN:.4f}, {J_MAX:.4f}] 밖입니다 -> {j_manual:.4f} 로 클램프합니다.")
        _banner(f"운전점 통일 — 지정 j = {j_manual:.4f} A/cm2 로 Case 2·3 재실행")
        print("  자체 최적 j*는 'j*(자체최적)' 열에 보존됩니다.")
        opt2_A, opt3_A = out2.get("opt"), out3.get("opt")
        j2_A, j3_A = out2["j_star"], out3["j_star"]
        out2 = run_case2(profile, fixed_stack, j_star=j_manual, optimize=False)
        out3 = run_case3(profile, fixed_stack, j_star=j_manual, optimize=False)
        out2["opt"], out2["j_star_own"] = opt2_A, j2_A
        out3["opt"], out3["j_star_own"] = opt3_A, j3_A

    _banner("헤드라인 운전점")
    print(pd.DataFrame([
        {"case": "Case 1", "j*": "N/A (전력 추종)",
         "j*(자체최적)": "N/A", "E_rated [kWh]": 0.0,
         "LCOH [$/kg]": out1["lcoh"]["LCOH"]},
        {"case": "Case 2", "j*": f"{out2['j_star']:.4f}",
         "j*(자체최적)": f"{out2.get('j_star_own', out2['j_star']):.4f}",
         "E_rated [kWh]": out2["E_rated"], "LCOH [$/kg]": out2["lcoh"]["LCOH"]},
        {"case": "Case 3", "j*": f"{out3['j_star']:.4f}",
         "j*(자체최적)": f"{out3.get('j_star_own', out3['j_star']):.4f}",
         "E_rated [kWh]": out3["E_rated"], "LCOH [$/kg]": out3["lcoh"]["LCOH"]},
    ]).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n  HEADLINE_J_MODE='{HEADLINE_J_MODE}'"
          + (" -> own j*" if j_manual is None else f" -> manual j={j_manual:.4f}"))

    j_headline = {"Case 2": float(out2["j_star"]), "Case 3": float(out3["j_star"])}
    ess_df = print_ess_summary(out2, out3,
                               title="ESS 용량 최적화 결과 — 헤드라인",
                               save_csv=True)

    results = {"Case 1": out1["result"], "Case 2": out2["result"],
               "Case 3": out3["result"]}
    lcohs = {"Case 1": out1["lcoh"], "Case 2": out2["lcoh"], "Case 3": out3["lcoh"]}

    _banner("결론 비교표")
    cmp_df = compare_table(results, lcohs)
    print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    best = min(lcohs, key=lambda k: lcohs[k]["LCOH"])
    print(f"\n  >>> 최저 LCOH : {best}  ({lcohs[best]['LCOH']:.3f} USD/kg-H2)")

    print("\n[손실 원장 3케이스 대조]")
    led = pd.DataFrame({name: {k: v / r.E_paid * 100 for k, v in r.ledger.items()}
                        for name, r in results.items()}).T
    led.columns = [f"{bucket} [%]" for bucket in led.columns]
    print(led.to_string(float_format=lambda x: f"{x:,.2f}"))

    _banner("완료조건 체크리스트")
    chk = completion_checks(out1["result"], out2, out3, inv_err)
    print(chk.to_string(index=False))
    n_fail = int((chk["판정"] != "PASS").sum())
    print(f"\n  PASS {len(chk) - n_fail}/{len(chk)}  "
          + ("모든 완료조건 통과" if n_fail == 0
             else "CHECK 항목은 파라미터 조정 대상 (버그 아님)"))

    sweeps = {}
    if make_plots and plt is not None:
        _banner("그래프 생성 (G01 ~ G16)")
        sweeps = make_all_figures(profile, results, lcohs, fixed_stack,
                                  out1, out2, out3)
        if PLOT_DETAIL_FIGURES:
            plot_case1(out1["result"], out1["window"], profile, fixed_stack, rating_df)
            plot_case2(out2, hyst_df)
            plot_case3(out3)
        print(f"  figures: {_ensure_fig_dir() or '(저장 안 함)'}")

    summary = print_final_summary(profile, fixed_stack, results, lcohs,
                                  out1, out2, out3, sweeps)

    return {"profile": profile, "stack": fixed_stack, "window": win,
            "case1": out1, "case2": out2, "case3": out3, "results": results,
            "lcohs": lcohs, "compare": cmp_df, "checks": chk, "summary": summary,
            "ess_summary": ess_df, "j_sweeps": sweeps,
            "rating_sweep": rating_df, "hysteresis": hyst_df,
            "j_headline": j_headline}


def sensitivity_stack_fraction(fractions=(0.25, 0.4, 0.55, 0.7, 1.0),
                               verbose: bool = True) -> pd.DataFrame:
    global STACK_PV_FRACTION
    profile = get_profile()
    backup, rows = STACK_PV_FRACTION, []
    try:
        for f in fractions:
            STACK_PV_FRACTION = f
            st = build_fixed_stack(profile, verbose=False)
            o1 = run_case1(profile, st, verbose=False)
            o2 = run_case2(profile, st, verbose=False)
            o3 = run_case3(profile, st, verbose=False)
            lc = {"C1": o1["lcoh"]["LCOH"], "C2": o2["lcoh"]["LCOH"],
                  "C3": o3["lcoh"]["LCOH"]}
            rows.append({"PV fraction": f, "nameplate [kW]": st.P_nameplate,
                         "C1 curtail [%]": o1["result"].E_curtail / o1["result"].E_paid * 100,
                         "C1 H2 [t]": o1["result"].H2_total / 1e3,
                         "C2 H2 [t]": o2["result"].H2_total / 1e3,
                         "C3 H2 [t]": o3["result"].H2_total / 1e3,
                         "C1 LCOH": lc["C1"], "C2 LCOH": lc["C2"], "C3 LCOH": lc["C3"],
                         "winner": min(lc.items(), key=lambda kv: kv[1])[0]})
            if verbose:
                print(f"  fraction {f:.2f}: nameplate {st.P_nameplate:,.0f} kW done")
    finally:
        STACK_PV_FRACTION = backup
    df = pd.DataFrame(rows)
    if verbose:
        _banner("민감도 — 스택 규모 vs 재생 규모")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    return df


# ###################################################################################
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    _out = main()

# %%
