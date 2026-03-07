import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize # 자동 최적화를 위한 라이브러리 추가

# ==========================================
# 0. 웹 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="hWTF Recharge Calculator", layout="wide")

# ==========================================
# 1. hWTF 계산 클래스
# ==========================================
class hWTF_Recharge_Calculator:
    def __init__(self, soil_type_idx, k, r_cr_input, h_max, verbose=False):
        self.k = float(k)
        self.r_cr_input = float(r_cr_input)
        self.h_max = float(h_max)
        self.verbose = bool(verbose)
        self.time_dry = 1 # 기본값 설정 후 내부에서 자동 계산됨

        # 12가지 토양 물성 DB (고정)
        self.soil_db = [
            [0.43, 0.045, 14.5, 2.68, 7.128], [0.41, 0.065, 7.5,  1.89, 1.061],
            [0.41, 0.057, 12.4, 2.28, 3.05],  [0.45, 0.067, 2.0,  1.41, 0.108],
            [0.46, 0.034, 1.6,  1.37, 0.06],  [0.38, 0.068, 0.8,  1.09, 0.048],
            [0.36, 0.070, 0.5,  1.09, 0.0048],[0.38, 0.100, 2.7,  1.23, 0.0288],
            [0.43, 0.089, 1.0,  1.23, 0.0168],[0.41, 0.095, 1.9,  1.31, 0.0624],
            [0.39, 0.100, 5.9,  1.48, 0.3144],[0.43, 0.078, 3.6,  1.56, 0.2496],
        ]
        self.theta_s, self.theta_r, self.alpha, self.n, self.Ks = self.soil_db[soil_type_idx]
        self.m = 1 - (1 / self.n)

    def _read_dataframe(self, df):
        if df.shape[1] < 2:
            raise ValueError("CSV는 최소 2개(강수량, 지하수위) 또는 3개(날짜, 강수량, 지하수위)의 컬럼이 필요합니다.")
        x = np.arange(len(df))
        if df.shape[1] >= 3:
            dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            if dt.notna().mean() > 0.8:
                x = dt
                return x, df.iloc[:, 1].astype(float).values, df.iloc[:, 2].astype(float).values
        return x, df.iloc[:, 0].astype(float).values, df.iloc[:, 1].astype(float).values

    def _prepare_units_and_gwl(self, P_in, H_in):
        P_in = np.asarray(P_in, dtype=float)
        rainfall_is_meter = (np.nanmax(P_in) <= 1.0 and np.nanmedian(P_in) <= 0.2)
        if rainfall_is_meter:
            P_m, P_mm = P_in, P_in * 1000.0
        else:
            P_mm, P_m = P_in, P_in / 1000.0

        H_in = np.asarray(H_in, dtype=float)
        min_val = np.nanmin(H_in)
        H_calc = H_in + (min_val - 1) * -1 if min_val < 0 else H_in
        return P_mm, P_m, H_calc

    def _vg(self, v, x):
        return v / ((1 + (self.alpha * x) ** self.n) ** self.m)

    def _quad_vba(self, v, c1, c3, f1, f2, f3):
        h = c3 - c1
        c2, x1, x2 = (c3 + c1) / 2, (c1 + c2) / 2, (c2 + c3) / 2
        y1, y2 = self._vg(v, x1), self._vg(v, x2)
        q1 = (h / 6) * (f1 + 4 * f2 + f3)
        q = (h / 12) * (f1 + 4 * y1 + 2 * f2 + 4 * y2 + f3)
        return q + (q - q1) / 15

    def _integral_piecewise_vba(self, v, dh, wet_event=True):
        if wet_event:
            dh_use = max(dh, 0.01)
            h = 0.13579 * dh_use
            x3, x4, x5, x7 = 2 * h, dh_use / 2, dh_use - 2 * h, dh_use
        else:
            h = 0.0013579
            x3, x4, x5, x7 = 2 * h, 0.005, 0.01 - 2 * h, 0.01
            dh_use = 0.01

        vg1, vg2, vg3 = self._vg(v, 0.0), self._vg(v, h), self._vg(v, x3)
        vg4, vg5 = self._vg(v, x4), self._vg(v, x5)
        vg6, vg7 = self._vg(v, dh_use - h), self._vg(v, x7)

        q2 = self._quad_vba(v, 0.0, x3, vg1, vg2, vg3) + \
             self._quad_vba(v, x3, x5, vg3, vg4, vg5) + \
             self._quad_vba(v, x5, x7, vg5, vg6, vg7)
        return (v * dh_use - q2) / dh_use if wet_event else (v * dh_use - q2) * 100

    # 연산 속도 향상을 위해 메인 루프 분리
    def run_simulation(self, P_mm, P_m, H_calc):
        # r_cr_input 처리
        r_cr_mm = self.r_cr_input * 1000.0 if (self.r_cr_input < 1.0 and np.nanmax(P_mm) > 1.0) else self.r_cr_input
        days = len(H_calc)
        rech = np.zeros(days)
        current_dry_days = int(self.time_dry)
        expk = np.exp(self.k)
        ns_sum, nr_count = 0.0, 0

        for i in range(days - 1):
            dh = H_calc[i + 1] - H_calc[i]
            ths = 0.5 ** self.m
            wet_event = (P_mm[i] - r_cr_mm) > 0

            for _ in range(1000):
                g = 1 - ths ** (1 / self.m)
                qt = (self.Ks * (self.n - 1) / 2 / self.h_max * g * (1 - g ** self.m) * (1 - g) ** (self.m / 2) * (1 + 4 * g ** (self.m - 1) - 5 * g ** self.m))
                ths = ths - (qt / 1000 * current_dry_days)

            th_tr = self.theta_s * ths + self.theta_r * (1 - ths)
            v = self.theta_s - th_tr
            v_final = float(np.clip(self._integral_piecewise_vba(v, dh, wet_event=wet_event), 0.0, max(self.theta_s - self.theta_r, 1e-9)))

            if (P_mm[i] - r_cr_mm) < 0:
                current_dry_days += 1
            else:
                current_dry_days = 1

            ns_sum += v_final
            if wet_event: nr_count += 1

            mn, i2 = min(i + 2, days - 1), max(i - 1, 0)
            h_min1, h_min2 = (H_calc[i] + H_calc[mn]) / 2.0, (H_calc[i2] + H_calc[i + 1]) / 2.0
            term_num = (H_calc[i + 1] - h_min1) - (H_calc[i] - h_min2) * expk

            rech[i] = v_final * self.k * (term_num / (expk - 1)) if P_mm[i] > 0 else 0.0

        total_rech = np.sum(rech[rech > 0])
        total_rain = np.sum(P_m)
        rate = (total_rech / total_rain) * 100 if total_rain > 0 else 0.0

        avg_v = (ns_sum / nr_count) if nr_count > 0 else 1.0
        param_fn = ((total_rech / total_rain) / avg_v) if avg_v > 0 and total_rain > 0 else 0.0

        h_min = np.min(H_calc)
        H_sim = np.zeros(days)
        H_sim[0] = H_calc[0] - h_min

        for i in range(days - 1):
            H_sim[i + 1] = H_sim[i] * expk + (P_m[i] * param_fn / self.k) * (expk - 1)

        return t_rain, t_rech, rate, H_sim + h_min

# ==========================================
# 2. UI 구성 (스트림릿 화면)
# ==========================================
st.title("🌱 Hybrid hWTF 지하수 함양률 산정 모델 (자동 최적화 적용)")
st.markdown("관측 데이터와 입력 파라미터를 기반으로 함양률을 산정하며, **100% 초과 시 AI가 물리적 한계치 내로 자동 최적화를 수행합니다.**")

soil_names = ["0: Sand", "1: Sandy Loam", "2: Loamy Sand", "3: Silt Loam", "4: Silt", "5: Clay",
              "6: Silty Clay", "7: Sandy Clay", "8: Silty Clay Loam", "9: Clay Loam", "10: Sandy Clay Loam", "11: Loam"]

with st.sidebar:
    st.header("1. 데이터 업로드")
    sample_file_path = "data/hWTF_input.csv"
    
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as file:
            st.download_button(label="📥 샘플 양식 다운로드 (CSV)", data=file, file_name="sample_hWTF_input.csv", mime="text/csv")
            
    uploaded_file = st.file_uploader("본인의 CSV 파일 업로드 (선택)", type=["csv"])
    
    st.markdown("---")
    st.header("2. 초기 파라미터 설정")
    s_idx = st.selectbox("토양 종류 (Soil Type, 고정값)", range(12), format_func=lambda x: soil_names[x], index=0)
    k = st.number_input("초기 기저유출 감수상수 (k)", value=-0.1, step=0.01, format="%.3f")
    r_cr = st.number_input("초기 임계 강수량 (r_cr, mm/m)", value=5.0, step=0.5)
    h_max = st.number_input("초기 모세관대 두께 (h_max, m)", value=2.0, step=0.1)
    # 초기 무강우 일수(time_dry) 입력창은 데이터 기반 자동 연산을 위해 제거되었습니다.
    
    run_btn = st.button("🚀 함양률 계산 및 최적화 실행", type="primary", use_container_width=True)

# ==========================================
# 3. 데이터 로드 및 실행 로직
# ==========================================
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("✅ 업로드하신 데이터 파일로 분석을 진행합니다.")
else:
    if os.path.exists(sample_file_path):
        df = pd.read_csv(sample_file_path)
        st.info("💡 안내: 서버에 내장된 기본 샘플 데이터가 로드되어 있습니다.")
    else:
        st.warning("데이터 파일이 없습니다. 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")

if df is not None:
    calc = hWTF_Recharge_Calculator(s_idx, k, r_cr, h_max)
    try:
        x_raw, P_in, H_in = calc._read_dataframe(df)
        P_mm, P_m, H_calc = calc._prepare_units_and_gwl(P_in, H_in)
        
        # [자동 계산] 첫 비가 내리기 전까지의 초기 무강우 일수 탐색
        first_rain_idx = np.argmax(P_mm > 0)
        calc.time_dry = first_rain_idx + 1 if (first_rain_idx > 0 or P_mm[0] > 0) else len(P_mm)
        st.caption(f"ℹ️ 데이터 분석 결과, 초기 무강우 일수(time_dry)는 **{calc.time_dry}일**로 자동 설정되었습니다.")

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

    if run_btn:
        with st.spinner("초기 파라미터로 hWTF 연산을 수행 중입니다..."):
            t_rain, t_rech, t_rate, H_sim = calc.run_simulation(P_mm, P_m, H_calc)
            
            # --- 최적화 엔진 가동 (함양률 100% 초과 시) ---
            if t_rate > 100.0:
                st.warning(f"⚠️ 초기 산정된 함양률이 **{t_rate:.1f}%** 로 물리적 한계치(100%)를 초과했습니다. 자동 파라미터 최적화를 시작합니다...")
                
                with st.spinner("지하수위 오차를 최소화하며 함양률을 100% 이하로 제어하는 최적 파라미터를 찾는 중입니다 (약 10~20초 소요)..."):
                    def objective(params):
                        calc.k, calc.r_cr_input, calc.h_max = params
                        _, _, rate, h_s = calc.run_simulation(P_mm, P_m, H_calc)
                        
                        rmse = np.sqrt(np.mean((H_calc - h_s)**2))
                        penalty = max(0, rate - 99.0) * 1000 # 99% 넘으면 강력한 페널티 부과
                        return rmse + penalty
                    
                    # (k, r_cr, h_max) 탐색 경계 범위 설정
                    bounds = ((-0.5, -0.001), (0.0, 50.0), (0.1, 10.0))
                    initial_guess = [k, r_cr, h_max]
                    
                    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
                    
                    # 최적화된 파라미터 다시 클래스에 반영 후 최종 계산
                    calc.k, calc.r_cr_input, calc.h_max = res.x
                    t_rain, t_rech, t_rate, H_sim = calc.run_simulation(P_mm, P_m, H_calc)
                    
                    st.success(f"✨ 최적화 완료! 물리적으로 타당한 새로운 파라미터가 자동으로 적용되었습니다. \n\n"
                               f"👉 **수정된 파라미터:** k = {calc.k:.4f}, r_cr = {calc.r_cr_input:.2f}, h_max = {calc.h_max:.2f}")
            
            # --- 결과 출력 ---
            st.markdown("---")
            st.subheader("✅ 최종 산정 결과 (Results)")
            col1, col2, col3 = st.columns(3)
            col1.metric("총 강수량", f"{t_rain*1000:.1f} mm")
            col2.metric("총 함양량", f"{t_rech*1000:.1f} mm")
            col3.metric("지하수 함양률", f"{t_rate:.2f} %")
            
            st.subheader("📉 지하수위 관측치 vs 모의치 피팅 (Fitting)")
            fig2, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_raw, H_calc, label="Observed (Measured)", color="black", linewidth=1.2)
            ax.scatter(x_raw, H_sim, label="Calculated (Simulated)", marker="o", facecolors="none", edgecolors="olivedrab", linewidths=1.2, s=35)
            ax.set_title(f"GWL Comparison (Recharge Rate: {t_rate:.1f}%)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Groundwater Level (m)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig2)
