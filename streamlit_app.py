import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize 

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
        self.r_cr_input = float(r_cr_input) # 항상 mm로 취급
        self.h_max = float(h_max)
        self.verbose = bool(verbose)
        self.time_dry = 1 

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

    def _prepare_units_and_gwl(self, P_in, H_in):
        # 강수는 무조건 mm, 지하수는 무조건 m로 확정하여 계산 오차 원천 차단
        P_mm = np.asarray(P_in, dtype=float)
        P_m = P_mm / 1000.0
        r_cr_mm = float(self.r_cr_input)

        H_in = np.asarray(H_in, dtype=float)
        min_val = np.nanmin(H_in)
        H_calc = H_in + (min_val - 1) * -1 if min_val < 0 else H_in
        return P_mm, P_m, r_cr_mm, H_calc

    def _vg(self, v, x):
        return v / ((1 + (self.alpha * x) ** self.n) ** self.m)

    def _quad_vba(self, v, c1, c3, f1, f2, f3):
        h = c3 - c1
        # [에러 해결] 파이썬의 동시 할당 오류를 막기 위해 줄을 나누어 순차적으로 계산
        c2 = (c3 + c1) / 2.0
        x1 = (c1 + c2) / 2.0
        x2 = (c2 + c3) / 2.0
        y1, y2 = self._vg(v, x1), self._vg(v, x2)
        q1 = (h / 6.0) * (f1 + 4.0 * f2 + f3)
        q = (h / 12.0) * (f1 + 4.0 * y1 + 2.0 * f2 + 4.0 * y2 + f3)
        return q + (q - q1) / 15.0 # Romberg error correction (논문 일치)

    def _integral_piecewise_vba(self, v, dh, wet_event=True):
        if wet_event:
            dh_use = max(dh, 0.01)
            h = 0.13579 * dh_use
            x3, x4, x5, x7 = 2 * h, dh_use / 2.0, dh_use - 2 * h, dh_use
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

    def run_simulation(self, P_mm, P_m, r_cr_mm, H_calc):
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
                qt = (self.Ks * (self.n - 1) / 2.0 / self.h_max * g * (1 - g ** self.m) * (1 - g) ** (self.m / 2.0) * (1 + 4 * g ** (self.m - 1) - 5 * g ** self.m))
                ths = ths - (qt / 1000.0 * current_dry_days)

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

        return total_rain, total_rech, rate, H_sim + h_min

# ==========================================
# 2. UI 구성 (스트림릿 화면)
# ==========================================
st.title("🌱 Hybrid hWTF 지하수 함양률 산정 모델 (자동 최적화)")
st.markdown("관측 데이터와 입력 파라미터를 기반으로 함양률을 산정하며, **100% 초과 시 AI가 물리적 한계치 내로 자동 최적화를 수행합니다.**")

# 번호표 떼고 깔끔하게 수정된 토양 리스트
soil_names = ["Sand", "Sandy Loam", "Loamy Sand", "Silt Loam", "Silt", "Clay",
              "Silty Clay", "Sandy Clay", "Silty Clay Loam", "Clay Loam", "Sandy Clay Loam", "Loam"]

with st.sidebar:
    st.header("1. 데이터 업로드 방식")
    
    upload_mode = st.radio("업로드 방식을 선택하세요:", ["A. 통합 파일 1개 업로드", "B. 강수량 / 지하수위 개별 업로드"])
    
    df_merged = None
    
    if upload_mode == "A. 통합 파일 1개 업로드":
        st.caption("✔️ 파일 형식: [날짜, 강수량(mm), 지하수위(m)]")
        uploaded_file = st.file_uploader("통합 CSV 파일 업로드", type=["csv"])
        if uploaded_file:
            df_temp = pd.read_csv(uploaded_file)
            if df_temp.shape[1] >= 3:
                df_merged = pd.DataFrame()
                df_merged['Date'] = pd.to_datetime(df_temp.iloc[:, 0])
                df_merged['Rainfall'] = df_temp.iloc[:, 1].astype(float)
                df_merged['GWL'] = df_temp.iloc[:, 2].astype(float)
            else:
                st.error("통합 파일은 반드시 3개 이상의 열(날짜, 강수량, 지하수위)이 필요합니다.")
                
    else:
        st.caption("✔️ 날짜를 기준으로 컴퓨터가 두 파일을 자동 병합합니다.")
        rain_file = st.file_uploader("🌧️ 강수량 파일 (날짜, 강수량mm)", type=["csv"])
        gwl_file = st.file_uploader("💧 지하수위 파일 (날짜, 지하수위m)", type=["csv"])
        
        if rain_file and gwl_file:
            df_rain = pd.read_csv(rain_file).iloc[:, :2]
            df_gwl = pd.read_csv(gwl_file).iloc[:, :2]
            
            df_rain.columns = ['Date', 'Rainfall']
            df_gwl.columns = ['Date', 'GWL']
            
            df_rain['Date'] = pd.to_datetime(df_rain['Date'])
            df_gwl['Date'] = pd.to_datetime(df_gwl['Date'])
            
            # 날짜를 기준으로 두 데이터 병합 (Inner Join)
            df_merged = pd.merge(df_rain, df_gwl, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
            st.success(f"두 파일이 날짜 기준으로 완벽하게 병합되었습니다! (총 {len(df_merged)}일 데이터)")

    st.markdown("---")
    st.header("2. 초기 파라미터 설정")
    s_idx = st.selectbox("토양 종류 (Soil Type, 고정값)", range(12), format_func=lambda x: soil_names[x], index=0)
    k = st.number_input("초기 기저유출 감수상수 (k)", value=-0.1, step=0.01, format="%.3f")
    r_cr = st.number_input("초기 임계 강수량 (r_cr, mm)", value=5.0, step=0.5)
    h_max = st.number_input("초기 모세관대 두께 (h_max, m)", value=2.0, step=0.1)
    
    run_btn = st.button("🚀 함양률 계산 및 최적화 실행", type="primary", use_container_width=True)

# ==========================================
# 3. 데이터 실행 로직
# ==========================================
if df_merged is not None:
    calc = hWTF_Recharge_Calculator(s_idx, k, r_cr, h_max)
    try:
        x_raw = df_merged['Date'].values
        P_mm, P_m, r_cr_mm, H_calc = calc._prepare_units_and_gwl(df_merged['Rainfall'].values, df_merged['GWL'].values)
        
        # [자동 계산] 최대 연속 무강우 일수(Maximum Dry Days)
        is_dry = (P_mm <= 0)
        max_dry_days, current_dry = 0, 0
        for dry in is_dry:
            if dry:
                current_dry += 1
                if current_dry > max_dry_days: max_dry_days = current_dry
            else:
                current_dry = 0
                
        calc.time_dry = max_dry_days if max_dry_days > 0 else 1
        st.caption(f"ℹ️ 데이터 스캔 결과, 최대 연속 무강우 일수(time_dry)는 **{calc.time_dry}일**로 자동 설정되었습니다.")
        
        # 프리뷰 그래프 그리기
        st.subheader("📊 입력 데이터 사전 점검 (Preview)")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(x_raw, H_calc, linewidth=1.2, label="GWL (m)", color="C0")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Groundwater Level (m)", color="C0")
        ax1.tick_params(axis='y', labelcolor="C0")

        ax2 = ax1.twinx()
        ax2.bar(x_raw, P_mm, alpha=0.35, width=0.8, label="Rainfall (mm)", color="C1")
        ax2.set_ylabel("Rainfall (mm)", color="C1")
        ax2.tick_params(axis='y', labelcolor="C1")
        ax2.invert_yaxis() 

        fig1.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1))
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

    if run_btn:
        with st.spinner("초기 파라미터로 hWTF 연산을 수행 중입니다..."):
            t_rain, t_rech, t_rate, H_sim = calc.run_simulation(P_mm, P_m, r_cr_mm, H_calc)
            
            # --- 최적화 엔진 가동 ---
            if t_rate > 100.0:
                st.warning(f"⚠️ 산정된 함양률이 **{t_rate:.1f}%** 로 물리적 한계치(100%)를 초과했습니다. 자동 파라미터 최적화를 시작합니다...")
                
                with st.spinner("지하수위 오차를 최소화하며 함양률을 100% 이하로 제어하는 최적 파라미터를 찾는 중입니다 (약 10초 소요)..."):
                    def objective(params):
                        calc.k, calc.r_cr_input, calc.h_max = params
                        # 최적화 중에도 mm 단위를 유지하도록 r_cr_mm 변수를 갱신하여 주입
                        r_cr_mm_opt = float(calc.r_cr_input)
                        _, _, rate, h_s = calc.run_simulation(P_mm, P_m, r_cr_mm_opt, H_calc)
                        
                        rmse = np.sqrt(np.mean((H_calc - h_s)**2))
                        penalty = max(0, rate - 99.0) * 1000 
                        return rmse + penalty
                    
                    bounds = ((-0.5, -0.001), (0.0, 50.0), (0.1, 10.0))
                    initial_guess = [k, r_cr, h_max]
                    
                    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
                    
                    calc.k, calc.r_cr_input, calc.h_max = res.x
                    r_cr_mm_final = float(calc.r_cr_input)
                    t_rain, t_rech, t_rate, H_sim = calc.run_simulation(P_mm, P_m, r_cr_mm_final, H_calc)
                    
                    st.success(f"✨ 최적화 완료! 물리적으로 타당한 새로운 파라미터가 자동으로 적용되었습니다. \n\n"
                               f"👉 **수정된 파라미터:** k = {calc.k:.4f}, r_cr = {calc.r_cr_input:.2f} mm, h_max = {calc.h_max:.2f} m")
            
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
else:
    st.info("👈 왼쪽 사이드바에서 분석할 데이터를 업로드해 주세요.")
