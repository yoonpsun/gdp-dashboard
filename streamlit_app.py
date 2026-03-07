import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 웹 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="hWTF Recharge Calculator", layout="wide")

# ==========================================
# 1. hWTF 계산 클래스
# ==========================================
class hWTF_Recharge_Calculator:
    def __init__(self, soil_type_idx, k, r_cr_input, h_max, time_dry_init, verbose=False):
        self.k = float(k)
        self.r_cr_input = float(r_cr_input)
        self.h_max = float(h_max)
        self.time_dry = int(time_dry_init)
        self.verbose = bool(verbose)

        # 12가지 토양 물성 DB
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
                rain = df.iloc[:, 1].astype(float).values
                gwl = df.iloc[:, 2].astype(float).values
                return x, rain, gwl

        rain = df.iloc[:, 0].astype(float).values
        gwl = df.iloc[:, 1].astype(float).values
        return x, rain, gwl

    def _prepare_units_and_gwl(self, P_in, H_in):
        P_in = np.asarray(P_in, dtype=float)
        P_max = np.nanmax(P_in)
        P_med = np.nanmedian(P_in)
        rainfall_is_meter = (P_max <= 1.0 and P_med <= 0.2)

        if rainfall_is_meter:
            P_m, P_mm = P_in, P_in * 1000.0
        else:
            P_mm, P_m = P_in, P_in / 1000.0

        if self.r_cr_input < 1.0 and not rainfall_is_meter:
            r_cr_mm = self.r_cr_input * 1000.0
        else:
            r_cr_mm = self.r_cr_input if not rainfall_is_meter else self.r_cr_input * 1000.0

        H_in = np.asarray(H_in, dtype=float)
        min_val = np.nanmin(H_in)
        H_calc = H_in + (min_val - 1) * -1 if min_val < 0 else H_in
        return P_mm, P_m, r_cr_mm, H_calc

    def _vg(self, v, x):
        return v / ((1 + (self.alpha * x) ** self.n) ** self.m)

    def _quad_vba(self, v, c1, c3, f1, f2, f3):
        h = c3 - c1
        c2 = (c3 + c1) / 2
        x1 = (c1 + c2) / 2
        x2 = (c2 + c3) / 2
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

    def calculate_recharge(self, df):
        x, P_in, H_in = self._read_dataframe(df)
        P_mm, P_m, r_cr_mm, H_calc = self._prepare_units_and_gwl(P_in, H_in)

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

        return x, P_mm, total_rain, total_rech, rate, H_calc, H_sim + h_min

# ==========================================
# 2. UI 구성 (스트림릿 화면)
# ==========================================
st.title("🌱 hWTF 기반 지하수 함양률 산정 모델")
st.markdown("관측된 강수량과 지하수위 데이터를 바탕으로 **Heuristic Water-Table Fluctuation (hWTF)** 알고리즘을 수행합니다.")

soil_names = ["0: Sand", "1: Sandy Loam", "2: Loamy Sand", "3: Silt Loam", "4: Silt", "5: Clay",
              "6: Silty Clay", "7: Sandy Clay", "8: Silty Clay Loam", "9: Clay Loam", "10: Sandy Clay Loam", "11: Loam"]

with st.sidebar:
    st.header("1. 데이터 업로드")
    
    # 샘플 데이터 다운로드 버튼 (깃허브에 hWTF_input.csv가 있어야 작동함)
    sample_file_path = "hWTF_input.csv"
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as file:
            st.download_button(
                label="📥 샘플 양식 다운로드 (CSV)",
                data=file,
                file_name="sample_hWTF_input.csv",
                mime="text/csv",
                help="이 양식에 맞춰 본인의 데이터를 작성 후 업로드하세요."
            )
            
    uploaded_file = st.file_uploader("본인의 CSV 파일 업로드 (선택)", type=["csv"])
    
    st.markdown("---")
    st.header("2. 모델 파라미터 설정")
    s_idx = st.selectbox("토양 종류 (Soil Type)", range(12), format_func=lambda x: soil_names[x], index=0)
    k = st.number_input("기저유출 감수상수 (k)", value=-0.1, step=0.01, format="%.3f")
    r_cr = st.number_input("임계 강수량 (r_cr, mm/m)", value=5.0, step=0.5)
    h_max = st.number_input("최대 모세관대 두께 (h_max, m)", value=2.0, step=0.1)
    time_dry = st.number_input("초기 무강우 일수 (time_dry, day)", value=3, step=1)
    
    run_btn = st.button("🚀 함양률 계산 실행", type="primary", use_container_width=True)

# ==========================================
# 3. 데이터 로드 로직 (기본 파일 vs 업로드 파일)
# ==========================================
df = None

if uploaded_file is not None:
    # 사용자가 직접 파일을 올린 경우
    df = pd.read_csv(uploaded_file)
    st.info("✅ 업로드하신 데이터 파일로 분석을 진행합니다.")
else:
    # 사용자가 파일을 올리지 않은 경우 -> 기본 파일 불러오기
    if os.path.exists(sample_file_path):
        df = pd.read_csv(sample_file_path)
        st.info("💡 **안내:** 현재 웹 서버에 내장된 **기본 샘플 데이터**가 로드되어 있습니다. 본인의 데이터를 분석하시려면 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
    else:
        st.warning("데이터 파일이 없습니다. 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")

# 데이터가 성공적으로 로드되었을 때만 메인 화면 출력
if df is not None:
    calc = hWTF_Recharge_Calculator(s_idx, k, r_cr, h_max, time_dry)
    try:
        x_raw, P_in, H_in = calc._read_dataframe(df)
        P_mm, _, _, H_calc = calc._prepare_units_and_gwl(P_in, H_in)
        
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

    # 계산 실행 버튼을 눌렀을 때
    if run_btn:
        with st.spinner("hWTF 알고리즘을 연산 중입니다..."):
            try:
                x, P_mm, t_rain, t_rech, t_rate, H_obs, H_sim = calc.calculate_recharge(df)
                
                st.markdown("---")
                st.subheader("✅ 산정 결과 (Results)")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("총 강수량", f"{t_rain*1000:.1f} mm")
                col2.metric("총 함양량", f"{t_rech*1000:.1f} mm")
                col3.metric("지하수 함양률", f"{t_rate:.2f} %")
                
                st.subheader("📉 지하수위 관측치 vs 모의치 피팅 (Fitting)")
                fig2, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x, H_obs, label="Observed (Measured)", color="black", linewidth=1.2)
                ax.scatter(x, H_sim, label="Calculated (Simulated)", marker="o", facecolors="none", edgecolors="olivedrab", linewidths=1.2, s=35)
                ax.set_title(f"GWL Comparison (Recharge Rate: {t_rate:.1f}%)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Groundwater Level (m)")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"계산 중 오류가 발생했습니다: {e}")
