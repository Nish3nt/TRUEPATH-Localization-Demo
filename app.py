import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time as tm
from simulation import *

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="TRUEPATH Demo",
    layout="wide",
    page_icon="🚜"
)

# --------------------------------------------------
# Global Dark Theme Styling
# --------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: #E5E7EB;
}
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}
[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}
.scrolling-box {
    height:220px;
    overflow-y:auto;
    padding:15px;
    background-color:#111827;
    border-radius:10px;
    color:#E5E7EB;
}
hr {
    border: 1px solid #1F2937;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title & Intro
# --------------------------------------------------
st.title("🚜 TRUEPATH: Hybrid Multi-Sensor Localization Prototype")

st.markdown("""
Demonstration of robust localization under GNSS uncertainty using  
IMU + Wheel + GNSS sensor fusion (Extended Kalman Filter).
""")

st.markdown("---")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.markdown("## ⚙️ Simulation Controls")

gps_noise = st.sidebar.slider("📡 GPS Noise Std (m)", 0.1, 5.0, 1.5)
imu_noise = st.sidebar.slider("🧭 IMU Noise Std (rad/s)", 0.001, 0.05, 0.01)
wheel_noise = st.sidebar.slider("🛞 Wheel Noise Std (m/s)", 0.05, 1.0, 0.2)
dropout = st.sidebar.checkbox("🚫 Enable GPS Dropout (20–40s)", value=True)

st.sidebar.markdown("---")

animate = st.sidebar.checkbox("▶ Enable Live Movement Animation", value=False)

st.sidebar.markdown("""
### 🎯 What You Are Testing
- GNSS noise impact  
- IMU drift accumulation  
- Wheel slip uncertainty  
- GNSS outage survival  
""")

# --------------------------------------------------
# Generate Simulation
# --------------------------------------------------
time, x_true, y_true, theta_true, omega_true, velocity_true = generate_true_motion()

gps_x, gps_y = generate_gps(x_true, y_true, time, noise_std=gps_noise, dropout=dropout)
imu_omega = generate_imu(omega_true, time, noise_std=imu_noise)
wheel_velocity = generate_wheel_velocity(velocity_true, noise_std=wheel_noise)

x_dr, y_dr, theta_dr = dead_reckoning(time, imu_omega, wheel_velocity)
x_ekf, y_ekf, theta_ekf = ekf_fusion(time, imu_omega, wheel_velocity, gps_x, gps_y)

# --------------------------------------------------
# Live / Static Trajectory View
# --------------------------------------------------
st.subheader("🛰️ Trajectory View")

plot_placeholder = st.empty()

if animate:
    for i in range(0, len(time), 5):

        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#000000")

        ax.plot(x_true, y_true, color="#00BFFF", linewidth=1)
        ax.plot(x_dr, y_dr, linestyle="--", color="#FFA500", linewidth=1)
        ax.plot(x_ekf, y_ekf, color="#00FF7F", linewidth=1)

        # Moving vehicle dot
        ax.scatter(x_ekf[i], y_ekf[i], color="white", s=100)

        # GNSS flashing indicator
        if dropout and 20 < time[i] < 40:
            flash_color = "red" if (i // 5) % 2 == 0 else "#550000"
            ax.text(x_ekf[i], y_ekf[i] + 2, "GNSS LOST", color=flash_color,
                    fontsize=10, ha='center')

        ax.set_title("TRUEPATH Live Tracking", color="white")
        ax.set_xlabel("X Position (m)", color="white")
        ax.set_ylabel("Y Position (m)", color="white")
        ax.tick_params(colors='white')
        ax.grid(color='#1F2937', linestyle='--', linewidth=0.3)
        ax.axis("equal")

        plot_placeholder.pyplot(fig)
        tm.sleep(0.03)

else:
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    ax.plot(x_true, y_true, label="Ground Truth", color="#00BFFF")
    ax.plot(x_dr, y_dr, linestyle="--", label="Dead Reckoning", color="#FFA500")
    ax.plot(x_ekf, y_ekf, label="TRUEPATH Fusion", color="#00FF7F")

    ax.legend(facecolor="#111827", edgecolor="white", labelcolor="white")
    ax.set_title("Fusion vs Dead Reckoning", color="white")
    ax.set_xlabel("X Position (m)", color="white")
    ax.set_ylabel("Y Position (m)", color="white")
    ax.tick_params(colors='white')
    ax.grid(color='#1F2937', linestyle='--', linewidth=0.3)
    ax.axis("equal")

    st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# Drift vs Time Graph
# --------------------------------------------------
st.subheader("📈 Drift Growth Over Time")

dr_error = np.sqrt((x_true - x_dr)**2 + (y_true - y_dr)**2)
ekf_error = np.sqrt((x_true - x_ekf)**2 + (y_true - y_ekf)**2)

fig2, ax2 = plt.subplots(figsize=(10,4))
fig2.patch.set_facecolor("#000000")
ax2.set_facecolor("#000000")

ax2.plot(time, dr_error, label="Dead Reckoning Drift", color="#FFA500")
ax2.plot(time, ekf_error, label="TRUEPATH Drift", color="#00FF7F")

ax2.set_xlabel("Time (s)", color="white")
ax2.set_ylabel("Position Error (m)", color="white")
ax2.set_title("Drift Accumulation Comparison", color="white")
ax2.tick_params(colors='white')
ax2.grid(color='#1F2937', linestyle='--', linewidth=0.3)
ax2.legend(facecolor="#111827", edgecolor="white", labelcolor="white")

st.pyplot(fig2)

st.markdown("""
**Interpretation:**  
- Orange curve → Drift grows rapidly during GNSS outage.  
- Green curve → Fusion bounds error and re-converges when GNSS returns.  
""")

st.markdown("---")

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------
rmse_dr = np.sqrt(np.mean(dr_error**2))
rmse_ekf = np.sqrt(np.mean(ekf_error**2))

col1, col2, col3 = st.columns(3)
col1.metric("Dead Reckoning RMSE (m)", f"{rmse_dr:.2f}")
col2.metric("TRUEPATH RMSE (m)", f"{rmse_ekf:.2f}")
col3.metric("Improvement (%)", f"{((rmse_dr-rmse_ekf)/rmse_dr)*100:.1f}")

st.markdown("---")

# --------------------------------------------------
# System Health
# --------------------------------------------------
st.subheader("🔥 System Health Diagnostic")

health_score = max(0, 100 - rmse_ekf*5)
st.progress(int(health_score))

if health_score > 80:
    st.success("HIGH RELIABILITY — Drift controlled, fast GNSS recovery, robust fusion.")
elif health_score > 50:
    st.warning("MODERATE RELIABILITY — Increased sensitivity to noise.")
else:
    st.error("DEGRADED — High uncertainty, unsafe for autonomous operation.")

st.markdown("""
System health represents overall localization stability based on RMSE.
Lower fusion error → higher reliability score.
""")

st.markdown("---")

# --------------------------------------------------
# Knowledge Panel
# --------------------------------------------------
st.subheader("📘 Parameter & Model Explanation")

st.markdown("""
<div class="scrolling-box">

<b>GPS Noise:</b> Simulates multipath, signal blockage, satellite geometry errors.<br><br>

<b>IMU Noise:</b> Represents gyroscope bias drift causing heading error accumulation.<br><br>

<b>Wheel Noise:</b> Simulates slip and encoder inaccuracies in harsh terrain.<br><br>

<b>Dead Reckoning:</b> Uses only internal sensors → error grows unbounded.<br><br>

<b>Extended Kalman Filter:</b> Predicts motion using IMU + wheel and corrects using GNSS.<br><br>

<b>Evaluation Metrics:</b><br>
- RMSE: Root Mean Square Error over trajectory.<br>
- Final Error: Drift at end of mission.<br>
- Improvement %: Relative reduction in drift via fusion.<br><br>

<b>Engineering Insight:</b> Multi-sensor redundancy ensures bounded localization error during GNSS outages.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# Prototype Impact
# --------------------------------------------------
st.subheader("🚀 Why This Prototype Matters")

st.markdown("""
This software prototype validates a hybrid localization framework designed for:

- Off-highway vehicles  
- Mining & agricultural automation  
- GNSS-denied environments  

It demonstrates:

- GNSS outage survival  
- Drift mitigation  
- Sensor fusion correction  
- Scalable architecture for real-world deployment  

This validates TRUEPATH’s feasibility before hardware integration.
""")

st.caption("TRUEPATH Prototype | GNSS-denied resilient localization demonstration.")
