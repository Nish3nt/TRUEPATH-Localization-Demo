# TRUEPATH: Hybrid Multi-Sensor Localization Prototype

A software validation prototype demonstrating resilient off-highway vehicle localization under GNSS uncertainty using multi-sensor fusion.

---

## 🚜 Problem Statement

Off-highway vehicles operating in mining, agriculture, forestry, and industrial environments frequently encounter:

- GNSS signal blockage due to buildings, terrain, and dense foliage  
- Multipath errors and signal degradation  
- Dust, vibration, and harsh environmental conditions  
- Sensor noise and drift accumulation  

When GNSS becomes unreliable, single-sensor localization approaches fail, leading to unsafe navigation due to unbounded position drift.

TRUEPATH addresses this challenge using hybrid sensor fusion.

---

## 🎯 Objective

To demonstrate a multi-sensor localization framework that:

- Maintains accurate vehicle positioning during GNSS dropouts  
- Combines IMU, wheel odometry, and GNSS intelligently  
- Degrades gracefully during signal loss  
- Re-converges once GNSS signal resumes  
- Quantitatively reduces drift compared to dead reckoning  

This prototype validates the feasibility of the TRUEPATH architecture before hardware deployment.

---

## 🧠 System Overview

The system simulates a vehicle moving in 2D space and generates:

- Ground truth trajectory  
- Noisy GNSS measurements  
- IMU angular rate with bias drift  
- Wheel encoder velocity with noise  

An Extended Kalman Filter (EKF) performs:

1. **Prediction Step**  
   Uses IMU + wheel velocity for motion propagation  

2. **Correction Step**  
   Uses GNSS position updates when available  

During GNSS outage (20–40 seconds), the system switches to prediction-only mode and maintains bounded drift.

---

## 📊 What This Demo Shows

The Streamlit application visualizes:

- Ground truth trajectory  
- Dead reckoning (IMU + wheel only)  
- TRUEPATH fusion estimate  
- Drift growth over time  
- RMSE improvement percentage  
- System health indicator  

It also includes:

- Real-time vehicle animation (optional)  
- GNSS outage flashing indicator  
- Adjustable sensor noise parameters  

This allows interactive evaluation of sensor robustness and fusion performance.

---

## 📈 Evaluation Metrics

The prototype computes:

- **RMSE (Root Mean Square Error)** over full trajectory  
- **Final Position Error**  
- **Relative Improvement (%)** over dead reckoning  
- **Drift accumulation profile**  

These metrics quantify the effectiveness of hybrid fusion under GNSS degradation.

---

## 🔬 Key Engineering Insight

Single-sensor systems fail under GNSS uncertainty.

Hybrid sensor fusion:
- Reduces drift accumulation  
- Ensures bounded localization error  
- Provides redundancy  
- Improves reliability in harsh environments  

This is the core principle behind TRUEPATH.

---

## 🏗️ Architecture Components

- `simulation.py`  
  Generates motion, sensors, and implements EKF fusion  

- `app.py`  
  Interactive Streamlit dashboard for visualization  

- `requirements.txt`  
  Dependencies for deployment  

---

## 🚀 Deployment

The prototype is deployed using:

- GitHub (code hosting)  
- Streamlit Cloud (web deployment)

Direct link  - https://truepath-localization-demo-rogbw5meqfqmangf245pgx.streamlit.app/
  
To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py

