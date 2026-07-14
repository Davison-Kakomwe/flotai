# 🏭 FlotAI — AI-Powered Froth Flotation Recovery Prediction

**Real-time copper recovery and concentrate grade prediction from froth video, using computer vision and machine learning.**

🔗 **[Live Demo](https://flotap-davika48.streamlit.app/)** | Built for [Kanz AI Hackathon] 2026
https://flotap-davika48.streamlit.app/
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Tests](https://img.shields.io/badge/Tests-16%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 The Problem

Copper flotation plants rely on operators visually judging froth appearance (color, bubble size, texture) to gauge recovery performance — a subjective process — while the only objective measurement, lab assay results, takes **1-4 hours** to come back. By the time results arrive, any underperforming ore has already been processed. Even a **0.5-1% improvement in recovery** can be worth millions of dollars annually for a mid-size concentrator.

## 💡 The Solution

FlotAI acts as a **"soft sensor"** — using computer vision to quantify froth characteristics (color, bubble size, texture, movement speed) directly from video, then predicting recovery % and grade % in real time, long before lab results return. This gives operators actionable, quantified feedback to guide reagent dosing decisions immediately.

## ✨ Features

- 🎥 **Live/upload toggle** — analyze uploaded video clips or simulate a live camera feed
- 🔬 **Computer vision feature extraction** — color (HSV), bubble size (contour detection), texture (Laplacian variance), and froth speed (Farneback optical flow)
- 🤖 **ML-based prediction** — Random Forest regression models predicting copper recovery % and concentrate grade %
- 📊 **Live dashboard** — real-time metrics, trend charts, and low-recovery alerts
- 🗄️ **Persistent history** — every reading logged to a relational database for trend analysis
- ✅ **16 automated tests** — covering database integrity, CV correctness, and ML prediction validity

## 🖼️ Screenshots

*(Add 2-3 screenshots of your dashboard here before submitting — drag image files into this README on GitHub's web editor, or reference an `assets/` folder)*

## 🏗️ Architecture