---
title: Lumina Pro AI Suite
emoji: ✨
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# ✨ Lumina Pro — Neural Creative Studio

Enterprise-grade AI Image & 3D Suite geïnspireerd op de designstandaarden van Apple Human Interface Guidelines (HIG), Vercel en Linear. 100% in-memory bestandsverwerking via Python en ONNX Runtime.

---

## 🚀 7 Geïntegreerde Studio Modules

1. **✂️ AI Background Remover** — BiRefNet (State-of-the-Art) met Alpha Matting & Edge Feathering.
2. **📸 AI Studio Product Maker** — 3D Infinity Cyclorama, Softbox Licht, Contactschaduw & Spiegeling.
3. **🔍 AI Super-Resolution 4K** — Real-ESRGAN Compact Neuraal Netwerk (2x & 4x vergroting).
4. **🎞️ Cinematic LUT Studio** — Reinhard/CIELAB kleurtransfer, kleurenpalette-extractie & 33x33x33 `.cube` LUT export.
5. **📐 Smart Batch Resizer** — Social Media & E-Commerce formaten met Smart Crop, Letterbox & Lanczos resampling.
6. **💧 Watermark & Branding** — 9-Grid positionering, Tiled patterns & Photoshop blend modes.
7. **🧊 AI Photo to 3D Studio** — SAM Interactive Segmentatie, Volumetric SDF & Three.js 3D Mesh export (`.obj`).

---

## 📱 Overal Beschikbaar: Gebruik als App (PWA)

Je hoeft Lumina Pro niet via een App Store te downloaden. Dankzij de ingebouwde **Progressive Web App (PWA)** ondersteuning kun je hem direct op je telefoon of desktop installeren:

### Op iPhone / iPad (iOS Safari):
1. Open de web-link in **Safari**.
2. Tik onderin op het **Deel-icoon** (vierkantje met pijl omhoog).
3. Scroll naar beneden en tik op **"Zet op beginscherm"** (Add to Home Screen).
4. *Resultaat:* Lumina Pro verschijnt met een echt app-icoon en opent volledig scherm **zonder browserbalken**!

### Op Android (Chrome):
1. Open de web-link in **Chrome**.
2. Tik rechtsboven op de **drie puntjes**.
3. Tik op **"App installeren"** of **"Toevoegen aan startscherm"**.

---

## ☁️ Deployment Opties (Online Zetten)

### Optie 1: Hugging Face Spaces (Aanbevolen voor AI — Gratis 16GB RAM)
1. Ga naar [huggingface.co/spaces](https://huggingface.co/spaces) en klik op **Create new Space**.
2. Kies **Streamlit** of **Docker** als SDK.
3. Koppel je GitHub repository of push je code met Git.
4. Je app is binnen enkele minuten live met gratis 16GB RAM en 2 vCPUs!

### Optie 2: Streamlit Community Cloud (Gratis & Direct via GitHub)
1. Ga naar [share.streamlit.io](https://share.streamlit.io).
2. Koppel je GitHub repository en selecteer `app.py`.
3. Klik op **Deploy**.

### Optie 3: Eigen Cloud / Docker (Render / Railway / Hetzner)
Gebruik de meegeleverde `Dockerfile`:
```bash
docker build -t lumina-pro .
docker run -p 7860:7860 lumina-pro
```

---

## 🛠️ Lokaal Draaien
```bash
pip install -r requirements.txt
streamlit run app.py
```
