@echo off
title BrandPulse AI - Streamlit
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creation de l'environnement virtuel local...
    python -m venv venv
)
call "venv\Scripts\activate.bat"
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    pip install streamlit
)
streamlit run app.py
pause