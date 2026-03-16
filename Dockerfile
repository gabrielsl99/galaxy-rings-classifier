FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
```

**Paso 5 — Asegúrate que `requirements.txt`** tenga esto:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
torchvision==0.20.1+cpu
timm==1.0.12
numpy==1.26.4
pandas==2.2.3
Pillow==11.1.0
astropy==6.1.7
scipy==1.14.1
scikit-image==0.24.0
opencv-python-headless==4.10.0.84
matplotlib==3.9.3
seaborn==0.13.2
requests==2.32.3
streamlit==1.41.1
