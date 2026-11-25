FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    streamlit==1.40.2 \
    pandas==2.2.3 \
    snowflake-connector-python==3.12.3 \
    cryptography==44.0.3 \
    pyOpenSSL==24.3.0

COPY monitoreo/tablero.py monitoreo/tablero.py

EXPOSE 8501
CMD ["streamlit", "run", "monitoreo/tablero.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]