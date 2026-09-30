FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 APP_HOME=/app
WORKDIR ${APP_HOME}
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY . .
CMD ["python", "web_app.py"]
