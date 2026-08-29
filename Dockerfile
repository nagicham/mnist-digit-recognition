# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user common/ ./common/
COPY --chown=user model/ ./model/
COPY --chown=user static/ ./static/

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]