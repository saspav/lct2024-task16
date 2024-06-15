FROM python:3.10-slim

WORKDIR /app

COPY . /app

VOLUME /app/data

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# ���������� ���������� ��� �������������� ����������� ������
ENV PYTHONUNBUFFERED=1

RUN chmod +x /app/api_app.py

# ������� ��� ������� ����������
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
