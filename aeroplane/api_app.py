from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DebertaV2Tokenizer, DebertaV2ForTokenClassification
from data_process import DataTransform

app = FastAPI()

# Разрешение CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null"  # Для покрытия случаев, когда файлы открыты напрямую в браузере
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модель для запросов
class TextRequest(BaseModel):
    text: str


bert_name = 'DeepPavlov/rubert-base-cased-conversational'
bert_tuned = './model'

dps = DataTransform(model_name=bert_name, model_path=bert_tuned)


# bert_name = 'microsoft/mdeberta-v3-base'
# bert_tuned = r'Z:\python-datasets\LCT_2024_16\models\mdeberta-v3-base'
#
# dps = DataTransform(model_name=bert_name, model_path=bert_tuned,
#                     tokenizer=DebertaV2Tokenizer,
#                     token_classification=DebertaV2ForTokenClassification)


# Модель для запросов
class TextRequest(BaseModel):
    text: str


@app.post("/ner")
def get_ner(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Преобразование предсказаний модели в метки
    entities, *_ = dps.get_entities(text)
    labels = dps.transform_text_labels(text, entities)

    return {"text": text, "labels": labels}

# Запуск сервера: uvicorn api_app:app --reload
