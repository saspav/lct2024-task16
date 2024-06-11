import requests
import json

url = "http://127.0.0.1:8000/ner"

text = (
    "двухкомнатная квартира пятьдесят квадратных метров с отделкой десять миллионов триста "
    "минимум пятнадцать процентов миллион шестьсот продаж дополнительно могу вам отправить "
    "скидку два процента она действует в течение двух дней сегодня и завтра")

payload = {"text": text}
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    print("Text:", result['text'])
    print("Labels:", result['labels'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
