# LCT2024-Task16 
Алгоритм для поиска предложенных скидок в телефонных разговорах с клиентами

Описание задачи: 

Разработайте сервис, который по транскрибации телефонного разговора сможет определить, предлагал ли оператор контакт-центра скидку, и если предлагал — то какую.

Ресурсы: 

Набор обезличенных данных, включающий: датасеты для обучения и валидации качества модели

# Порядок работы

1. Работоспособность модели проверена на Python 3.10 и 3.11.  
2. В файле requirements.txt необходимые модули для работы.  
3. Структура проекта:  
├ predict.bat – командный файл для получения сабмита  
│  
├ jupiter_notebooks/ - каталог с тетрадками jupiter notebook  
│       ├── eda_merged.ipynb - EDA - исследование датасета  
│       ├── fake-text-gpt.ipynb - генерация синтетического датасета 1  
│       ├── fake-text-gpt_квартиры.ipynb - генерация синтетического датасета 2  
│       ├── make_extend_dataset.ipynb - добавление к датасету синтетических данных  
│       ├── lct-task-16-berta.ipynb - обучение моделей семейства BERT  
│       ├── lct-task-16-mdeberta.ipynb - обучение моделей семейства DeBERTa  
│       └── predict-test.ipynb - получение предсказаний для сабмита  
│  
├ aeroplane/ - каталог проекта  
├── index.html       - HTML-файл для работы с API приложения через браузер  
├── api_app.py       - приложение для работы с моделью  
├── api_app_reqs.py  - проверка работы API приложения из Python  
├── data_process.py  - класс для обработки текста и поиск сущностей  
├── main_predict.py  - скрипт предсказания сущностей моделью BERT  
├── main_predict_mdeberta.py - скрипт предсказания сущностей моделью DeBERTa  
├── ner_testing.py   - скрипт для тестирования класса DataTransform  
├── data/            - каталог с датасетами  
│       ├── about.txt - описание структуры датасетов  
│       ├── gt_test.csv - тестовые данные для предсказаний  
│       ├── train_data.csv - тренировочный датасет  
│       └── train_test_extend_Z.zip - архивы с синтетическими данными  
├── docs/            - каталог с документацией  
│       └── Алгоритм работы модели.docx  
├── model/           - каталог с предобученной моделью BERT  
│       ├── config.json  
│       ├── model.safetensors  
│       ├── special_tokens_map.json  
│       ├── tokenizer_config.json  
│       └── vocab.txt  
├── model_mdb/ 	     - каталог с предобученной моделью DeBERTa  
│       ├── config.json  
│       ├── model.safetensors  
│       ├── spm.model  
│       └── tokenizer_config.json  
├── requirements.txt - необходимые библиотеки для проекта  
└── Dockerfile       - инструкции для создания образа контейнера  

4. Предобученные модели нужно скачать из датасета https://www.kaggle.com/datasets/saspav/x5-tech-ai-hack (в датасете модели находятся в каталоге .model/ и .model_mdb/) или с Яндекс-диска по ссылке https://disk.yandex.ru/d/h87XPOmDABTBBA - всё содержимое каталогов положить в соответствующий каталог проекта aeroplane/model/ и aeroplane/model_mdb/  
5. Получение предсказаний из файла gt_test.csv:  
python.exe main_predict.py  
или  
python.exe main_predict_mdeberta.py  
6. Сборка и запуск докер-контейнера (используется модель BERT, т.к. менее ресурсоемкая):  
перейти в каталог с проектом .aeroplane/ в нем выполнить команды:  
docker build -t aeroplane_app .  
docker run -d -p 8000:8000 --name aeroplane aeroplane_app  
7. Проверка работоспособности API приложения скриптом на Python:  
python.exe api_app_reqs.py
8. Работоспособность API можно проверить из командной строки командой:  
curl -X POST "http://127.0.0.1:8000/ner" -H "Content-Type: application/json" -d "{\"text\":\"Привет как дела Предоставите скидку в пять процентов на все товары\"}"  
ответом будет: {"text":"Привет как дела Предоставите скидку в пять процентов на все товары","labels":["O","O","O","O","B-discount","B-value","I-value","I-value","O","O","O"]}