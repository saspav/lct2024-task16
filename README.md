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
aeroplane/  
├── index.html       - HTML-файл для работы с API приложения через браузер  
├── api_app.py       - приложение для работы с моделью  
├── api_app_reqs.py  - проверка работы API приложения из Python  
├── data_process.py  - класс для обработки текста и поиск сущностей  
├── main_predict.py  - скрипт предсказания сущностей по файлу 'gt_test.csv'  
├── ner_testing.py   - скрипт для тестирования класса DataTransform  
├── data/            - каталог с датасетами  
│       ├── about.txt - описание структуры датасетов  
│       ├── gt_test.csv - тестовые данные для предсказаний  
│       ├── train_data.csv - тренировочный датасет  
│       ├── train_test_extend_Z.zip - архивы с синтетическими данными  
├── docs/            - каталог с документацией  
│       ├── Алгоритм работы модели.docx  
└── jupiter_notebooks/ - каталог с тетрадками jupiter notebook  
│       ├── lct-task-16-berta.ipynb - обучение моделей семейства BERT  
│       ├── lct-task-16-mdeberta.ipynb - обучение моделей семейства DeBERTa  
│       ├── predict-test.ipynb - получение предсказаний  
├── model/           - каталог с предобученной моделью BERT или DeBERTa  
│       ├── config.json  
│       ├── model.safetensors  
│       ├── special_tokens_map.json  
│       ├── tokenizer_config.json  
│       ├── vocab.txt  
├── requirements.txt - необходимые библиотеки для проекта  
└── Dockerfile       - инструкции для создания образа контейнера  

4. Предобученную модель нужно скачать из датасета https://www.kaggle.com/datasets/saspav/x5-tech-ai-hack (в датасете модель находится в каталоге .model/) - всё содержимое каталога положить в каталог проекта aeroplane/model/
5. Получение предсказаний из файла gt_test.csv:  
python.exe main_predict.py  
6. Сборка и запуск докер-контейнера:  
перейти в каталог с проектом .aeroplane/ в нем выполнить команды:  
docker build -t aeroplane_app .  
docker run -d -p 8000:8000 --name aeroplane aeroplane_app  
7. Работоспособность API можно проверить из командной строки командой:  
curl -X POST "http://127.0.0.1:8000/ner" -H "Content-Type: application/json" -d "{\"text\":\"Привет как дела Предоставите скидку в пять процентов на все товары\"}"  
ответом будет: {"text":"Привет как дела Предоставите скидку в пять процентов на все товары","labels":["O","O","O","O","B-discount","B-value","I-value","I-value","O","O","O"]}