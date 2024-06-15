import re
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertForTokenClassification
from transformers import DebertaV2Tokenizer, DebertaV2ForTokenClassification

# пытаемся импортировать самодельный экспорт в эксель с красивостями
try:
    from df_addons import df_to_excel
except ModuleNotFoundError:
    df_to_excel = lambda sdf, spt, *args, **kwargs: sdf.to_excel(spt, index=False)

__import__('warnings').filterwarnings("ignore")

MAX_LEN = 512
OVERLAP = 0.2

target_labels = ['B-value', 'I-value', 'B-discount']
tag_values = target_labels + ['O', 'PAD']
tag2idx = {tag: idx for idx, tag in enumerate(tag_values)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}


class DataTransform:
    """ Класс для поиска сущностей"""

    def __init__(self, model_name=None, model_path=None,
                 tokenizer=None, token_classification=None,
                 load_model=True, cuda=True):
        """
        Инициализация экземпляра класса
        :param model_name: имя модели
        :param model_path: путь к предобученной модели
        :param tokenizer: токенизатор
        :param token_classification: классификатор
        :param cuda: использовать GPU
        """

        # Используем GPU если доступно
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

        # путь к локальной модели
        if model_name is None:
            self.model_name = './model'
        else:
            self.model_name = model_name

        # путь к локальной модели
        if model_path is None:
            self.model_path = './model'
        else:
            self.model_path = model_path

        if load_model:
            if tokenizer is None:
                tokenizer = BertTokenizer

            self.tokenizer = tokenizer.from_pretrained(self.model_name,
                                                       do_lower_case=False)

            if token_classification is None:
                token_classification = BertForTokenClassification
            # Загрузка модели
            self.model = token_classification.from_pretrained(self.model_path,
                                                              num_labels=len(tag2idx),
                                                              output_attentions=False,
                                                              output_hidden_states=False,
                                                              ignore_mismatched_sizes=True,
                                                              )
            self.model.to(self.device)
            self.model.eval()

        else:
            self.model = None

        # Использовать регулярки для поиска сущностей
        self.regexp_text = False
        self.discount_pattern = re.compile('скид[ко]\w*', flags=re.IGNORECASE)
        self.percent_pattern = re.compile('процент\w*', flags=re.IGNORECASE)

    def preprocess_text(self, text):
        """
        Преобразование текста в нужный формат: токенизация и преобразование входных данных
        :param text: текст
        :return: words, tokenized_words, input_ids, attention_mask
        """
        words = text.lower().split()
        tokenized_words = self.tokenizer.tokenize(text.lower())
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_words)

        input_ids = input_ids[:MAX_LEN]

        attention_mask = [1] * len(input_ids)

        return words, tokenized_words, input_ids, attention_mask

    def split_text_with_overlap(self, text, max_len=MAX_LEN, overlap=OVERLAP):
        """
        Разделить текст на части с перекрытием.
        :param text: текст для разделения
        :param max_len: максимальная длина части
        :param overlap: процент перекрытия
        :return: список частей текста
        """

        words = text.lower().split()
        chunk_size = max(max_len - 12, 10)  # Оставляем место для специальных токенов
        overlap_size = int(chunk_size * overlap)
        chunks = []
        start_index = end_index = 0
        while end_index < len(words):
            end_index = start_index + chunk_size
            chunk_words = words[start_index:end_index]
            chunk_text = ' '.join(chunk_words)

            # Токенизируем текст, чтобы убедиться, что кол-во токенов не превышает max_len
            tokens = self.tokenizer.tokenize(chunk_text)
            while len(tokens) > chunk_size and len(chunk_words):
                chunk_words = chunk_words[:-1]
                chunk_text = ' '.join(chunk_words)
                tokens = self.tokenizer.tokenize(chunk_text)

            chunks.append((chunk_text, start_index))

            # У нас один чанк и нечего дальше крутить цикл - виснет
            if len(chunk_words) == len(words):
                break

            # найдем сколько слов входит в перекрытие для вычисления индекса смещения
            tokens = []
            overlap_index = 0
            reversed_words = chunk_words[::-1]
            # пока длина токенов перекрытия меньше размера перекрытия добавляем по слову
            while len(tokens) < overlap_size:
                overlap_index += 1
                tokens = self.tokenizer.tokenize(' '.join(reversed_words[:overlap_index]))

            end_index = start_index + len(chunk_words)
            # Следующая часть начинается с учетом перекрытия
            start_index += len(chunk_words) - overlap_index

        return chunks

    @staticmethod
    def get_words_positions(input_words, pattern):
        found_index = []
        for idx, word in enumerate(input_words):
            if pattern.match(word):
                found_index.append(idx)
        return found_index

    def get_entities(self, text):
        """
        Функция принимает на вход текст и возвращает найденные сущности и их индексы.
        :param text: текст
        :return: найденные сущности и их индексы
        """
        # Разделить текст на части с перекрытием
        chunks = self.split_text_with_overlap(text, MAX_LEN, OVERLAP)
        all_labels_positions = []
        all_results = []

        # print(f'\nlen(chunks): {len(chunks)}', 'chunks:', *chunks, sep='\n')

        for chunk, start_index in chunks:
            words, tokenized_words, input_ids, attention_mask = self.preprocess_text(chunk)

            # Создайте тензоры для входных данных
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)

            # Прогоните текст через модель для предсказания меток
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            logits = outputs[0].detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=2)[0]

            labels_positions, result = self.get_entities_with_labels(tokenized_words,
                                                                     predicted_labels,
                                                                     start_index)

            # print('labels_positions:', labels_positions, result)

            all_labels_positions.append(labels_positions)
            all_results.extend(result)

        words = text.lower().split()

        # Если используем регулярки для дополнительного поиска сущностей
        if self.regexp_text:
            labels_positions, result = {}, []
            found_idxs = self.get_words_positions(words, self.discount_pattern)
            if found_idxs:
                labels_positions['B-discount'] = found_idxs
                result.extend([(words[idx], 2, idx) for idx in found_idxs])
                # found_idxs = self.get_words_positions(words, self.percent_pattern)
                # labels_positions['I-value'] = found_idxs
                # result.extend([(words[idx], 1, idx) for idx in found_idxs])
                all_labels_positions.append(labels_positions)
                all_results.extend(result)

        # print('all_lpr:', all_labels_positions, all_results, len(text.split()), sep='\n')

        try:
            # Удаление дублированных меток и приведение к исходному тексту
            labels_positions, final_results = self.merge_chunks_results(all_labels_positions,
                                                                        all_results,
                                                                        len(words))
        except TypeError as err:
            print(err)
            print(text)
            print(all_labels_positions, all_results, len(text.split()), sep='\n')

        # print('lp_fr:', labels_positions, final_results, len(text.split()), sep='\n')

        # если это список словарей
        if isinstance(labels_positions, list):
            # объединение позиций слов по ключам
            combined_values = defaultdict(list)
            for item in labels_positions:
                for key, values in item.items():
                    combined_values[key].extend(values)
            labels_positions = {key: sorted(set(combined_values[key]))
                                for key in sorted(combined_values)}

        final_positions = []
        for values in labels_positions.values():
            final_positions.extend(values)
        final_positions = [(idx, words[idx]) for idx in sorted(final_positions)]

        return labels_positions, final_results, final_positions

    def get_entities_with_labels(self, tokenized_words, predicted_labels, start_index):
        """
        Объединение токенов в сущности с метками
        :param tokenized_words: токенизированные слова
        :param predicted_labels: предсказанные метки
        :param start_index: начальный индекс чанка
        :return:
        """
        current_word = ""
        current_label = []
        words_with_labels = []

        # print('token_words, pred_labels:', tokenized_words, predicted_labels, sep='\n')

        if isinstance(self.tokenizer, DebertaV2Tokenizer):
            for token, label in zip(tokenized_words, predicted_labels):
                if token.startswith('▁'):
                    if current_word:
                        words_with_labels.append((current_word, current_label))
                    current_word = token[1:]
                    current_label = [label]
                else:
                    current_word += token
                    current_label.append(label)

        else:
            for token, label in zip(tokenized_words, predicted_labels):
                if token.startswith("##"):
                    current_word += token[2:]
                    current_label.append(label)
                else:
                    if current_word:
                        words_with_labels.append((current_word, current_label))
                    current_word = token
                    current_label = [label]

        if current_word:
            words_with_labels.append((current_word, current_label))

        result = []
        previous_label = None
        previous_index = None
        for idx, (word, labels) in enumerate(words_with_labels):
            label = Counter(labels).most_common(1)[0][0]
            if label < 3:
                if label != 2:
                    label = int(previous_index is not None and
                                previous_label is not None and
                                idx == previous_index + 1 and previous_label < 2)
                result.append((word, label, start_index + idx))
                previous_label = label
                previous_index = idx

        target_labels_positions = {}
        for target, target_label in enumerate(target_labels):
            labels = [idx for word, label, idx in result if label == target]
            if labels:
                target_labels_positions[target_label] = labels
        return target_labels_positions, result

    @staticmethod
    def merge_chunks_results(labels_positions, results, original_length):
        """
        Объединение результатов из перекрывающихся частей в один результат.
        :param labels_positions: Список позиций меток из частей
        :param results: список результатов из частей
        :param original_length: длина оригинального текста в словах
        :return: объединенные метки и результаты
        """
        final_labels_positions = [*filter(bool, labels_positions)]
        final_results = [3] * original_length

        # print('labels_positions:', labels_positions)
        # print('results:', results)

        for word, label, idx in results:
            if final_results[idx] > 2:
                final_results[idx] = label

        return final_labels_positions, final_results

    def predict_entities(self, test_df):
        """
        Поиск сущностей в датафрейме
        :param test_df: ДФ
        :return:
        """
        true_labels = []
        pred_labels = []
        predict_labels = []
        predict_results = []
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            input_text = row['processed_text']

            if 'target_labels_positions' in row:
                true_labels.append(self.transform_text_labels(input_text,
                                                              row['target_labels_positions']))

            entities, result, positions = self.get_entities(input_text)
            predict_labels.append(entities)
            predict_results.append(positions)

            pred_labels.append(self.transform_text_labels(input_text, entities))

        test_df['predict_labels'] = predict_labels
        test_df['predict_results'] = predict_results

        return test_df, true_labels, pred_labels

    @staticmethod
    def transform_text_labels(text, labels):
        """
        Формирование списка меток для каждого слова в тексте.
        :param text: текст со словами, разделенными пробелами
        :param labels: список со словарями меток и их позициями
        :return: список меток для каждого слова в тексте
        """
        # если метки - это словарь, засунем его в список
        if isinstance(labels, dict):
            labels = [labels]
        len_words = len(text.split())
        idx_labels = ['O'] * len_words
        for label_dict in labels:
            for key, values in label_dict.items():
                for value in values:
                    if value < len_words:
                        idx_labels[value] = key
        return idx_labels


if __name__ == "__main__":
    print(tag2idx)
    print(idx2tag)
