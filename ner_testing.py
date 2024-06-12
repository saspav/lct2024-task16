import unittest
from data_process import DataTransform


class TestDataTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # bert_name = 'DeepPavlov/rubert-base-cased-conversational'
        bert_name = './model'
        bert_tuned = './model'
        cls.dps = DataTransform(model_name=bert_name, model_path=bert_tuned)

    def test_preprocess_text(self):
        text = "Привет как дела Предоставите скидку в пять процентов на все товары"
        words, tokenized_words, input_ids, attention_mask = self.dps.preprocess_text(text)

        true_words = ['привет', 'как', 'дела', 'предоставите', 'скидку', 'в', 'пять',
                      'процентов', 'на', 'все', 'товары']
        true_tokens = ['привет', 'как', 'дела', 'предоставит', '##е', 'скидку', 'в', 'пять',
                       'процентов', 'на', 'все', 'товары']
        self.assertEqual(words, true_words)
        self.assertEqual(tokenized_words, true_tokens)
        self.assertTrue(len(tokenized_words) == 12)
        self.assertTrue(len(input_ids) == 12)
        self.assertTrue(attention_mask == [1] * 12)

    def test_split_text_with_overlap(self):
        text = "Привет как дела Предоставите скидку в пять процентов на все товары"
        chunks = self.dps.split_text_with_overlap(text, max_len=10, overlap=0.2)
        true_chunks = [('привет как дела предоставите скидку в пять процентов на', 0),
                       ('процентов на все товары', 7)]

        self.assertTrue(len(chunks) == 2)
        self.assertTrue(all(len(chunk[0].split()) <= 10 for chunk in chunks))
        self.assertTrue(chunks == true_chunks)

    def test_get_words_positions(self):
        words = ['скидка', 'процент', 'товар', 'скидки']
        pattern = self.dps.discount_pattern
        positions = self.dps.get_words_positions(words, pattern)

        self.assertEqual(positions, [0, 3])

    def test_get_entities_with_labels(self):
        tokenized_words = ['привет', 'как', 'дела', 'предоставит', '##е', 'скидку', 'в',
                           'пять', 'процентов', 'на', 'все', 'товары']
        predicted_labels = [3, 3, 3, 3, 3, 2, 0, 1, 1, 3, 3, 3]
        start_index = 0
        labels_positions, result = self.dps.get_entities_with_labels(tokenized_words,
                                                                     predicted_labels,
                                                                     start_index)
        true_labels_positions = {'B-discount': [4], 'B-value': [5], 'I-value': [6, 7]}
        true_result = [('скидку', 2, 4), ('в', 0, 5), ('пять', 1, 6), ('процентов', 1, 7)]
        self.assertTrue(labels_positions == true_labels_positions)
        self.assertTrue(result == true_result)

    def test_merge_chunks_results(self):
        labels_positions = [{'B-discount': [4], 'B-value': [5], 'I-value': [6, 7]}]
        results = [('скидку', 2, 4), ('в', 0, 5), ('пять', 1, 6), ('процентов', 1, 7)]
        original_length = 11
        final_labels_positions, final_results = self.dps.merge_chunks_results(
            labels_positions, results, original_length)

        true_labels_positions = [{'B-discount': [4], 'B-value': [5], 'I-value': [6, 7]}]
        self.assertTrue(final_labels_positions == true_labels_positions)
        self.assertEqual(final_results, [3, 3, 3, 3, 2, 0, 1, 1, 3, 3, 3])

    def test_transform_text_labels(self):
        text = "Привет как дела Предоставите скидку в пять процентов на все товары"
        labels = {'B-discount': [4], 'B-value': [5], 'I-value': [6, 7]}
        idx_labels = self.dps.transform_text_labels(text, labels)

        true_idx = ['O', 'O', 'O', 'O', 'B-discount', 'B-value', 'I-value', 'I-value',
                    'O', 'O', 'O']

        self.assertEqual(idx_labels, true_idx)

    def test_get_entities(self):
        text = "Привет как дела Предоставите скидку в пять процентов на все товары"
        labels_positions, final_results, final_positions = self.dps.get_entities(text)

        true_labels_positions = {'B-discount': [4], 'B-value': [5], 'I-value': [6, 7]}
        true_final_results = [3, 3, 3, 3, 2, 0, 1, 1, 3, 3, 3]
        true_final_positions = [(4, 'скидку'), (5, 'в'), (6, 'пять'), (7, 'процентов')]

        self.assertTrue(labels_positions == true_labels_positions)
        self.assertTrue(final_results == true_final_results)
        self.assertTrue(final_positions == true_final_positions)


if __name__ == '__main__':
    unittest.main()
