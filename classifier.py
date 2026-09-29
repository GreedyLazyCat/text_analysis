import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from stylometry import POS_BIGRAMS, RUSSIAN_ALPHABET


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsClassifier:
    def __init__(self):
        self.models = {
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
        }
        # Keep the original default model available for callers using predict().
        self.model = self.models['naive_bayes']
        self.is_trained = False

    def extract_features(self, stylometry_data):
        try:
            data = json.loads(stylometry_data) if isinstance(stylometry_data, str) else stylometry_data
            required = {
                'avg_sentence_length', 'avg_word_length', 'lexical_density',
                'lexical_diversity', 'char_frequencies', 'pos_ngrams'
            }
            if not required.issubset(data):
                raise ValueError("Отсутствуют обязательные поля")

            features = [
                float(data['avg_sentence_length']),
                float(data['avg_word_length']),
                float(data['lexical_density']),
                float(data['lexical_diversity'])
            ]
            char_frequencies = data['char_frequencies']
            features.extend(float(char_frequencies.get(char, 0.0)) for char in RUSSIAN_ALPHABET)

            pos_frequencies = data['pos_ngrams']
            features.extend(
                float(pos_frequencies.get(f'{first}|{second}', 0.0))
                for first, second in POS_BIGRAMS
            )
            return features
        except Exception as error:
            logger.error(f"Ошибка извлечения признаков: {error}")
            return None

    def load_data(self, old_news_file, recent_news_file):
        records = []
        for filename, label in [(old_news_file, 0), (recent_news_file, 1)]:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Файл данных не найден: {filename}")
            with open(filename, 'r', encoding='utf-8') as file:
                for row_number, row in enumerate(csv.DictReader(file), 2):
                    if not row.get('stylometry'):
                        continue
                    features = self.extract_features(row['stylometry'])
                    if features is not None:
                        records.append({
                            'features': features,
                            'label': label,
                            'source_file': str(filename),
                            'row_number': row_number,
                            'text': row.get('text', ''),
                            'stylometry': row['stylometry']
                        })
        if not records:
            raise ValueError("Не удалось загрузить данные")
        return records

    def train(self, old_news_file, recent_news_file, test_size=0.2,
              split_file=Path('data/classifier_split.json')):
        logger.info("Начало обучения модели")
        records = self.load_data(old_news_file, recent_news_file)
        indices = np.arange(len(records))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=[record['label'] for record in records]
        )
        x_train = np.array([records[index]['features'] for index in train_indices])
        y_train = np.array([records[index]['label'] for index in train_indices])
        x_test = np.array([records[index]['features'] for index in test_indices])
        y_test = np.array([records[index]['label'] for index in test_indices])

        accuracies = {}
        for name, model in self.models.items():
            model.fit(x_train, y_train)
            accuracies[name] = accuracy_score(y_test, model.predict(x_test))

        self.is_trained = True
        self.test_records = [records[index] for index in test_indices]
        self._save_split(records, train_indices, test_indices, split_file)
        for name, accuracy in accuracies.items():
            logger.info(f"Обучение {name} завершено. Точность: {accuracy:.2%}")
        return accuracies

    def _save_split(self, records, train_indices, test_indices, split_file):
        split_file = Path(split_file)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'random_state': 42,
            'test_size': 0.2,
            'train': [self._split_record(records[index]) for index in train_indices],
            'test': [self._split_record(records[index]) for index in test_indices]
        }
        temporary_file = split_file.with_suffix('.tmp')
        temporary_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary_file, split_file)

    @staticmethod
    def _split_record(record):
        return {
            'source_file': record['source_file'],
            'row_number': record['row_number'],
            'label': record['label'],
            'text': record['text']
        }

    def predict(self, stylometry_data, classifier='naive_bayes'):
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        if classifier not in self.models:
            raise ValueError(f"Неизвестный классификатор: {classifier}")
        features = self.extract_features(stylometry_data)
        if features is None:
            raise ValueError("Неверный формат данных")
        return self.models[classifier].predict([features])[0]

    def predict_proba(self, stylometry_data, classifier='naive_bayes'):
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        if classifier not in self.models:
            raise ValueError(f"Неизвестный классификатор: {classifier}")
        features = self.extract_features(stylometry_data)
        if features is None:
            raise ValueError("Неверный формат данных")
        return self.models[classifier].predict_proba([features])[0]
