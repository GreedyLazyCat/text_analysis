import csv
import json
import logging
import os
from json import JSONEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TupleEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': list(obj)}
        return super().default(obj)
def tuple_hook(dct):
    if '__tuple__' in dct:
        return tuple(dct['items'])
    return dct

class NewsClassifier:
    def __init__(self):
        self.model = GaussianNB()
        self.is_trained = False
    def _convert_tuples(self, data):
        if isinstance(data, tuple):
            return {'__tuple__': True, 'items': list(data)}
        elif isinstance(data, dict):
            return {k: self._convert_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_tuples(item) for item in data]
        return data

    def _restore_tuples(self, data):
        if isinstance(data, dict):
            if '__tuple__' in data:
                return tuple(data['items'])
            return {k: self._restore_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._restore_tuples(item) for item in data]
        return data

    def extract_features(self, stylometry_data):
        try:
            if isinstance(stylometry_data, str):
                data = json.loads(stylometry_data, object_hook=tuple_hook)
            elif isinstance(stylometry_data, dict):
                data = self._restore_tuples(stylometry_data)
            else:
                raise ValueError("Неподдерживаемый формат данных")
            required = ['avg_sentence_length', 'avg_word_length',
                        'lexical_density', 'lexical_diversity',
                        'char_frequencies', 'pos_ngrams']
            if not all(field in data for field in required):
                raise ValueError("Отсутствуют обязательные поля")
            features = [
                float(data['avg_sentence_length']),
                float(data['avg_word_length']),
                float(data['lexical_density']),
                float(data['lexical_diversity'])
            ]
            char_freq = data['char_frequencies']
            top_chars = sorted(char_freq.items(), key=lambda x: -x[1])[:10]
            features.extend([freq for _, freq in top_chars] + [0.0] * (10 - len(top_chars)))
            pos_ngrams = data['pos_ngrams']
            top_pos = sorted(pos_ngrams.items(), key=lambda x: -x[1])[:5]
            features.extend([freq for _, freq in top_pos] + [0.0] * (5 - len(top_pos)))
            return features
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {str(e)}")
            return None
    def load_data(self, old_news_file, recent_news_file):
        if not all(os.path.exists(f) for f in [old_news_file, recent_news_file]):
            raise FileNotFoundError("Файлы данных не найдены")
        X, y = [], []
        for file, label in [(old_news_file, 0), (recent_news_file, 1)]:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'stylometry' not in row:
                            continue

                        features = self.extract_features(row['stylometry'])
                        if features:
                            X.append(features)
                            y.append(label)
            except Exception as e:
                logger.error(f"Ошибка чтения {file}: {str(e)}")
                continue
        if not X:
            raise ValueError("Не удалось загрузить данные")
        return np.array(X), np.array(y)
    def train(self, old_news_file, recent_news_file, test_size=0.2):
        logger.info("Начало обучения модели")
        try:
            X, y = self.load_data(old_news_file, recent_news_file)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            self.model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, self.model.predict(X_test))
            logger.info(f"Обучение завершено. Точность: {accuracy:.2%}")
            self.is_trained = True
            return accuracy
        except Exception as e:
            logger.error(f"Ошибка обучения: {str(e)}")
            raise
    def predict(self, stylometry_data):
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        features = self.extract_features(stylometry_data)
        if features is None:
            raise ValueError("Неверный формат данных")
        return self.model.predict([features])[0]
    def predict_proba(self, stylometry_data):
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        features = self.extract_features(stylometry_data)
        if features is None:
            raise ValueError("Неверный формат данных")
        return self.model.predict_proba([features])[0]