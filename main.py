import csv
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from stylometry import StylometricAnalyzer
from datetime import datetime
from classifier import NewsClassifier
from recent_news import RecentNewsParser
from old_news import OldNewsParser

DATA_DIR = Path("data")

def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(i) for i in obj]
    else:
        return obj

def load_articles(filename):
    articles = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append({
                'title': row['title'],
                'text': row['text']
            })
    return articles

def load_analyzed_articles(filename):
    articles = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('stylometry'):
                continue
            articles.append({
                'text': row.get('text', ''),
                'stylometry': row['stylometry']
            })
    return articles

def load_raw_articles(filename):
    articles = []
    with open(filename, 'r', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            articles.append({
                'title': row.get('title', ''),
                'text': row.get('text', ''),
                'date': row.get('date', ''),
                'url': row.get('url', '')
            })
    return articles

def analyze_articles(articles, source):
    analyzer = StylometricAnalyzer()
    print(f"\nСтарт анализа для: {source}")
    results = []
    for i, art in enumerate(articles, 1):
        print(f"Анализ {i}/{len(articles)}: {art['title'][:60]}")
        analysis = analyzer.analyze(art['text'])
        art['stylometry'] = analysis
        results.append(art)
    return results

def save_articles_csv(filename, articles):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['text', 'stylometry']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for a in articles:
            writer.writerow({
                'text': a['text'],
                'stylometry': json.dumps(stringify_keys(a['stylometry']), ensure_ascii=False, indent=2)
            })

def save_raw_articles_csv(filename, articles):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['title', 'text', 'date', 'url']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            writer.writerow({
                'title': article['title'],
                'text': article['text'],
                'date': article['date'].strftime('%Y-%m-%d')
                if isinstance(article['date'], datetime) else article['date'],
                'url': article['url']
            })

def create_raw_csv_writer(filename):
    file = open(filename, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(file, fieldnames=['title', 'text', 'date', 'url'])
    writer.writeheader()
    return file, writer

def write_raw_article(writer, article):
    writer.writerow({
        'title': article.get('title', ''),
        'text': article.get('text', ''),
        'date': article.get('date').strftime('%Y-%m-%d')
        if isinstance(article.get('date'), datetime) else article.get('date', ''),
        'url': article.get('url', '')
    })

def create_temp_file(filename):
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f'.{filename.stem}.', suffix='.tmp', dir=DATA_DIR
    )
    os.close(file_descriptor)
    return Path(temp_name)

def parse_news(parse_ria, parse_gazeta):
    DATA_DIR.mkdir(exist_ok=True)
    temporary_files = []
    writers = []
    try:
        if parse_ria:
            ria_temp_file = create_temp_file(DATA_DIR / "ria.csv")
            temporary_files.append(ria_temp_file)
            ria_file, ria_writer = create_raw_csv_writer(ria_temp_file)
            writers.append(ria_file)
            ria = RecentNewsParser()
            try:
                ria_articles = ria.parse_site_ria("https://ria.ru/politics/")
            except Exception as error:
                print(f"[RIA] Не удалось получить список статей: {error}")
                ria_articles = []
            ria_full = []
            for i, article in enumerate(ria_articles, 1):
                print(f"[RIA] Статья {i}/{len(ria_articles)}: {article['link']}")
                try:
                    full = ria.parse_ria_article(article['link'])
                    if full:
                        article_data = {
                            'title': full.get('title', ''), 'text': full.get('text', ''),
                            'date': full.get('date') or article.get('date'),
                            'url': article['link']
                        }
                        ria_full.append(article_data)
                        write_raw_article(ria_writer, article_data)
                        ria_file.flush()
                except Exception as error:
                    print(f"[RIA] Статья пропущена из-за ошибки: {error}")
                time.sleep(1)
        else:
            ria_full = load_raw_articles(DATA_DIR / "ria.csv")

        if parse_gazeta:
            gazeta_temp_file = create_temp_file(DATA_DIR / "gazeta.csv")
            temporary_files.append(gazeta_temp_file)
            gazeta_file, gazeta_writer = create_raw_csv_writer(gazeta_temp_file)
            writers.append(gazeta_file)
            gazeta = OldNewsParser()
            try:
                gazeta_articles = gazeta.parse_site_gazeta(
                    "https://www.gazeta.ru/search.shtml?p=default&page=1&text=%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D0%B8&input=utf8&article=2&section=0&from=2003-01-01&to=2008-01-01&check=1"
                )
            except Exception as error:
                print(f"[Gazeta] Не удалось получить список статей: {error}")
                gazeta_articles = []
            gazeta_full = []
            for i, article in enumerate(gazeta_articles, 1):
                print(f"[Gazeta] Статья {i}/{len(gazeta_articles)}: {article['link']}")
                try:
                    full = gazeta.parse_gazeta_article(article['link'])
                    if full:
                        article_data = {
                            'title': full.get('title', ''), 'text': full.get('text', ''),
                            'date': full.get('date') or article.get('date'),
                            'url': article['link']
                        }
                        gazeta_full.append(article_data)
                        write_raw_article(gazeta_writer, article_data)
                        gazeta_file.flush()
                except Exception as error:
                    print(f"[Gazeta] Статья пропущена из-за ошибки: {error}")
                time.sleep(1)
        else:
            gazeta_full = load_raw_articles(DATA_DIR / "gazeta.csv")

        for file in writers:
            file.close()
        writers.clear()
        if parse_ria:
            os.replace(ria_temp_file, DATA_DIR / "ria.csv")
        if parse_gazeta:
            os.replace(gazeta_temp_file, DATA_DIR / "gazeta.csv")
        temporary_files.clear()
        return gazeta_full, ria_full
    finally:
        for file in writers:
            file.close()
        for temporary_file in temporary_files:
            if temporary_file.exists():
                print(f"Временный файл сохранён для восстановления: {temporary_file}")

def load_existing_dataset():
    filenames = [DATA_DIR / "gazeta.csv", DATA_DIR / "ria.csv"]
    missing = [filename for filename in filenames if not os.path.exists(filename)]
    if missing:
        raise FileNotFoundError(
            "Не найдены готовые наборы данных: " + ", ".join(str(filename) for filename in missing) +
            ". Выберите режим парсинга или добавьте эти файлы."
        )
    print("Используются готовые исходные файлы data/gazeta.csv и data/ria.csv")
    return load_raw_articles(filenames[0]), load_raw_articles(filenames[1])

def load_existing_analyzed_dataset():
    filenames = [DATA_DIR / "gazeta_analyzed.csv", DATA_DIR / "ria_analyzed.csv"]
    missing = [filename for filename in filenames if not os.path.exists(filename)]
    if missing:
        raise FileNotFoundError(
            "Не найдены результаты стилометрии: " + ", ".join(str(filename) for filename in missing)
        )
    return load_analyzed_articles(filenames[0]), load_analyzed_articles(filenames[1])

def analyze_and_save(gazeta_articles, ria_articles):
    print("\nНачинается стилометрический анализ")
    ria_analyzed = analyze_articles(ria_articles, "RIA")
    gazeta_analyzed = analyze_articles(gazeta_articles, "Gazeta")
    temporary_files = []
    try:
        for destination, articles in [
            (DATA_DIR / "ria_analyzed.csv", ria_analyzed),
            (DATA_DIR / "gazeta_analyzed.csv", gazeta_analyzed)
        ]:
            temporary_file = create_temp_file(destination)
            temporary_files.append(temporary_file)
            save_articles_csv(temporary_file, articles)
        for temporary_file, destination in zip(
            temporary_files,
            [DATA_DIR / "ria_analyzed.csv", DATA_DIR / "gazeta_analyzed.csv"]
        ):
            os.replace(temporary_file, destination)
        return gazeta_analyzed, ria_analyzed
    finally:
        for temporary_file in temporary_files:
            temporary_file.unlink(missing_ok=True)

def ask_source_parsing_mode(source):
    while True:
        answer = input(f"Парсить новости {source} заново? [д/н]: ").strip().lower()
        if answer in {'д', 'да', 'y', 'yes'}:
            return True
        if answer in {'н', 'нет', 'n', 'no'}:
            return False
        print("Введите 'д' для нового парсинга или 'н' для использования готовых данных.")

def ask_stylometry_mode():
    while True:
        answer = input("Провести стилометрический анализ заново? [д/н]: ").strip().lower()
        if answer in {'д', 'да', 'y', 'yes'}:
            return True
        if answer in {'н', 'нет', 'n', 'no'}:
            return False
        print("Введите 'д' для нового анализа или 'н' для использования готовой стилометрии.")


def main():
    parse_ria = ask_source_parsing_mode("РИА")
    parse_gazeta = ask_source_parsing_mode("Газеты.Ru")
    gazeta_articles, ria_articles = parse_news(parse_ria, parse_gazeta)

    if ask_stylometry_mode():
        gazeta_analyzed, ria_analyzed = analyze_and_save(gazeta_articles, ria_articles)
    else:
        gazeta_analyzed, ria_analyzed = load_existing_analyzed_dataset()

    print("\nЗапуск классификатора")
    classifier = NewsClassifier()
    accuracy = classifier.train(
        DATA_DIR / "gazeta_analyzed.csv",
        DATA_DIR / "ria_analyzed.csv"
    )
    print(f"Точность классификатора: {accuracy:.2%}")
    example_article = classifier.test_records[0]
    prediction = classifier.predict(example_article['stylometry'])
    proba = classifier.predict_proba(example_article['stylometry'])
    print(f"\nПример классификации:")
    print(f"Предсказанный класс: {'Новая' if prediction == 1 else 'Старая'} новость")
    print(f"Вероятности: [Старая: {proba[0]:.2%}, Новая: {proba[1]:.2%}]")
    print("\nПолный цикл завершён")
if __name__ == "__main__":
    main()
