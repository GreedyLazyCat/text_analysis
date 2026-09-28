import csv
import json
import os
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

def parse_news():
    DATA_DIR.mkdir(exist_ok=True)
    print("Начинается парсинг новостей RIA и Gazeta")
    ria_url = "https://ria.ru/politics/"
    ria = RecentNewsParser()
    ria_articles = ria.parse_site_ria(ria_url)
    ria_full = []
    for i, article in enumerate(ria_articles, 1):
        print(f"[RIA] Статья {i}/{len(ria_articles)}: {article['link']}")
        full = ria.parse_ria_article(article['link'])
        if full:
            ria_full.append({
                'title': full.get('title', ''),
                'text': full.get('text', ''),
                'date': full.get('date') or article.get('date'),
                'url': article['link']
            })
        time.sleep(1)
    with open(DATA_DIR / "ria.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ['title', 'text', 'date', 'url']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in ria_full:
            writer.writerow({
                'title': a['title'],
                'text': a['text'],
                'date': a['date'].strftime('%Y-%m-%d') if isinstance(a['date'], datetime) else a['date'],
                'url': a['url']
            })

    gazeta = OldNewsParser()
    gazeta_urls = [
        "https://www.gazeta.ru/search.shtml?p=main&page=1&text=новости&input=utf8&article=1&section=2820815&from=2003-01-01&to=2008-01-01&sort_order=published_desc"
    ]
    gazeta_articles = []
    for url in gazeta_urls:
        gazeta_articles.extend(gazeta.parse_site_gazeta(url))
    gazeta_full = []
    for i, article in enumerate(gazeta_articles, 1):
        print(f"[Gazeta] Статья {i}/{len(gazeta_articles)}: {article['link']}")
        full = gazeta.parse_gazeta_article(article['link'])
        if full:
            gazeta_full.append({
                'title': full.get('title', ''),
                'text': full.get('text', ''),
                'date': full.get('date') or article.get('date'),
                'url': article['link']
            })
        time.sleep(1)
    with open(DATA_DIR / "gazeta.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ['title', 'text', 'date', 'url']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in gazeta_full:
            writer.writerow({
                'title': a['title'],
                'text': a['text'],
                'date': a['date'].strftime('%Y-%m-%d') if isinstance(a['date'], datetime) else a['date'],
                'url': a['url']
            })

    print("\nНачинается стилометрический анализ")
    ria_analyzed = analyze_articles(ria_full, "RIA")
    save_articles_csv(DATA_DIR / "ria_analyzed.csv", ria_analyzed)
    gazeta_analyzed = analyze_articles(gazeta_full, "Gazeta")
    save_articles_csv(DATA_DIR / "gazeta_analyzed.csv", gazeta_analyzed)
    return gazeta_analyzed, ria_analyzed

def load_existing_dataset():
    filenames = [DATA_DIR / "gazeta_analyzed.csv", DATA_DIR / "ria_analyzed.csv"]
    missing = [filename for filename in filenames if not os.path.exists(filename)]
    if missing:
        raise FileNotFoundError(
            "Не найдены готовые наборы данных: " + ", ".join(str(filename) for filename in missing) +
            ". Выберите режим парсинга или добавьте эти файлы."
        )
    print("Используются готовые файлы data/gazeta_analyzed.csv и data/ria_analyzed.csv")
    return load_analyzed_articles(filenames[0]), load_analyzed_articles(filenames[1])

def ask_parsing_mode():
    while True:
        answer = input("Парсить новости заново? [д/н]: ").strip().lower()
        if answer in {'д', 'да', 'y', 'yes'}:
            return True
        if answer in {'н', 'нет', 'n', 'no'}:
            return False
        print("Введите 'д' для нового парсинга или 'н' для использования готовых данных.")


def main():
    if ask_parsing_mode():
        gazeta_analyzed, ria_analyzed = parse_news()
    else:
        gazeta_analyzed, ria_analyzed = load_existing_dataset()

    print("\nЗапуск классификатора")
    classifier = NewsClassifier()
    accuracy = classifier.train(
        DATA_DIR / "gazeta_analyzed.csv",
        DATA_DIR / "ria_analyzed.csv"
    )
    print(f"Точность классификатора: {accuracy:.2%}")
    example_article = gazeta_analyzed[0]
    prediction = classifier.predict(example_article['stylometry'])
    proba = classifier.predict_proba(example_article['stylometry'])
    print(f"\nПример классификации:")
    print(f"Предсказанный класс: {'Новая' if prediction == 1 else 'Старая'} новость")
    print(f"Вероятности: [Старая: {proba[0]:.2%}, Новая: {proba[1]:.2%}]")
    print("\nПолный цикл завершён")
if __name__ == "__main__":
    main()
