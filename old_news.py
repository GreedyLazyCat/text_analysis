from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
import csv
from urllib.parse import urlparse
import pymorphy3
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class OldNewsParser:
    def __init__(self):
        self.driver = None
        self.base_url = "https://www.gazeta.ru"
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")
        self.STOP_WORDS = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
            'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к',
            'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне',
            'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему',
            'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
            'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
            'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего',
            'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней',
            'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
            'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
            'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой',
            'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
            'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем',
            'всех', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
            'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти',
            'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве',
            'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
            'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя',
            'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
        return self.driver
    def extract_date_from_url(self, href):
        try:
            match = re.search(r'/(\d{4})/(\d{2})/(\d{2})', href)
            if match:
                return datetime.strptime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", "%Y-%m-%d")
        except Exception as e:
            print(f"Ошибка извлечения даты из URL {href}: {e}")
        return None
    def parse_site_gazeta(self, url, max_clicks=20):
        if not self.setup_driver():
            return []
        try:
            self.driver.get(url)
            time.sleep(2)
            for i in range(max_clicks):
                try:
                    show_more_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.b_showmorebtn-link'))
                    )
                    self.driver.execute_script("arguments[0].click();", show_more_btn)
                    time.sleep(2)  # Даем странице обновиться
                except Exception:
                    print(f"Кнопка 'Показать еще' не найдена {i + 1}")
                    break
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.b_ear-title a"))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            results = []
            for a in soup.select("div.b_ear-title a"):
                href = a.get("href")
                title = a.get_text(strip=True)
                if not title or not href or not href.endswith('.shtml'):
                    continue
                if not href.startswith("http"):
                    href = self.base_url + href
                results.append({
                    'title': title,
                    'link': href,
                    'date': self.extract_date_from_url(href)
                })
            return results
        except Exception as e:
            print(f"Ошибка парсинга {url}: {e}")
            return []
        finally:
            self.driver.quit()
    def parse_gazeta_article(self, url):
        if not self.setup_driver():
            return None
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "p"))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            cleaned_text = self.process_text(text)
            title = soup.find("h1")
            title_text = title.get_text(strip=True) if title else ""
            return {
                'title': title_text,
                'text': cleaned_text,
                'date': self.extract_date_from_url(url.replace(self.base_url, '')),
                'url': url
            }
        except Exception as e:
            print(f"Ошибка парсинга статьи {url}: {e}")
            return None
        finally:
            self.driver.quit()
    def process_text(self, text):
        tokens = word_tokenize(text.lower())
        processed = []
        for word in tokens:
            if not word.isalpha() or word in self.STOP_WORDS:
                continue
            lemma = self.morph.parse(word)[0].normal_form
            stem = self.stemmer.stem(lemma)
            processed.append(stem)
        return " ".join(processed)
    def analyze_old_news(self, articles, min_text_length=50):
        if not articles:
            print("Нет статей для фильтрации")
            return []
        filtered_articles = []
        print(f"\nФильтрация по длине текста (минимум {min_text_length} символов)")
        for article in articles:
            try:
                if len(article.get('text', '')) >= min_text_length:
                    filtered_articles.append(article)
            except Exception as e:
                print(f"Ошибка при фильтрации статьи: {e}")
        print(f"После фильтрации осталось статей: {len(filtered_articles)}")
        return filtered_articles
    def average_text_length(self, articles):
        if not articles:
            return 0
        total_length = 0
        for article in articles:
            text = article.get('text', article.get('title', ''))
            total_length += len(text)
        return round(total_length / len(articles), 2)
    def save_to_csv(self, filename, articles, source):
        date_analysis = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        avg_length = self.average_text_length(articles)
        count = len(articles)
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Дата анализа',
                'Источник',
                'Средняя длина текста',
                'Количество статей'
            ])
            writer.writerow([
                date_analysis,
                source,
                avg_length,
                count
            ])
            writer.writerow([])
            fieldnames = ['title', 'text', 'date', 'url']
            dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
            dict_writer.writeheader()
            for article in articles:
                dict_writer.writerow({
                    'title': article.get('title', ''),
                    'text': article.get('text', ''),
                    'date': article.get('date', '').strftime('%Y-%m-%d') if article.get('date') else '',
                    'url': article.get('url', '')
                })
