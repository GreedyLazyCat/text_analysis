import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import re
from collections import Counter
import pymorphy3
from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize as razdel_tokenize

class RecentNewsParser:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")
        self.driver = None
        self.STOP_WORDS = {
            'который', 'которые', 'которой', 'котором', 'которых',
            'этот', 'этого', 'этому', 'этим', 'этом',
            'весь', 'всего', 'всему', 'всем', 'всём',
            'свой', 'своего', 'своему', 'своим', 'своём'
        }
    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)
        return self.driver
    def select_all_time_period(self):
        try:
            period_filter = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "list-date"))
            )
            self.driver.execute_script("arguments[0].click();", period_filter)
            time.sleep(1)
            all_time_option = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "li[data-range-days='all']"))
            )
            self.driver.execute_script("arguments[0].click();", all_time_option)
            time.sleep(3)
            return True
        except:
            return False
    def load_more_articles(self, max_clicks=15):
        clicks = 0
        while clicks < max_clicks:
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                more_button = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "list-more"))
                )
                self.driver.execute_script("arguments[0].click();", more_button)
                time.sleep(2)
                clicks += 1
            except:
                break
        return clicks
    def parse_site_ria(self, url):
        self.setup_driver()
        self.driver.get(url)
        time.sleep(3)
        self.select_all_time_period()
        self.load_more_articles()
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        items = soup.find_all('a', class_='list-item__title')
        articles = []
        for item in items:
            title = item.get_text(strip=True)
            link = item.get('href')
            if title and link:
                articles.append({
                    'title': title,
                    'link': link,
                    'date': self.extract_date_from_url(link)
                })
        self.driver.quit()
        return articles
    def parse_ria_article(self, url):
        self.setup_driver()
        self.driver.get(url)
        time.sleep(3)
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.article__text"))
            )
        except:
            pass
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        title = soup.find('h1', class_='article__title')
        date = soup.find('div', class_='article__info-date')
        blocks = soup.find_all('div', class_='article__text')
        raw_text = "\n".join(b.get_text(strip=True) for b in blocks if b)
        processed_text = self.process_text(raw_text)
        self.driver.quit()
        return {
            'title': title.get_text(strip=True) if title else '',
            'date': self.parse_article_date(date.get_text(strip=True) if date else ''),
            'url': url,
            'text': processed_text
        }
    def process_text(self, text):
        tokens = [_.text.lower() for _ in razdel_tokenize(text)]
        tokens = [t for t in tokens if t.isalpha() and t not in self.STOP_WORDS]
        lemmas = [self.morph.parse(t)[0].normal_form for t in tokens]
        stems = [self.stemmer.stem(lemma) for lemma in lemmas]
        return " ".join(stems)
    def parse_article_date(self, date_str):
        match = re.search(r'(\d{2}\.\d{2}\.\d{4})', date_str)
        if match:
            return datetime.strptime(match.group(1), '%d.%m.%Y')
        return None

    def extract_date_from_url(self, url):
        match = re.search(r'/(\d{4})(\d{2})(\d{2})/', url)
        if match:
            return datetime.strptime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", '%Y-%m-%d')
        return None
