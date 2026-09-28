from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import json
import time
import re
from urllib.parse import urljoin, urlparse

class OldNewsParser:
    def __init__(self):
        self.driver = None
        self.base_url = "https://www.gazeta.ru"
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
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, '_id_search_result'))
            )
            time.sleep(1)
            for i in range(max_clicks):
                try:
                    show_more_btn = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable(
                            (By.ID, '_id_gazeta_show_more_btn')
                        )
                    )
                    if 'switch_off' in show_more_btn.get_attribute('class').split():
                        break
                    time.sleep(0.5)  # Даем кнопке стабилизироваться перед нажатием
                    old_count = len(self.driver.find_elements(By.CSS_SELECTOR, 'a.b_ear'))
                    self.driver.execute_script("arguments[0].click();", show_more_btn)
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: len(driver.find_elements(By.CSS_SELECTOR, 'a.b_ear')) > old_count
                        or 'switch_off' in show_more_btn.get_attribute('class').split()
                    )
                except Exception:
                    break
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.b_ear'))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            results = []
            seen = set()
            for a in soup.select('a.b_ear'):
                href = a.get("href")
                title_node = a.select_one('.b_ear-title')
                title = title_node.get_text(' ', strip=True) if title_node else ''
                if not title or not href or href in seen:
                    continue
                href = urljoin(self.base_url, href)
                if urlparse(href).netloc not in {'www.gazeta.ru', 'gazeta.ru'}:
                    continue
                seen.add(href)
                date_node = a.select_one('time.b_ear-time')
                results.append({
                    'title': title,
                    'link': href,
                    'date': self.extract_date_from_url(href)
                    or self.parse_search_date(date_node.get_text(' ', strip=True) if date_node else '')
                })
            return results
        except Exception as e:
            print(f"Ошибка парсинга {url}: {e}")
            return []
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

    def parse_search_date(self, value):
        match = re.search(r'(\d{2}\.\d{2}\.\d{4})', value)
        if match:
            return datetime.strptime(match.group(1), '%d.%m.%Y')
        return None

    def parse_gazeta_article(self, url):
        try:
            if not self.setup_driver():
                return None
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                lambda driver: driver.find_elements(By.CSS_SELECTOR, 'h1')
                or driver.find_elements(By.CSS_SELECTOR, '[itemprop="articleBody"]')
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            text = ''
            article_body = soup.select_one('[itemprop="articleBody"]')
            if article_body:
                text = article_body.get_text('\n', strip=True)
            if not text:
                for selector in (
                    'div.article__text',
                    'div.b_article-text',
                    'div.b_article-body',
                    'article .article-body',
                    'article',
                ):
                    blocks = soup.select(selector)
                    if blocks:
                        text = '\n'.join(
                            block.get_text('\n', strip=True)
                            for block in blocks
                            if block.get_text(strip=True)
                        )
                        if text:
                            break
            if not text:
                for script in soup.select('script[type="application/ld+json"]'):
                    try:
                        data = json.loads(script.string or script.get_text())
                    except (TypeError, json.JSONDecodeError):
                        continue
                    records = data if isinstance(data, list) else [data]
                    for record in records:
                        if isinstance(record, dict) and record.get('articleBody'):
                            text = record['articleBody'].strip()
                            break
                    if text:
                        break
            title = soup.find("h1")
            title_text = title.get_text(strip=True) if title else ""
            return {
                'title': title_text,
                'text': text,
                'date': self.extract_date_from_url(url.replace(self.base_url, '')),
                'url': url
            }
        except Exception as e:
            print(f"Ошибка парсинга статьи {url}: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
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
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['title', 'text', 'date', 'url'],
                lineterminator='\n'
            )
            writer.writeheader()
            for article in articles:
                date = article.get('date')
                writer.writerow({
                    'title': article.get('title', ''),
                    'text': article.get('text', ''),
                    'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date or '',
                    'url': article.get('url', '')
                })
