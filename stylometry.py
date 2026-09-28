import re
from collections import Counter, defaultdict
import pymorphy3
import nltk
from razdel import sentenize, tokenize
from nltk.util import ngrams
nltk.download('punkt')

class StylometricAnalyzer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def analyze(self, text):
        results = {}

        tokens = [t.text for t in tokenize(text) if t.text.isalpha()]
        lemmas = [self.morph.parse(token)[0].normal_form for token in tokens]
        pos_tags = [self.morph.parse(token)[0].tag.POS for token in tokens if self.morph.parse(token)[0].tag.POS]

        results['char_frequencies'] = self.char_frequencies(text)

        results['avg_sentence_length'] = self.avg_sentence_length(text)
        results['avg_word_length'] = self.avg_word_length(tokens)
        results['lexical_density'] = self.lexical_density(lemmas)
        results['lexical_diversity'] = self.lexical_diversity(lemmas)

        results['pos_ngrams'] = self.pos_ngrams(pos_tags, n=2)
        return results
    def char_frequencies(self, text):
        text = text.lower()
        letters_only = re.findall(r'[а-яё]', text)
        counter = Counter(letters_only)
        total = sum(counter.values())
        return {char: round(count / total, 4) for char, count in counter.items()}
    def avg_sentence_length(self, text):
        sentences = list(sentenize(text))
        if not sentences:
            return 0
        word_counts = [len([t for t in tokenize(sent.text) if t.text.isalpha()]) for sent in sentences]
        return round(sum(word_counts) / len(word_counts), 2)
    def avg_word_length(self, tokens):
        if not tokens:
            return 0
        return round(sum(len(word) for word in tokens) / len(tokens), 2)
    def lexical_density(self, lemmas):
        content_pos = {'NOUN', 'ADJF', 'VERB', 'ADVB'}
        content_words = [lemma for lemma in lemmas
                         if self.morph.parse(lemma)[0].tag.POS in content_pos]
        return round(len(content_words) / len(lemmas), 4) if lemmas else 0
    def lexical_diversity(self, lemmas):
        return round(len(set(lemmas)) / len(lemmas), 4) if lemmas else 0
    def pos_ngrams(self, pos_tags, n=2):
        if len(pos_tags) < n:
            return {}
        ng = list(ngrams(pos_tags, n))
        return dict(Counter(ng))
