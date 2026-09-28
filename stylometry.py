import re
from collections import Counter

import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
POS_TAGS = (
    "NOUN", "ADJF", "ADJS", "COMP", "VERB", "INFN", "PRTF", "PRTS",
    "GRND", "NUMR", "ADVB", "NPRO", "PRED", "PREP", "CONJ", "PRCL",
    "INTJ"
)
POS_BIGRAMS = tuple((first, second) for first in POS_TAGS for second in POS_TAGS)


class StylometricAnalyzer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def analyze(self, text):
        tokens = [token.lower() for token in word_tokenize(text, language='russian') if token.isalpha()]
        analyses = [self.morph.parse(token)[0] for token in tokens]
        lemmas = [analysis.normal_form for analysis in analyses]
        pos_tags = [analysis.tag.POS for analysis in analyses if analysis.tag.POS]

        return {
            'char_frequencies': self.char_frequencies(text),
            'avg_sentence_length': self.avg_sentence_length(text),
            'avg_word_length': self.avg_word_length(tokens),
            'lexical_density': self.lexical_density(analyses, len(lemmas)),
            'lexical_diversity': self.lexical_diversity(lemmas),
            'pos_ngrams': self.pos_ngrams(pos_tags)
        }

    def char_frequencies(self, text):
        letters = re.findall(r'[а-яё]', text.lower())
        total = len(letters)
        if not total:
            return {char: 0.0 for char in RUSSIAN_ALPHABET}
        counts = Counter(letters)
        return {
            char: round(counts[char] / total, 6)
            for char in RUSSIAN_ALPHABET
        }

    def avg_sentence_length(self, text):
        sentences = sent_tokenize(text, language='russian')
        if not sentences:
            return 0
        word_counts = [
            sum(1 for token in word_tokenize(sentence, language='russian') if token.isalpha())
            for sentence in sentences
        ]
        return round(sum(word_counts) / len(word_counts), 2)

    def avg_word_length(self, tokens):
        return round(sum(map(len, tokens)) / len(tokens), 2) if tokens else 0

    def lexical_density(self, analyses, total_words):
        content_pos = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'ADVB'}
        content_words = sum(
            analysis.tag.POS in content_pos for analysis in analyses
        )
        return round(content_words / total_words, 4) if total_words else 0

    def lexical_diversity(self, lemmas):
        return round(len(set(lemmas)) / len(lemmas), 4) if lemmas else 0

    def pos_ngrams(self, pos_tags):
        if len(pos_tags) < 2:
            return {f'{first}|{second}': 0.0 for first, second in POS_BIGRAMS}
        counts = Counter(ngrams(pos_tags, 2))
        total = sum(counts.values())
        return {
            f'{first}|{second}': round(counts[(first, second)] / total, 6)
            for first, second in POS_BIGRAMS
        }
