import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet, stopwords, words
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
from itertools import chain
from googletrans import Translator, LANGUAGES
import gensim.downloader as api

random.seed(0)

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('words')

WORD_LIST = words.words()

# Word2Vec = api.load("word2vec-google-news-300")

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the 
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
TYPO_MAPPING = {
    'a': ['s', 'q', 'w', 'z', 'x', 'u', 'ei'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v', 's', 'ck', 'sc'],
    'd': ['e', 'r', 's', 'f', 'x', 'c', 't'],
    'e': ['w', 'r', 's', 'd', 'ee', 'ea', 'i', 'ei'],
    'f': ['r', 't', 'd', 'g', 'c', 'v', 'ph', 'gh'],
    'g': ['t', 'y', 'f', 'h', 'v', 'b', 'j'],
    'h': ['g', 'j', 'b', 'n', 'y', 'u'],
    'i': ['u', 'o', 'j', 'k', 'e', 'ee', 'ea', 'ai', 'ay', 'ie', 'y'],
    'j': ['u', 'i', 'h', 'k', 'n', 'm', 'g'],
    'k': ['i', 'o', 'j', 'l', 'm', 'ch', 'ck'],
    'l': ['o', 'p', 'k'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'g', 'h', 'j', 'm'],
    'o': ['i', 'p', 'k', 'l', 'oa', 'ou', 'oh', 'ow', 'au', 'ao'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'd', 'f'],
    's': ['a', 'd', 'w', 'x', 'e', 'z', 'th', 'c', 'sc'],
    't': ['r', 'y', 'f', 'g', 'd'],
    'u': ['y', 'i', 'h', 'j', 'a', 'oo'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'e', 'a', 's'],
    'x': ['z', 'c', 's', 'd'],
    'y': ['t', 'u', 'g', 'h', 'i', 'ie', 'ey'],
    'z': ['a', 's', 'x', 'th'],
    'ai': ['ay', 'i'],
    'ao': ['o', 'au', 'oa', 'ou', 'ow'],
    'au': ['o', 'ao', 'oa', 'ou', 'ow'],
    'ay': ['ai'],
    'ch': ['sh', 'tr', 'tch', 'c'],
    'ck': ['c', 'k', 'ch'],
    'ea': ['i', 'e', 'ee', 'ie'],
    'ee': ['i', 'e', 'ea', 'ie'],
    'ei': ['a', 'e', 'ie'],
    'gh': ['f'],
    'ie': ['i', 'y', 'ea', 'ee', 'ei'],
    'ph': ['f'],
    'mb': ['m'],
    'oa': ['o', 'ou', 'ow', 'ao', 'au'],
    'ou': ['o', 'oa', 'ow', 'ao', 'au'],
    'ow': ['o', 'oa', 'ou', 'ao', 'au'],
    'oo': ['u', 'o'],
    'sc': ['s'],
    'sh': ['ch', 's'],
    'ti': ['sh'],
    'th': ['s'],
    'tt': ['t'],
    'tr': ['ch'],
    'wr': ['r'],
    'wh': ['w'],
}


def to_typo(token):
    max_attempts = len(token) * 2
    attempts = 0

    while attempts < max_attempts:
        if random.random() < 0.5:
            start_index = random.randint(0, len(token) - 1)
            if start_index < len(token) - 1 and random.choice([True, False]):
                end_index = start_index + 2
            else:
                end_index = start_index + 1

            substring = token[start_index:end_index]
            if substring in TYPO_MAPPING:
                token = token[:start_index] + random.choice(TYPO_MAPPING[substring]) + token[end_index:]
                break
        else:
            if len(token) > 1:
                start_index = random.randint(0, len(token) - 2)
                end_index = start_index + 1
                if end_index < len(token):
                    token = token[:start_index] + token[end_index] + token[start_index] + token[end_index + 1:]
                    break

        attempts += 1

    return token


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def get_synonyms(word, pos):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def tag_to_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def back_translate(text):
    translator = Translator(service_urls=['translate.googleapis.com'])
    non_english_languages = [lang for lang in LANGUAGES.values() if lang != 'english']
    random_language_1 = random.choice(non_english_languages)
    first_translated = translator.translate(text, src='en', dest=random_language_1).text
    random_language_2 = random.choice(non_english_languages)
    second_translated = translator.translate(first_translated, src=random_language_1, dest=random_language_2).text
    back_translated = translator.translate(second_translated, sec=random_language_2, dest='en').text

    return back_translated


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    # proportion = .15
    #
    # # if random.random() < proportion:
    # #     example["text"] = back_translate(example["text"])
    # #     return example
    #
    # tokens = word_tokenize(example["text"])
    # tagged_tokens = pos_tag(tokens)
    # num_to_replace = max(1, int(len(tokens) * proportion))
    # stop_words = set(stopwords.words('english'))
    # replaceable_indices = [i for i, (token, _) in enumerate(tagged_tokens)
    #                        if token.lower() not in stop_words and token.isalnum()]
    # replace_indices = random.sample(replaceable_indices, min(num_to_replace, len(replaceable_indices)))
    # new_tokens = []
    # for idx, token in enumerate(tokens):
    #     # if token.lower() in stop_words:
    #     if idx not in replace_indices:
    #         new_tokens.append(token)
    #         continue
    #     _, tag = tagged_tokens[idx]
    #     if random.random() < proportion:
    #         token = token.lower()
    #     if random.random() < proportion:
    #         wordnet_pos = tag_to_wordnet_pos(tag)
    #         # if random.random() < 0.5 or token not in Word2Vec.key_to_index:
    #         synonyms = get_synonyms(token, wordnet_pos)
    #         if synonyms:
    #             synonyms = [syn for syn in synonyms if syn.lower() != token.lower()]
    #             if synonyms:
    #                 token = synonyms[0]
    #         # else:
    #         #     synonyms = [synonym for synonym, _ in Word2Vec.most_similar(token, topn=10)]
    #         #     for synonym in synonyms:
    #         #         synsets = wordnet.synsets(synonym)
    #         #         if any(s.pos() == wordnet_pos for s in synsets):
    #         #             token = synonym
    #         #             break
    #     if random.random() < proportion:
    #         token = to_typo(token)
    #     if random.random() < proportion and idx < len(tokens) - 1:
    #         new_tokens.append(tokens[idx + 1])
    #         tokens[idx + 1] = token
    #         continue
    #     if random.random() < proportion:
    #         token = random.choice(WORD_LIST)
    #     if random.random() < proportion:
    #         token = ''
    #     new_tokens.append(token)
    # example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)
    proportion = .6

    tokens = word_tokenize(example["text"])
    tagged_tokens = pos_tag(tokens)
    num_to_replace = max(1, int(len(tokens) * proportion))
    stop_words = set(stopwords.words('english'))
    replaceable_indices = [i for i, (token, _) in enumerate(tagged_tokens)
                           if token.lower() not in stop_words and token.isalnum()]
    replace_indices = random.sample(replaceable_indices, min(num_to_replace, len(replaceable_indices)))
    for idx in replace_indices:
        token, tag = tagged_tokens[idx]
        if random.random() < .25:
            wordnet_pos = tag_to_wordnet_pos(tag)
            synonyms = get_synonyms(token, wordnet_pos)
            if synonyms:
                synonyms = [syn for syn in synonyms if syn.lower() != token.lower()]
                if synonyms:
                    tokens[idx] = synonyms[0]
        if random.random() < .25:
            token = token.lower()
            tokens[idx] = to_typo(token)
        if random.random() < .25 and idx < len(tokens) - 1:
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        if random.random() < .25:
            tokens[idx] = ''

    example["text"] = TreebankWordDetokenizer().detokenize(tokens)
    ##### YOUR CODE ENDS HERE ######

    return example
