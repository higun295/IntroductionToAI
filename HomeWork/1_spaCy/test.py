import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy

candidate_sentences = pd.read_csv('./data/wiki_sentences_v2.csv')
candidate_sentences.shape