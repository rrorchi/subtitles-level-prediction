##
####################################################################
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
import re, pysrt

from tempfile import NamedTemporaryFile
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
####################################################################
## уровни CEFR
CATS = ['A2', 'B1', 'B2', 'C1']

####################################################################
## определение pos-тега слова
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

####################################################################
## удаление лишних символов и стоп-слов, лемматизация
def sub_preprocess(s):
    sub_prep = re.sub(r"[\d\n]", ' ', # удаляем цифры и пробелы
                re.sub("[^a-z]", ' ', # удаляем все знаки
                    re.sub(r"\<[^<>]+\>", ' ', # удаляем все что в скобках
                        re.sub(r"\([^()]+\)", ' ', s.lower()))))
    sub_prep = re.findall(r'\S+', sub_prep)
    sub_prep = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in sub_prep]
    sub_prep = ' '.join([word for word in sub_prep if word not in stopwords.words('english')])
    return sub_prep
	
## работа с файлами
####################################################################
## загрузка субтитров
def open_sub(file):
	with NamedTemporaryFile(suffix='.srt', delete=False) as tempfile:
		tempfile.write(file.getbuffer())
		return pysrt.open(tempfile.name, encoding='iso-8859-1').text

####################################################################
## предказание уровня и вероятностей
def get_prediction(sub):
	with open('vectorizer.pickle', 'rb') as f:
		tfp = pickle.load(f)
		
	subtf = tfp.transform([sub]).toarray()

	with open('model.pickle', 'rb') as f:
		model = pickle.load(f)
		
	pred = model.predict(subtf)
	proba = pd.DataFrame(data=np.round(model.predict_proba(subtf) * 100), columns=CATS, index=[' ']).T
	
	return pred, proba
	
####################################################################
## статистика по субтитру
def get_statistics(sub, level):
	with open('wordlist.pickle', 'rb') as f:
		oxford_list = pickle.load(f)

	sub_stat = []
	for j in oxford_list['level']:
		stats_ = []
		words = oxford_list[oxford_list['level']==j]['words'].values[0]
		vb = CountVectorizer()
		vb.fit(words)

		# слова субтитра, имеющиеся в словаре
		imp = list(filter(lambda count: count > 0, vb.transform([sub]).toarray()[0]))
		# уникальные слова в субтитре
		unique = len(np.unique(sub.split()))
		# все слова в субтитре
		total = len(sub.split())

		stats_.append([sum(imp), len(imp), unique, total])
		sub_stat.append([level, j, *pd.DataFrame(stats_).mean().values])
		
	sub_stat = pd.DataFrame(sub_stat, columns=['sub_level', 'dict_level', 'dict_words_sum', 'dict_words_uniq',
												 'sub_words_uniq', 'sub_words_total'])
	return sub_stat

####################################################################
## статистика по субтитрам того же уровня
def get_info(level):
	with open('stats.pickle', 'rb') as f:
		stats = pickle.load(f)
	return round(stats[stats['sub_level'] == level], 1)
##
####################################################################
