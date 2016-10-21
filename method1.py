import requests
import re
import nltk
import pymorphy2
import numpy as np
from html2text import HTML2Text
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def classify(web_url, topn=10):
    t = """/Наука и образование
    /Наука и образование/наука
    /Наука и образование/наука/математика
    /Наука и образование/наука/физика
    /Наука и образование/наука/химия
    /Наука и образование/наука/информатика
    /Наука и образование/наука/информатика/биоинформатика
    /Наука и образование/наука/информатика/анализ данных
    /Наука и образование/наука/литература
    /Наука и образование/образование
    /Наука и образование/образование/школьное
    /Наука и образование/образование/высшее
    /Наука и образование/образование/дополнительное
    /Наука и образование/образование/дополнительное/GoTo
    /Политика
    /Политика/Внутренняя
    /Политика/Внешняя
    /Экономика и бизнес
    /Экономика и бизнес/Бизнес
    /Экономика и бизнес/Бизнес/Стартапы
    /Экономика и бизнес/Бизнес/Стартапы/E-Contenta
    /Экономика и бизнес/Бизнес/Крупные компании
    /Экономика и бизнес/Экономика
    /Отдых и развлечения
    /Отдых и развлечения/Кино
    /Отдых и развлечения/Театр
    /Отдых и развлечения/Компьютерные игры
    /Здоровье и красота/Фитнес
    /Здоровье и красота/Медицина
    /Здоровье и красота/Косметология""".split("\n")
    topics = [x[1:].split("/")[-1] for x in t]
    restore = {topics[i]: t[i] for i in range(len(topics))}
    
    
    # получим текст
    page = requests.get(web_url)
    soup = BeautifulSoup(page.content, "lxml")
    h = HTML2Text()
    h.ignore_links = True
    raw_text = h.handle(soup.body.text)
    only_russian_text = re.sub("[^а-яА-Я]", " ", raw_text).strip()
    
    # оставим только слова без стоп-слов длины [3, 15] и только существительные и глаголы в нормальной форме
    morph = pymorphy2.MorphAnalyzer()
    words = only_russian_text.split()
    stoplist = stopwords.words('russian')
    data = []
    for word in words:
        if 2 < len(word) < 16 and word not in stoplist:
            p = morph.parse(word)[0]
            if p.tag.POS == 'NOUN' or "VERB":
                data.append(str(p.normal_form))
                
    morph = pymorphy2.MorphAnalyzer()
    count_vect = CountVectorizer()
    tf_idf = TfidfTransformer()
    X = count_vect.fit_transform([" ".join(data)])
    X = tf_idf.fit_transform(X)
    
    tfidf_repr = sorted([(a[0][1], a[1]) for a in X.todok().items()], key=lambda a: a[1], reverse=True)
    tfidf_dict = {id_ : tfidf for id_, tfidf in tfidf_repr}
    
    good_vocab = {str(_id) : word for word, _id in count_vect.vocabulary_.items()}
    
    res = []
    a = []
    b = []
    for k in topics:
        # сначала название топика приведем к нормальной форме, чтобы лучше найти его в count_vect
        word = morph.parse(k)[0].normal_form
        _id = count_vect.vocabulary_.get(word)
        if _id:
            # по слову - айдишник, по айдишнику - tfidf значение. Если оно существует, то добавляем топик и tfidf в соотв. массивы
            tfidf = tfidf_dict.get(_id)
            if tfidf:
                a.append(restore[k])
                b.append(tfidf)
                
    v = np.array(b)
    if np.sum(v) == 0:
        print("Вероятно, вы показываете мне английский текст, я так не умею")
        print("Ну или вы нашли сайт, который не подходит ни под одну категорию")
        return []
    multiplier = 100 / np.sum(v)
    b = list(v * multiplier)

    # только сейчас добавляем в результирующий массив изначальное название топика и его процентное содержание
    for i in range(len(a)):
        res.append("{} — {:.2f}%".format(a[i].capitalize(), b[i]))
        
    return sorted(res, key=lambda x: float(x.split(" — ")[1][:-1]), reverse=True)[:topn]