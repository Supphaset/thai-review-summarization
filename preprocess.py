import os
import numpy as np
import pandas as pd
import re
from pythainlp import sent_tokenize
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from transformers import pipeline
from operator import itemgetter
from more_itertools import unique_everseen
from sklearn.preprocessing import minmax_scale


# ===================================== Import Model==============================================
sentiment_model = pipeline(
    "text-classification",
    model='poom-sci/WangchanBERTa-finetuned-sentiment',
    tokenizer='poom-sci/WangchanBERTa-finetuned-sentiment'
)

not_food_word_list = []
with open(os.path.join(os.path.dirname(__file__), 'corpus/not_food_word.txt'), 'r') as fp:
    for item in fp:
        not_food_word_list.append(item[:-1])

food_list = []
with open(os.path.join(os.path.dirname(__file__), 'corpus/food_list.txt'), 'r') as fp:
    for item in fp:
        food_list.append(item[:-1])


def review_segmentation(sentences_list, review_id_list):
    '''
    sentences_list: list of all reivew before segmentation
    review_id_list: list of all review id before segmentation

    Segmentation:
    - reviews that have str len less than 120 will not be segmented
    - reviews that are in form of bullet will be split with "\n-", "\n -","\n[int]."
        - if reviews is still longer than 120 str len use segmentation from pythainlp
    - review that are longer than 120 str len use sentences segmentation from pythainlp
    '''
    review_list = []
    id_list = []
    for reviews, review_id in zip(sentences_list, review_id_list):
        temp_review_list = []
        reviews = reviews.strip('\n\r\t .!*#@').replace('\u200b', '')
        text_list = re.split('\\n\\r\\n|\\n\\n\\r|\\n\\n', reviews)
        for text in text_list:
            is_par = bool(re.search('\(.+\)', text))
            if not is_english(text):
                if len(text) <= 120:  # Short text
                    temp_review_list = preprocessing([text], temp_review_list)
                elif '\n' in text:
                    if '\n-' in text or '\n -' in text:  # Bullet point "- asdfasdf -asdfasdf"
                        temp_list = re.split('\\n-|\\n -', text)
                    elif re.search('\\n.\.', text):  # Bullet point "1. 2. 3."
                        temp_list = re.split('\\n.\.', text)
                    else:  # Bullet piont / Long text
                        temp_list = text.split('\n')
                    for temp in temp_list:
                        if len(temp) <= 120:
                            temp_review_list = preprocessing(
                                [temp], temp_review_list)
                        else:
                            temp_review_list = preprocessing(
                                get_sent_tokenize(temp, is_par), temp_review_list)
                else:
                    temp_review_list = preprocessing(
                        get_sent_tokenize(text, is_par), temp_review_list)
        review_list.extend(temp_review_list)
        id_list.extend([review_id]*len(temp_review_list))
    unique = unique_everseen(zip(review_list, id_list), key=itemgetter(0))
    unique_list = list(zip(*unique))
    return list(unique_list[0]), list(unique_list[1])


def review_segmentation_w_date(sentences_list, review_id_list, reviewed_date_list):
    '''
    *use this method if reviewd_date is required

    sentences_list: list of all reivew before segmentation
    review_id_list: list of all review id before segmentation
    reviewd_date_list: list of all reviewd_date before segmentation

    Segmentation:
    - reviews that have str len less than 120 will not be segmented
    - reviews that are in form of bullet will be split with "\\n-", "\\n -","\\n[int]."
        - if reviews is still longer than 120 str len use segmentation from pythainlp
    - review that are longer than 120 str len use sentences segmentation from pythainlp
    '''
    review_list = []
    id_list = []
    date_list = []
    for reviews, review_id, review_date in zip(sentences_list, review_id_list, reviewed_date_list):
        temp_review_list = []
        reviews = reviews.strip('\n\r\t .!*#@').replace('\u200b', '')
        text_list = re.split('\\n\\r\\n|\\n\\n\\r|\\n\\n', reviews)
        for text in text_list:
            is_par = bool(re.search('\(.+\)', text))
            if not is_english(text):
                if len(text) <= 120:  # Short text
                    temp_review_list = preprocessing([text], temp_review_list)
                elif '\n' in text:
                    if '\n-' in text or '\n -' in text:  # Bullet point "- asdfasdf -asdfasdf"
                        temp_list = re.split('\\n-|\\n -', text)
                    elif re.search('\\n.\.', text):  # Bullet point "1. 2. 3."
                        temp_list = re.split('\\n.\.', text)
                    else:  # Bullet piont / Long text
                        temp_list = text.split('\n')
                    for temp in temp_list:
                        if len(temp) <= 120:
                            temp_review_list = preprocessing(
                                [temp], temp_review_list)
                        else:
                            temp_review_list = preprocessing(
                                get_sent_tokenize(temp, is_par), temp_review_list)
                else:
                    temp_review_list = preprocessing(
                        get_sent_tokenize(text, is_par), temp_review_list)
        review_list.extend(temp_review_list)
        id_list.extend([review_id]*len(temp_review_list))
        date_list.extend([review_date]*len(temp_review_list))
    unique = unique_everseen(
        zip(review_list, id_list, date_list), key=itemgetter(0))
    unique_list = list(zip(*unique))
    return list(unique_list[0]), list(unique_list[1]), list(unique_list[2])


def get_sent_tokenize(text, is_par):
    '''
    text: reviews that have str len more than 120
    is_par: if the reviews contain "(" and ")"

    return segmented sentence using pythianlp "sent_tokenize"
    '''
    sentences_list = sent_tokenize(text)
    return parenthesis_join(sentences_list) if is_par else sentences_list


def parenthesis_join(sentences_list):
    '''
    sentences_list: segmented sentences list

    complete parenthesis -> if sentences contain "(" must also contain ")"
    '''
    new_sent_list = []
    is_open = False
    temp = []
    for sentence in sentences_list:
        if not bool(re.search('\(.+\)', sentence)) and bool(re.search('\(', sentence)):
            is_open = True
            temp.append(sentence)
        elif is_open:
            temp.append(sentence)
            if bool(re.search('\)', sentence)):
                temp_sent = ''.join(temp)
                new_sent_list.append(temp_sent)
                is_open = False
        else:
            new_sent_list.append(sentence)
    return new_sent_list


def preprocessing(text_list, review_list):
    '''
    text_list: segmented of 1 review list
    review_list: segmented of all review list

    clean text -> drop too short and too long text
        - strip text
    '''
    for text in text_list:
        text = text.strip(" .-\rnt")
        text = re.sub("\\n|\\r", " ", text)
        if len(text) >= 18 and len(text) < 180 and not bool(re.search('|'.join(not_food_word_list), text)) and is_complete(text):
            review_list.append(text)
    return review_list


def is_english(sentence):
    '''
    sentence: review sentence

    retrun if the review is reviewed in English.
    '''
    try:
        sentence.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_complete(sentence):
    '''
    sentence: review sentence

    return if the reivew contains "NN" (Noun)
    '''
    word_w_pos_tag = pos_tag(word_tokenize(sentence), corpus='lst20')
    tag = [x[1] for x in word_w_pos_tag]
    return 'NN' in tag


def seperate_review_by_sentiment(sentences_list, review_id_list, reviewed_date=None, is_personalize=False, sentiment='pos'):
    '''
    sentences_list: list of segmented review sentences
    review_id_list: list of segmented review ID
    reviewd_date: list of segmented review date
    is_personalize: if True return personalize dict else not
    sentiment: which sentiment wanted

    select only sentence that have the same sentiment as sentiment wanted
    if is_personalize:
        1. scale reviewed_date in range of 0-0.5 with minmax_scale()/2
        2. if contains food word:
            i. personalize = 0.5 + sclaed_reviewed_date
            ii. else: personalize = 0
    '''
    reviews = []
    ids = []
    sentiments = sentiment_model(sentences_list)
    if is_personalize:
        if reviewed_date is not None:
            reviewed_date = minmax_scale(reviewed_date)/2
        personalize = dict()
        count = 0
        for i, sent in enumerate(sentiments):
            if sent['label'] == sentiment:
                if bool(re.search('|'.join(food_list), sentences_list[i])):
                    personalize[str(count)] = 0.5 + \
                        reviewed_date[i] if reviewed_date is not None else 1
                else:
                    personalize[str(count)] = 0
                reviews.append(sentences_list[i])
                ids.append(review_id_list[i])
                count += 1
        return reviews, ids, personalize
    for i, sent in enumerate(sentiments):
        if sent['label'] == sentiment:
            reviews.append(sentences_list[i])
            ids.append(review_id_list[i])
    return reviews, ids
