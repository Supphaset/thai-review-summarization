import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tag import pos_tag
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import tensorflow_text


usem_model = hub.load(
    'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')


def baseline_model(sentences_list, review_ids, n=5, sim_matrix=None, is_dup_id=True, patient=3, drop_redundant=True, threshold=0.7):
    '''
    *Fallback when pagerank don't converge*

    sentences_list: list of review sentences that have been segmented.
    review_ids: list of review id of each sentences.
    n: number of review wanted to be return
    sim_matrix: cosine similarity matrix if the matrix(for evalation)
    is_dup_id: if you want duplicate id or not if True the review return could have the same review ID
    patient: number of max iteration when skipping redundat sentences and duplicate review ID
    threshold: used to drop the redundant sentences

    Baseline model:
        1. get similarity metrix of all review and ranked them by the mean cosine similarity score.
        2. rank the sentences based on highest similarity mean
        3. if drop_redundant: the sentences that have simliarity score higher than threshold, compare to selected review, will be drop
        4. if is_dup_id: change sentences to the sentences that have the higest cosine similarity with the senctences.
        5. return selected n sentences
    '''
    if sim_matrix is None:
        sim_matrix, _ = get_graph(sentences_list)
    scores = np.mean(sim_matrix, axis=0)
    ranked_sentences = sorted(([scores[i], i, s, review_ids[i]] for i, s in enumerate(
        sentences_list)), reverse=True, key=lambda score: score[0])
    ranked_index = [review[1] for review in ranked_sentences]
    if drop_redundant:
        ranked_index = drop_redundant_review(
            ranked_index, sim_matrix, threshold, patient, n)
    else:
        ranked_index = ranked_index[:n]
    if not is_dup_id:
        ranked_index = change_dup_id(
            ranked_index, sim_matrix, review_ids, patient)
    return get_output(ranked_index, sentences_list, review_ids, scores)


def get_textrank_mod(sentences_list, review_ids, n=5, is_dup_id=True, patient=3, personalize=None, drop_redundant=True, threshold=0.7):
    '''
    sentences_list: list of review sentences that have been segmented.
    review_ids: list of review id of each sentences.
    n: number of review wanted to be return
    is_dup_id: if you want duplicate id or not if True the review return could have the same review ID
    patient: number of max iteration when skipping redundat sentences and duplicate review ID
    personalize: dict where keys are index of the sentences(for personalize pagerank)
    drop_redundant: if True the sentences that have simliarity score higher than threshold, compare to selected review, will be drop
    threshold: used to drop the redundant sentences

    Personalize Pagerank with drop redundant:
        1. construct grap based on similarity matrix (similarity matrix is a matrix of pairwise cosine similarity of every sentences)
        2. rank the sentences using personalize pagerank algorithm
        3. if drop_redundant: the sentences that have simliarity score higher than threshold, compare to selected review, will be drop
        4. if is_dup_id: change sentences to the sentences that have the higest cosine similarity with the senctences.
        5. return selected n sentences
    '''
    sim_matrix, nx_graph = get_graph(sentences_list)
    if personalize is not None:
        personalize = dict([int(key), value]
                           for key, value in personalize.items())
    scores = nx.pagerank(nx_graph, max_iter=5000, personalization=personalize)
    ranked_sentences = sorted(([scores[i], i, s, review_ids[i]] for i, s in enumerate(
        sentences_list)), reverse=True, key=lambda score: score[0])
    ranked_index = [review[1] for review in ranked_sentences]
    if drop_redundant:
        ranked_index = drop_redundant_review(
            ranked_index, sim_matrix, threshold, patient, n)
    else:
        ranked_index = ranked_index[:n]
    if not is_dup_id:
        ranked_index = change_dup_id(
            ranked_index, sim_matrix, review_ids, patient)
    return get_output(ranked_index, sentences_list, review_ids, scores)


def drop_redundant_review(ranked_index, sim_matrix, threshold, patient, n):
    temp_ranked_index = []
    i = 0
    while len(temp_ranked_index) < n:
        temp_ranked_index.append(ranked_index[i])
        for j in range(patient if patient < len(ranked_index)-i-n+len(temp_ranked_index) else len(ranked_index)-i-1-n+len(temp_ranked_index)):
            temp2 = ranked_index[i+j+1]
            is_sim = any(sim_matrix[temp_ranked_index, temp2] > threshold)
            if not is_sim or j == patient-1:
                i += j+1
                break
    return temp_ranked_index


def get_output(ranked_index, sentences_list, review_id_list, scores=None):
    '''
    ranked_index: index of ranked sentences (sorted  by score)
    sentences_list: list of all review sentences
    review_id_list: list of all review 
    scores: score from the pagerank algorithm

    convert to output format.
    '''
    if scores is not None:
        return [{'text': sentences_list[i], 'review_id':review_id_list[i], 'score': scores[i]} for i in ranked_index]
    return [{'text': sentences_list[i], 'review_id':review_id_list[i]} for i in ranked_index]


def change_dup_id(index_list, sim_matrix, review_id_list, patient=3):
    '''
    index_list: list of top n selected sentences
    sim_matrix: pairwise cosine similarity of every sentences
    review_list: list of top n selected review ID
    patient: max iteration that the algorithm will skip if it get the same review id

    change sentences that is duplicated with the selected sentences.
    '''
    new_index_list = []
    temp_ids = []
    for index in index_list:
        temp_sim_matrix = sim_matrix[index].copy()
        temp_sim_matrix[[index_list]] = 0
        count = 0
        tmp_index = index
        while review_id_list[index] in temp_ids and count <= patient:
            index = np.argmax(temp_sim_matrix)
            temp_sim_matrix[index] = 0
            count += 1
            if count >= patient:
                index = tmp_index
                break
        new_index_list.append(index)
        temp_ids.append(review_id_list[index])
    return new_index_list


def get_graph(sentences_list):
    '''
    sentences_list: list of sentences that are segmented

    return pairwise cosine similarity metrix and graph construct from similarity matrix
    '''
    emb_list = usem_model(sentences_list)
    sim_matrix = get_cos_sim_matrix(emb_list)
    graph = nx.from_numpy_array(sim_matrix)
    return sim_matrix, graph


def get_cos_sim_matrix(emb_list):
    '''
    emb_list: list of embedding vector of all sentences

    return pairwise cosine similarity metrix
    '''
    sim_matrix = cosine_similarity(emb_list, emb_list)
    sim_matrix -= np.identity(sim_matrix.shape[0])
    return sim_matrix
