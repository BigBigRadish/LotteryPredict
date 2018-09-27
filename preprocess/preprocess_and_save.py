# -*- coding: utf-8 -*-
'''
Created on 2018年9月27日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
import os
import pickle
import numpy as np
from collections import Counter
def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def preprocess_and_save_data(dataset_path, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    text = text.lower()
    #text = text.split()
    
    words = [word for word in text.split()]

    reverse_words = [text.split()[idx] for idx in (range(len(words)-1, 0, -1))]
    vocab_to_int, int_to_vocab = create_lookup_tables()#text
    #int_text = [vocab_to_int[word] for word in text]
    int_text = [vocab_to_int[word] for word in reverse_words]
    pickle.dump((int_text, vocab_to_int, int_to_vocab), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))

def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))
def create_lookup_tables():#创建一个查找表
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int = {str(ii).zfill(3) : ii for ii in range(1000)}
    int_to_vocab = {ii : str(ii).zfill(3) for ii in range(1000)}
    return vocab_to_int, int_to_vocab
data_dir = '../dataset/cp.txt'
text = load_data(data_dir)
view_sentence_range = (0, 10)
print('数据情况：')
print('不重复单词(彩票开奖记录)的个数: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('开奖期数: {}期'.format(int(np.average(sentence_count_scene))))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('行数: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('平均每行单词数: {}'.format(np.ceil(np.average(word_count_sentence))))    
print()
print('开奖记录从 {} 到 {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
#print(text)
preprocess_and_save_data(data_dir, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab = load_preprocess()
    