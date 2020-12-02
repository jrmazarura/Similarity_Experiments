# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:01:23 2020

@author: u10220420
"""
import pickle
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CGibbs_gpm import GPM

def initial_clean(text):
    """
    Function to clean text of websites, email addressess and any punctuation
    We also lower case the text
    Args:
        text: raw corpus

    Returns: tokenized corpus

    """
    text = re.sub(r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    return text


def remove_stop_words(text):
    """
    Function that removes all stopwords from tokenized corpus
    Args:
        text: corpus

    Returns: corpus w/o stopwords

    """
    stop_words = stopwords.words('english')
    stop_words.extend(['covid','asl','im','hey','hi','hello','caus','from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'et', 'al', 'without', 'use', 'figur', 'howev','background'])

    return [word for word in text if word not in stop_words]


def clean(text):
        
    return remove_stop_words(initial_clean(text))



def corpus_stats(df, name):
    print("Corpus statistics for "+str(name))
    
    doc_lengths = df['tokenized'].apply(lambda x: len(x))

    print("length of list:", len(doc_lengths),
          "\naverage document length", np.average(doc_lengths),
          "\nminimum document length", min(doc_lengths),
          "\nmaximum document length", max(doc_lengths))
    
    plt.hist(doc_lengths, bins='auto')
    plt.title("Histogram of document lengths")
    plt.show()
    
def cut_docs(x,max_length):
    return x
    #return x[0:max_length]    
    
def remove_docs(df,min_length,max_length):
    #cut docs
    df['tokenized'] = df['tokenized'].apply(lambda x: cut_docs(x,max_length))
    
    #remove short docs
    df = df[df['tokenized'].map(len) >= min_length]
    df2 = df[df['tokenized'].map(type) == list]          
    df2.reset_index(drop=True, inplace=True)    
    return df2

def save_obj(obj, name):
    """
    Save a data-type of interest as a .pkl file

    Args:
        obj (any): variable name of interest
        name (str): string-name for .pkl file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Load .pkl file from current working directory

    Args:
        name (str): name for .pkl file of interest

    Returns:
        [any]: unpacked .pkl file either in the form of a pd.DataFrame or list
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def jsd_mat(p, q):

    n_p = len(p)
    n_q = len(q)
    matrix = np.zeros((n_p, n_q))
    for j in range(n_q):
        for i in range(n_p):
            m=0.5*(p[i][:] + q[j][:])
            if entropy(p[i][:],m) > 0. and entropy(q[j][:],m) > 0:
                #matrix[i, j] = 1 - jensenshannon(p[i][:], q[j][:])  # ADJUST JSD HERE
                matrix[i, j]=1-np.sqrt(0.5*(entropy(p[i][:],m) + entropy(q[j][:],m)))
            else:
                matrix[i, j] = 1
    return matrix

def js_plots(data, labels, color=None, **kwargs):
    """
    Generates kde plots of jensen-shannon distances for categories specifed

    Args:
        data: list of two JSDs
        labels: list of category names
        color: list of 2 colors for plots
    Returns:
        KDE plot of JSDs on same axis
    """

    if color is None:
        color = ['orange', 'blue']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 1, figsize=(12.5, 10))

    [sns.distplot(data[i], ax=axs, kde=True, color=wd, **kwargs, kde_kws={'bw': 0.1}) for i, wd in enumerate(color)]
    axs.set_ylabel('Frequency', fontsize='medium')
    axs.set_xlabel('Jensen-Shannon Distances', fontsize='medium')

    axs.set_title('Jensen-Shannon distances of ' + labels[0] + ' and ' + labels[1],
                  fontsize='large')
    plt.tight_layout()
    # fig.subplots_adjust(top=0.88)
    fig.legend(labels=labels, fontsize='medium', loc="right")
    plt.savefig('GPM_kde' + labels[0] + labels[1] + '.jpg', bbox_inches="tight")

    plt.show()

def plot_histogram(x):
    plt.hist(x, bins='auto')
    plt.title("Histogram of document lengths")
    plt.show()
    
def js_plots_stacked(d1, d2, l1, l2, p1=None, p2=None, **kwargs):
    """
    Generates subplot kde plots of jensen-shannon distances for categories specifed stacked.
    Reference sets intended to be the same colors.
    Args:
        d1: list of two JSDs
        d2: list of two JSDs
        l1: list of category names for d1
        l2: list of category names for d2
        p1: colors for d1. Default = ['orange', 'blue']
        p2: colors for d2. Default = ['orange', 'green']
    Returns:
        sns.distplot() plot of JSDs on same axis on subplots
    """

    if p1 is None:
        p1 = ['orange', 'blue']
    if p2 is None:
        p2 = ['red', 'green']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 10))

    [sns.distplot(d1[i], ax=axs[0], kde=True, label=l1[i], color=wd, **kwargs, kde_kws={'bw': 0.1}) for i, wd in enumerate(p1)]
    [sns.distplot(d2[j], ax=axs[1], kde=True, label=l2[j], color=wd, **kwargs, kde_kws={'bw': 0.1}) for j, wd in enumerate(p2)]

    [axs[i].set_ylabel('Frequency', fontsize='x-large') for i in range(2)]
    [axs[i].set_xlabel('Adjusted Jensen-Shannon Distances', fontsize='x-large') for i in range(2)]

    axs[0].set_title('GPM: Adjusted Jensen-Shannon distances of ' + l1[0] + ' and ' + l1[1],
                     fontsize='x-large')
    axs[1].set_title('Word2vec: Soft-Cosine Similarities of ' + l2[0] + ' and ' + l2[1],
                     fontsize='x-large')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, hspace = 0.35)
    fig.legend(fontsize='x-large', ncol=2, loc='upper center',bbox_to_anchor=(0.48, 1.02))
    
    plt.savefig( 'PHD_NEWS_2020.jpg', bbox_inches="tight")

    plt.show()
