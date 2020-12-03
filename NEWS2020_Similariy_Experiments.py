import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim import corpora
import time
import seaborn as sns
from numpy import zeros
from math import gamma
from numpy import array
from CGibbs_gpm import GPM
import matplotlib.pyplot as plt
from preprocessing2 import save_obj, load_obj, corpus_stats, clean, remove_docs, jsd_mat, js_plots, js_plots_stacked
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
import multiprocessing

sns.set_style("darkgrid")

#select classes of interest
ref_class=2 #health
query_class=4 #science

#select total corpus size to use
total_docs=5000

data = pd.read_csv('labelled_newscatcher_dataset.csv')


#selected desired columns
data_df=data.filter(['topic', 'title'])
data_df.rename(columns = {'title': 'text', 'topic':'text_label'}, inplace = True)

labels=np.unique(np.array(data_df['text_label']))

def create_label_column(x):

    for num, txt in enumerate(labels):
        if x==txt:
            final_label=num
    
    return final_label


data_df['label'] = data_df['text_label'].apply(create_label_column)  

#clean data
data_df['tokenized']  = data_df['text'].apply(lambda x: clean(str(x)))

#cut documents that are too long and remove documents that are too short
min_length=1
max_length=100
all_data_df=remove_docs(data_df,min_length,max_length)

#ensure each label is equally represented
number_of_docs_per_class=round(total_docs/len(labels))
split_data_df = pd.DataFrame()
for cl in range(len(labels)):
    cl_indicator=(all_data_df['label'] == cl)
    cl_docs = all_data_df[cl_indicator]
    sample_cl_docs = cl_docs.sample(n=number_of_docs_per_class,random_state=1,axis=0)
    split_data_df = pd.concat((split_data_df,sample_cl_docs))
split_data_df.reset_index(drop=True, inplace=True)

#reference and query corpora
ref_indicator=(split_data_df['label'] == ref_class)
reference_corpus = split_data_df[ref_indicator]

query_indicator = (split_data_df['label'] == query_class)
query_corpus = split_data_df[query_indicator]

corpus_stats(reference_corpus,'Reference corpus')
corpus_stats(query_corpus,'Query corpus')

#split datasets: testing and training
reference_train, reference_test= train_test_split(reference_corpus, test_size=0.2, random_state=1)
query_train, query_test= train_test_split(query_corpus, test_size=0.2, random_state=1)

#concatenate: create train and test data
train = pd.concat((reference_train,query_train))
train.reset_index(drop=True, inplace=True)

test = pd.concat((reference_test,query_test))
test.reset_index(drop=True, inplace=True)

train_df=train

#topic modelling
dct = corpora.Dictionary(train_df['tokenized'])  # dictionary for indexing
save_obj(dct, 'GPMDICT-NEWS2020')

corpus = [dct.doc2bow(line) for line in train_df['tokenized']]  # bowCorpus
save_obj(train_df, 'GPM_training_NEWS2020')

print('Running GPM...')
#normalisation for GPM
def normalize_and_productoffactorials(N, corpus1):
    totals=[]
    for doc in corpus1:
        Nm=0
        for word in doc:
            Nm=Nm+word[1]
        totals.append(Nm)
    array(totals)
    
    #x_mv (new):
    corpus=[]
    m=0
    for doc2 in corpus1:
        document=[]    
        for word in doc2:
            a=list(word) #(word_id, count)
            a[1]=(float(word[1])/totals[m])*N
            document.append(tuple(a))
        m=m+1
        corpus.append(document)
        
    prodFactorialCounts=zeros(len(corpus))

    i=0
    for doc in corpus:
        prod=1
        
        for word in doc:
            prod*=gamma(word[1]+1)
        prodFactorialCounts[i]=prod
        i=i+1
       
    return prodFactorialCounts, corpus


t1 = time.time()

name='NEWS2020'
N=20
nTopics=10
alpha=0.001
beta=0.25
niters = 15
rep=0

id2word = dct  # dictionary for indexing
prodFactorialCounts, train_corpus = normalize_and_productoffactorials(N, corpus)
train_gpm = GPM(nTopics, 0.1, alpha, beta, id2word, train_corpus, name, prodFactorialCounts,rep, niters, N)
train_gpm.topicAssigmentInitialise()
train_topicWordCount, train_sumTopicWordCount, train_theta = train_gpm.inference() #output the topic-word counts

#print topics
finalAssignments=train_gpm.writeTopicAssignments()

# training corpus document by topic matrix
doc_topic_dist_corpus = train_theta

#split theta according to class
dtd_topic={}
for x in [ref_class,query_class]:
    dtd_topic[x]=train_theta[np.array(train_df['label'])==x,:]

#testing
df_test=test
save_obj(df_test, 'GPM_test_NEWS2020')
    
def doc_topic_dist(corpus):
    """
    Creates a document by topic distribution for a specified category using the trained DMM model
    """
    data = [id2word.doc2bow(doc) for doc in corpus]
    test_prodFactorialCounts, test_corpus = normalize_and_productoffactorials(N, data)
    test_gpm = GPM(nTopics, 0.1, alpha, beta, id2word, test_corpus, name, test_prodFactorialCounts, rep, 15, N)
    test_gpm.topicAssigmentTestset()
    test_theta = test_gpm.test_inference(train_topicWordCount, train_sumTopicWordCount)
    return test_theta

theta_test=doc_topic_dist(df_test['tokenized'])  
    
dtd_test_ref=doc_topic_dist(reference_test['tokenized'])
dtd_test_query=doc_topic_dist(query_test['tokenized'])

# Reference corpus
sim_ref=dtd_topic[ref_class]
sim_query=dtd_test_ref

js_check1 = jsd_mat(sim_ref, sim_query)
js_ch1_mean = np.mean(js_check1, axis=0)

ref = jsd_mat(sim_ref, sim_ref)
ref = ref[~np.eye(ref.shape[0], dtype=bool)].reshape(ref.shape[0], -1)  # remove ones => 250 by 250
means = np.mean(ref, axis=0)  # vector 250 means

prob_simGPM = []
for j in range(len(js_ch1_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch1_mean[j]]
    prob_simGPM.append(len(count1) / len(means)) # probability less than means

save_obj(prob_simGPM, 'sim_probs_applicationGPM_NEWS2020')

# Query corpus
disim_query=dtd_test_query

js_check2 = jsd_mat(sim_ref, disim_query)
js_ch2_mean = np.mean(js_check2, axis=0)


prob_disimGPM = []
for j in range(len(js_ch2_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch2_mean[j]]  
    prob_disimGPM.append(len(count1) / len(means)) # probability less than means

save_obj(prob_disimGPM, 'disim_probs_applicationGPM_NEWS2020')

js_plots([js_check1,js_check2],['%s (ref) vs %s (query)' % (labels[ref_class],labels[ref_class]),'%s (ref) vs %s (query)' % (labels[ref_class],labels[query_class])],color=['green','red'])


##############################################################################
#WORD2VEC
print('Running W2V...')
df = load_obj('GPM_training_NEWS2020')

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=3,  # Ignore words that appear less than this
                     size=200,  # Dimensionality of word embeddings
                     workers=cores,  # Number of processors (parallelisation)
                     window=5,  # Context window for words during training
                     iter=30,
                     sg=0)  # CBOW

# generate a vocabulary

w2v_model.build_vocab(df['tokenized'], progress_per=10000)


# Train the model
t = time.time()
w2v_model.train(df['tokenized'], total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


w2v_model.init_sims(replace=True)
w2v_model.save("w2v-application")


len(w2v_model.wv.vocab)
termsim_index = WordEmbeddingSimilarityIndex(w2v_model.wv)  # get termsim index
dictionary = load_obj('GPMDICT-NEWS2020')

bow_corpus = [dictionary.doc2bow(document) for document in df['tokenized']]  # generate a bow corpus
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

df_test_dominant_topic=load_obj('GPM_test_NEWS2020')

test1=df_test_dominant_topic['label']==ref_class #NB
test0=df_test_dominant_topic['label']==query_class #NB
df_test1=df_test_dominant_topic[test1]
df_test1.reset_index(drop=True, inplace=True)
df_test0=df_test_dominant_topic[test0]
df_test0.reset_index(drop=True, inplace=True)


def get_bow(corpus):
    """ some basic text preprocessing on a specified category of the 20 newsgroups dataset. Refer to sklean documentation for 20 newsgroups for potential categories.


    Args:
        corpus: (List) Tokenized data
    Return:
        gensim BOW in a list format
    """

    bow = [dictionary.doc2bow(doc) for doc in corpus]  # transform into a gen bow
    bow = [i for i in bow if len(i) > 0]  # remove empty lists
    return bow


test1BOW=get_bow(df_test1['tokenized'])
train1=df['label']==ref_class #Note
dftrain1=df[train1]
dftrain1.reset_index(drop=True, inplace=True)

train1BOW=get_bow(dftrain1['tokenized'])


testvstraining_1=SoftCosineSimilarity(train1BOW,similarity_matrix)
scs_topic1=testvstraining_1[test1BOW]


test0BOW=get_bow(df_test0['tokenized'])
scs_topic1vstopic0=testvstraining_1[test0BOW]

fig = plt.figure(1, figsize=(10, 5))
sns.distplot(scs_topic1, bins=50, color='red', label='%s (Testing vs Training)' % labels[ref_class])  # 
sns.distplot(scs_topic1vstopic0, bins=50, color='green', label='%s (Testing) vs %s (Training)' % (labels[ref_class],labels[ref_class]))

plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
fig.legend()
plt.title('Semantic Similarity and Disimilarity Testing')
plt.tight_layout()
plt.savefig("w2v-application.jpg")
fig.legend()
plt.show()

def get_ref_query(corpus):
    """
    Carry out some basic text preprocessing on a specified category of the 20 newsgroups dataset
    params: cat - str: category from 20 newsgroups. Refer to sklean documentation for 20 newsgroups.
    """

    ref_df = corpus
    bow_r = [dictionary.doc2bow(doc) for doc in ref_df]
    bow_r = [i for i in bow_r if len(i) > 0]
    return bow_r

 
top1_ref=train1BOW
top1_query= get_ref_query(df_test1['tokenized'])   

top0_query=get_ref_query(df_test0['tokenized'])

docsim_index = SoftCosineSimilarity(top1_ref, similarity_matrix) 

scm_self = docsim_index[top1_ref]
ref = scm_self[~np.eye(scm_self.shape[0], dtype=bool)].reshape(scm_self.shape[0], -1)
means = np.mean(ref, axis=0)

SCM_sim = docsim_index[top1_query]
SCM_sim = np.mean(SCM_sim, axis=1)

prob_simW2V = []
for j in range(len(SCM_sim)):
    count2 = [i for i in means if i <= SCM_sim[j]]
    prob_simW2V.append(len(count2) / len(means))
    
save_obj(prob_simW2V, 'probsim_w2v-application_NEWS2020')

scm_diff = docsim_index[top0_query]
scm_diff = np.mean(scm_diff, axis=1)

prob_disimW2V = []
for j in range(len(scm_diff)):
    count1 = [i for i in means if i <= scm_diff[j]]
    prob_disimW2V.append(len(count1) / len(means))
    

save_obj(prob_disimW2V, 'probdisim_w2v-application_NEWS2020')

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
js_plots_stacked([js_check1, js_check2],[scs_topic1, scs_topic1vstopic0], ['Health (ref) vs Health (query)', 'Health (ref) vs Science (query)'], ['Health (ref) vs Health (query)', 'Health (ref) vs Science (query)'])

##############################################################################
#CLASSIFICATION METRICS
print('Calculating classification metrics...')
sns.set_style('darkgrid')


sim_probG = load_obj('sim_probs_applicationGPM_NEWS2020')
disim_probG = load_obj('disim_probs_applicationGPM_NEWS2020')

disim_probG = pd.DataFrame(disim_probG)
sim_probG = pd.DataFrame(sim_probG)


unrel_labels = ['r'] * len(disim_probG)
unrel_labelsG = [0] * len(disim_probG)

rel_labels = ['b'] * len(sim_probG)
rel_labelsG = [1] * len(sim_probG)

labels2 = np.concatenate((unrel_labels, rel_labels))
labels01G=np.concatenate((unrel_labelsG, rel_labelsG))

jsd = np.concatenate((disim_probG, sim_probG))

# create dataframe of rel_index and label
df_original = pd.DataFrame(data=[labels2, jsd, labels01G]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]
df_original['labels01'] = df_original[2]


df = df_original.sort_values('rel_index', ascending=False)
dfG = df_original.sort_values('rel_index', ascending=False) 

threshold = 0.1
thresholdG = threshold

true_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] >= threshold)]
false_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] < threshold)]
true_irrelevantGPM = df.loc[(df["labels"] == 'r') & (df["rel_index"] < threshold)]
false_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] >= threshold)]

confusion_matrix = np.mat([[len(true_relevant), len(true_irrelevantGPM), sum((len(true_relevant), len(true_irrelevantGPM)))],
                           [len(false_relevant), len(false_irrelevant),
                            sum((len(false_irrelevant), len(false_relevant)))],
                           [sum((len(true_relevant), len(false_relevant))),
                            sum((len(false_irrelevant), len(true_irrelevantGPM))), len(df)]])
conf_df = pd.DataFrame(confusion_matrix
                       , columns=["Relevant", "Irrelevant", "Total"]
                       , index=["True", "False", "Total"])

perc_ignoreGPM = float(len(true_irrelevantGPM)) / (len(df)) * 100  # percentage of documents to ignore
precision = float(len(true_irrelevantGPM)) / float(len(false_irrelevant) + len(true_irrelevantGPM))
accuracy = (len(true_relevant) + len(true_irrelevantGPM)) / len(df)
recall = float(len(true_irrelevantGPM)) / float(len(false_relevant) + len(true_irrelevantGPM))
F1_Score = 2*(recall * precision) / (recall + precision)

fig = plt.figure(figsize=(10, 7.5), tight_layout=True)

GPM_df = df
GPM_df.reset_index(drop=True, inplace=True)

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.ylim([0, 1])
plt.xlim([0, len(labels2)])
plt.scatter(range(len(df)), df['rel_index'], c=df['labels'], s=30, alpha=0.7)

plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance Index', fontsize='x-large')
plt.xlabel('Document Number', fontsize='x-large')
plt.title('GPM Similarity Test: Relevance Index ', fontsize='x-large')
plt.savefig('GPM_RelIndexMeans.jpg', bbox_inches='tight')
plt.show()

print('* GPM Evaluation')
print("Threshold:", threshold)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy: " + str(accuracy))
print("F1 Score:" + str(F1_Score)+"\n")


sim_probW = load_obj('probsim_w2v-application_NEWS2020')
disim_probW = load_obj('probdisim_w2v-application_NEWS2020')

disim_probW = pd.DataFrame(disim_probW)
sim_probW = pd.DataFrame(sim_probW)

unrel_labels = ['r'] * len(disim_probW)
unrel_labels1 = [0] * len(disim_probW)

rel_labels = ['b'] * len(sim_probW)
rel_labels1 = [1] * len(sim_probW)

labels3 = np.concatenate((unrel_labels, rel_labels))
labels01W=np.concatenate((unrel_labels1, rel_labels1))

jsd = np.concatenate((disim_probW, sim_probW))
df_original = pd.DataFrame(data=[labels3, jsd, labels01W]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]
df_original['labels01'] = df_original[2]

df = df_original.sort_values('rel_index', ascending=False)
dfW = df_original.sort_values('rel_index', ascending=False) 

threshold = 0.5
true_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] >= threshold)]
false_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] < threshold)]
true_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] < threshold)]
false_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] >= threshold)]

confusion_matrix = np.mat([[len(true_relevant), len(true_irrelevant), sum((len(true_relevant), len(true_irrelevant)))],
                           [len(false_relevant), len(false_irrelevant),
                            sum((len(false_irrelevant), len(false_relevant)))],
                           [sum((len(true_relevant), len(false_relevant))),
                            sum((len(false_irrelevant), len(true_irrelevant))), len(df)]])
conf_df = pd.DataFrame(confusion_matrix
                       , columns=["Relevant", "Irrelevant", "Total"]
                       , index=["True", "False", "Total"])

perc_ignore = float(len(true_irrelevant)) / (len(df)) * 100  # percentage of documents to ignore
precision = float(len(true_irrelevant)) / float(len(false_irrelevant) + len(true_irrelevant))
accuracy = (len(true_relevant) + len(true_irrelevant)) / len(df)
recall = float(len(true_irrelevant)) / float(len(false_relevant) + len(true_irrelevant))

w2v_df = df
w2v_df.reset_index(drop=True, inplace=True)

fig = plt.figure(figsize=(10, 7.5), tight_layout=True)

plt.ylim([0, 1])
plt.xlim([0, len(labels3)])
plt.scatter(range(len(df)), df['rel_index'], c=df['labels'], s=30, alpha=0.7)
plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance Index', fontsize='x-large')
plt.xlabel('Document Number', fontsize='x-large')
plt.title('Word2vec Similarity Test: Relevance Index', fontsize='x-large')
plt.savefig('w2v_RelIndexMeans.jpg', bbox_inches='tight')
plt.show()
F1_Score = 2*(recall * precision) / (recall + precision)

print('* W2V Evaluation')
print("Threshold:", threshold)
print("Accuracy: " + str(accuracy))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", F1_Score)

# plotting subplots
fig, axs = plt.subplots(2,1, figsize=(12.5, 10))
[axs[i].set_ylim([0, 1]) for i in range(2)]
[axs[k].set_xlim([0, len(labels3)]) for k in range(2)]
axs[0].scatter(range(len(GPM_df)), GPM_df['rel_index'], c=GPM_df['labels'], s=30, alpha=0.7)
axs[1].scatter(range(len(w2v_df)), w2v_df['rel_index'], c=w2v_df['labels'], s=30, alpha=0.7)


axs[0].axhline(thresholdG, c='green', linewidth=1.5)
axs[1].axhline(threshold, c='green', linewidth=1.5)

[axs[i].set_ylabel('Relevance Index (Probabilities)', fontsize='x-large') for i in range(2)]
[axs[j].set_xlabel('Document Count', fontsize='x-large') for j in range(2)]

axs[0].set_title('GPM Relevance Index: Relevant documents (Blue) and Irrelevant documents (Red)', fontsize='x-large')
axs[1].set_title('Word2vec Relevance Index: Relevant documents (Blue) and Irrelevant documents (Red)', fontsize='x-large')


fig.subplots_adjust(top=0.88, hspace = 0.3)
plt.savefig('PHD_relindex_news_2020.jpg', bbox_inches='tight')
plt.show()
