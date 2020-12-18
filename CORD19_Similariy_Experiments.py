import pandas as pd
import numpy as np
from gensim import corpora
from numpy import zeros
from math import gamma
from numpy import array
from CGibbs_gpm import GPM
import matplotlib.pyplot as plt
import pickle
from scipy.stats import entropy
import seaborn as sns

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
    axs.set_ylabel('Frequency', fontsize='x-large')
    axs.set_xlabel('Jensen-Shannon Distances', fontsize='x-large')

    axs.set_title('Jensen-Shannon distances of ' + labels[0] + ' and ' + labels[1],
                  fontsize='x-large')
    plt.tight_layout()
    # fig.subplots_adjust(top=0.88)
    fig.legend(labels=labels, fontsize='x-large', loc='upper right',bbox_to_anchor=(0.95, 0.95))
    plt.savefig('GPM_kde' + labels[0] + labels[1] + '.jpg', bbox_inches="tight")

    plt.show()    

#select classes of interest
ref_class=0
query_class=4

#load data
test = load_obj('GPM_test_abstracts')
df_test=test
data = load_obj('GPM_training_abstract')
labels=np.unique(np.array(data['Dominant_Topic']))
train_df=data

#topic modelling
dct = corpora.Dictionary(train_df['tokenized'])  # dictionary for indexing
corpus = [dct.doc2bow(line) for line in train_df['tokenized']]  # bowCorpus
#%%

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
        
    #product of x_mv (new):
    prodFactorialCounts=zeros(len(corpus))
    #N_m=zeros(len(corpus))
    i=0
    for doc in corpus:
        prod=1
        
        for word in doc:
            prod*=gamma(word[1]+1)
        prodFactorialCounts[i]=prod
        i=i+1
       
    return prodFactorialCounts, corpus

name='CORD-19'
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
train_topicWordCount, train_sumTopicWordCount, train_theta = train_gpm.inference()

# training corpus document by topic matrix
doc_topic_dist_corpus = train_theta

#split theta according to class
dtd_topic={}
for x in [ref_class,query_class]:
    dtd_topic[x]=train_theta[np.array(train_df['Dominant_Topic'])==x,:]
  
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

#split test set according to label
dtd_test={}
for x in [0,4]:
    dtd_test[x]=theta_test[np.array(df_test['Dominant_Topic'])==x,:]

#%%
# Relevance index
sim_ref=dtd_topic[ref_class]
sim_query=dtd_test[ref_class]
disim_query=dtd_test[query_class]

js_check1 = jsd_mat(sim_ref, sim_query)
js_ch1_mean = np.mean(js_check1, axis=0)

ref = jsd_mat(sim_ref, sim_ref)
ref = ref[~np.eye(ref.shape[0], dtype=bool)].reshape(ref.shape[0], -1)  # remove ones => 250 by 250
means = np.mean(ref, axis=0)  # vector 250 means

prob_simGPM = []
for j in range(len(js_ch1_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch1_mean[j]]  # at most inside means
    prob_simGPM.append(len(count1) / len(means))


js_check2 = jsd_mat(sim_ref, disim_query)
js_ch2_mean = np.mean(js_check2, axis=0)

prob_disimGPM = []
for j in range(len(js_ch2_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch2_mean[j]]  # at most inside means
    prob_disimGPM.append(len(count1) / len(means))

#JS plot
js_plots([js_check1,js_check2],['Virology (ref) vs Virology (query)','Virology (ref) vs Pulmonology (query)'],color=['green','red'])

sim_probG = prob_simGPM
disim_probG = prob_disimGPM

disim_probG = pd.DataFrame(disim_probG)


sim_probG = pd.DataFrame(sim_probG)

unrel_labels = ['r'] * len(disim_probG)
unrel_labelsG = [0] * len(disim_probG)

rel_labels = ['b'] * len(sim_probG)
rel_labelsG = [1] * len(sim_probG)

labels2 = np.concatenate((unrel_labels, rel_labels))
labels01G=np.concatenate((unrel_labelsG, rel_labelsG))

jsd = np.concatenate((disim_probG, sim_probG))
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
plt.title('GPM Relevance Index: Virology Ref-Query (Blue) vs Virology-Pulmonology (Red)', fontsize='x-large')
plt.savefig('RelIndexMeans.jpg', bbox_inches='tight')
plt.show()
F1_Score = 2*(recall * precision) / (recall + precision)

print('* GPM Evaluation *\n')
print("Threshold:", threshold)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy: " + str(accuracy))
print("F1 Score:", F1_Score)

