# -*- coding: utf-8 -*-


# Poisson + collapsed Gibbs sampling

#from math import gamma
import numpy as np
import random
from scipy import special
from scipy import log
from scipy.stats import entropy

class GPM:
    def __init__(self, ntopics, gam, beta, alpha, id2word, corpus, name, prodFactorialCounts,rep,niters,N):
        
        print("corpus=%d, words=%d, a= 0.001, b=%f, ntopics=%i, rep=%i, " % (len(corpus), len(id2word), beta, ntopics, rep))
        self.ntopics = ntopics         # number of topics
        self.alpha = alpha # shape parameter of words prior
        self.beta = beta   # scale parameter of words prior
        self.gam = gam # new: parameter of topics prior (alpha -> gam)
        self.corpus = corpus
        self.id2word = id2word
        self.niters = niters
        self.prodFactorialCounts=prodFactorialCounts
        self.numDocuments = len(corpus)
        self.numWordsInCorpus = 0        
        self.occurenceToIndexCount = []
        self.topicAssignments = []
        self.docTopicCount = [] #number of documents in topic (m_z)
        self.topicWordCount = [] #number of occurrences of word w in topic z (n_z_w)
        self.sumTopicWordCount = [] #number of words in topic z (n_z)
        self.alphaSum=len(self.id2word)*self.alpha
        self.conditional_prob = []
        self.N=N
        self.psi = []
        self.theta = np.zeros((self.numDocuments, self.ntopics))
        
        self.output = ''
        self.name = name
        self.twords = 10
        self.rep=rep
        
        
    def topicAssigmentInitialise(self):
        self.docTopicCount = [0 for x in range(self.ntopics)] #initialise
        self.sumTopicWordCount = [0 for x in range(self.ntopics)] #initialise

        for i in range(self.ntopics):
            self.topicWordCount.append([0 for x in range(len(self.id2word))]) #initialise

        for d in range (self.numDocuments):
            topic = random.randint(0,self.ntopics-1) #for each document, sample a topic
            self.docTopicCount[topic]+=1 #update number of documents in topic (m_z)
            N_d = np.sum([word[1] for word in self.corpus[d]]) #number of words in document
            self.sumTopicWordCount[topic]+= N_d #update number of words in topic
 
            for j in range (len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]]+=word[1] #update number of occurences of word w in document

            self.topicAssignments.append(topic) #record the current topic of this document
        #print(self.docTopicCount)
        #print(self.sumTopicWordCount)
        #print(self.topicWordCount)
        #print('\n')
        
    def nextDiscrete(self,a):
        b = 0.

        for i in range(len(a)):
            b+=a[i]
        
        r = random.uniform(0.,1.)*b
		
        b=0.
        #print(r)
        for i in range (len(a)):
            b+=a[i]
            if(b>r):
                return i
        return len(a)-1
    
    def sampleInSingleIteration(self,x):
        print ("iteration: "+str(x))
        print(self.sumTopicWordCount)
        #print(self.topicAssignments)
        #print('\n')
        for d in range(self.numDocuments):
            #print ("document: "+str(d))
            topic = self.topicAssignments[d] #record the current cluster of d
            #print ("topic assignment: "+str(topic))
            #print ("topic assignment before: "+str(self.docTopicCount))
            self.docTopicCount[topic]-=1 #remove this document from assigned topic
            #print ("topic assignment after: "+str(self.docTopicCount))
            N_d = np.sum([word[1] for word in self.corpus[d]]) #number of words in document
            #print("number of words in doc " + str(N_d))
            #print ("topic words before: "+str(self.sumTopicWordCount))
            self.sumTopicWordCount[topic]-= N_d #remove number of words from this topic
            #print ("topic words  after: "+str(self.sumTopicWordCount))

            #print("nzw before" + str(self.topicWordCount))
            for j in range(len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]]-=word[1] #remove number of occurences of word w in document
            #print("nzw after " + str(self.topicWordCount))
            #sample a topic for d:
            for t in range(self.ntopics):
                #102 is the numerator of the first term of equation 4
                #self.conditional_prob[t] = log(((self.docTopicCount[t]+self.gam) * (self.beta**N_d) * (self.docTopicCount[t]*self.beta+1)**(self.sumTopicWordCount[t]+self.alphaSum)) /  \
                #(self.prodFactorialCounts[d] * (self.docTopicCount[t]*self.beta+self.beta+1)**(self.sumTopicWordCount[t]+N_d+self.alphaSum)))
                
                self.conditional_prob[t]= special.logsumexp(log([self.docTopicCount[t]+self.gam, self.beta**N_d]))+(self.sumTopicWordCount[t]+self.alphaSum)*log(self.docTopicCount[t]*self.beta+1) \
                - (log(self.prodFactorialCounts[d]) + (self.sumTopicWordCount[t]+N_d+self.alphaSum)*log(self.docTopicCount[t]*self.beta+self.beta+1))
                #denominator terms: - log(self.prodFactorialCounts[d]) + (self.sumTopicWordCount[t]+N_d+self.alphaSum)*log(self.docTopicCount[t]*self.beta+self.beta+1)
                #numerator terms: special.logsumexp(log([self.docTopicCount[t]+self.gam, self.beta**N_d]))+(self.sumTopicWordCount[t]+self.alphaSum)*log(self.docTopicCount[t]*self.beta+1)
                #print(self.conditional_prob[t])
                
                #print("document: " + str(self.corpus[d]))
                ### old code ### (1/2)
                #i = 0 #i is a counter to get the total number of words in the document (length of the document)
                ### old code ###
                for w in range(len(self.corpus[d])):
                    
                    #print('w ' + str(w))
                    word = self.corpus[d][w] 
                    #print("word: " + str(word))                    
                    #for j in range(word[1]): 
                    
                    
                    #print("doc: "+ str(d))
                    #print("n_zv: "+ str(self.topicWordCount[t][word[0]]))
                    #print("word id: " + str(word[0]))
                    #print("freq: " + str(word[1]))
                    #print("log gamma: ")
                    self.conditional_prob[t] += special.loggamma(self.topicWordCount[t][word[0]]+word[1]+self.alpha).real-special.loggamma(self.topicWordCount[t][word[0]]+self.alpha).real
                        
                        ### old code ### (2/2)
                        #i = i + 1
                        #self.conditional_prob[t] += log(self.topicWordCount[t][word[0]]+self.alpha + (j+1) - 1) 
                        ### old code ###
				
            #print(np.exp(self.conditional_prob))
            topic = self.nextDiscrete(np.exp(self.conditional_prob))
            self.theta[d,:] =  np.exp(self.conditional_prob) / np.sum(np.exp(self.conditional_prob))

            self.docTopicCount[topic]+=1
            self.sumTopicWordCount[topic] += N_d #remove number of words from this topic

            for j in range(len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]] += word[1] #remove number of occurences of word w in document

            self.topicAssignments[d] = topic

    def inference(self):
        out=[]
        self.conditional_prob = [0 for x in range(self.ntopics)]
        for x in range(self.niters):
            numtopics=self.sampleInSingleIteration(x)
            out.append(numtopics)
        return self.topicWordCount,self.sumTopicWordCount, self.theta  
        
    def worddist(self):
        """get topic-word distribution"""
        
        """get topic-word distribution"""

        psi_file = open("output/psi/%s_GPM_psi_Kstart_%i_rep_%i.psi" % (self.name,self.ntopics,self.rep),"w")
        self.psi = np.zeros((len(self.id2word), self.ntopics))
        for t in range(self.ntopics):
            for w in range(len(self.id2word)):
                self.psi[w,t] = (self.topicWordCount[t][w] + self.alpha)/(self.docTopicCount[t] + 1/self.beta)
                #self.psi[w,t] = (self.topicWordCount[t][w] + self.alpha)/(self.docTopicCount[t] + self.alpha) #2
                psi_file.write(str(self.psi[w,t]) + " ")
                #phi_file_csv.write(str(self.psi[w,t]) + ",")
            psi_file.write("\n")
            #phi_file_csv.write("\n")
        psi_file.close()
        #phi_file_csv.close()
        
        #save doc x topic (theta)
        theta_file = open("output/thetas/%s_GSGPM_thetas_N_%i_rep_%i.theta" % (self.name,self.N,self.rep),"w")
        for m in range(self.numDocuments):
            for k in range(self.ntopics):
                theta_file.write(str(self.theta[m,k]) + " ")    
            theta_file.write("\n")
        theta_file.close()
        
        return self.psi, self.theta

    def writeTopicAssignments(self):
        """
        file1 = open("output/assignments/%s_GPM_topicAssignments_rep_%i.txt" % (self.name,self.rep),"w")
        file2 = open("output/assignments/%s_GPM_selectedTopics_rep_%i.txt" % (self.name,self.rep),"w")
        #for i in range(self.numDocuments):
        #[file.write(str(self.topicAssignments[i])+"\n") for i in range(self.numDocuments)]
        for doc_assignment in self.topicAssignments:
            file1.write(str(doc_assignment)+"\n")
        #print(self.topicAssignments)
        
        for selected_topic in np.unique(self.topicAssignments):
            file2.write(str(selected_topic)+"\n")
            
        #print(np.unique(self.topicAssignments))
        
        file1.close()
        file2.close()
        """
        return np.unique(self.topicAssignments)

    def writeTopTopicalWords(self, selected_topics):
        file = open("%s_GSGPMy_rep_%i.topWords" % (self.name,self.rep),"w") 
        coherence_index_all=[]
        for t in selected_topics:
            wordCount = {w:self.topicWordCount[t][w] for w in range(len(self.id2word))}
			
            count =0
            string="Topic "+str(t)+": "
            coherence_index_per_topic=[]
			
            for index in sorted(wordCount, key=wordCount.get, reverse=True):
                coherence_index_per_topic.append(index)
                string += self.id2word[index]+" "
                count+=1
                #print(count)
                if count>=self.twords:
                    file.write(string+"\n") 
                    print(string)
                    break
            coherence_index_all.append(coherence_index_per_topic)
        file.close()
        return(coherence_index_all)

### New code: 
        
    def topicAssigmentTestset(self):
        self.docTopicCount = [0 for x in range(self.ntopics)] #initialise


        for d in range (self.numDocuments):
            topic = random.randint(0,self.ntopics-1) #for each document, sample a topic
            self.docTopicCount[topic]+=1 #update number of documents in topic (m_z)
 
            self.topicAssignments.append(topic) #record the current topic of this document
        #print(self.docTopicCount)
        #print(self.sumTopicWordCount)
        #print(self.topicWordCount)
        #print('\n')        

    def test_inference(self,topicWordCount, sumTopicWordCount):
        self.conditional_prob = [0 for x in range(self.ntopics)]
        test_iter = 15
        [self.expectedTopicDistr(x,topicWordCount, sumTopicWordCount) for x in range(test_iter)]
        return self.theta 
      
    def expectedTopicDistr(self, x, topicWordCount, sumTopicWordCount):
        print ("iteration: "+str(x))

        for d in range(self.numDocuments):

            topic = self.topicAssignments[d] #record the current cluster of d
            self.docTopicCount[topic]-=1
            
            N_d = np.sum([word[1] for word in self.corpus[d]]) #number of words in document

            for t in range(self.ntopics):

                self.conditional_prob[t]= special.logsumexp(log([self.docTopicCount[t]+self.gam, self.beta**N_d]))+(sumTopicWordCount[t]+self.alphaSum)*log(self.docTopicCount[t]*self.beta+1) \
                - (log(self.prodFactorialCounts[d]) + (sumTopicWordCount[t]+N_d+self.alphaSum)*log(self.docTopicCount[t]*self.beta+self.beta+1))

                for w in range(len(self.corpus[d])):
                    word = self.corpus[d][w] 
                    self.conditional_prob[t] += special.loggamma(topicWordCount[t][word[0]]+word[1]+self.alpha).real-special.loggamma(topicWordCount[t][word[0]]+self.alpha).real
                        
            topic = self.nextDiscrete(np.exp(self.conditional_prob))
            self.theta[d,:] =  np.exp(self.conditional_prob) / np.sum(np.exp(self.conditional_prob))

            self.docTopicCount[topic]+=1

            self.topicAssignments[d] = topic

    def jensen_shannon(self, sent1, sent2):
        """
        This function implements a Jensen-Shannon similarity
        between two documents (an LDA topic distribution for a document)
        It returns an array of length M where M is the number of documents in the corpus
        """
        # lets keep with the p,q notation above
        p = sent1[None,:].T # take transpose
        q = sent2[None,:].T # transpose matrix
        m = 0.5*(p + q)
        if entropy(p,m) > 0. and entropy(q,m) > 0:
            return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))
        else:
            return 0. 