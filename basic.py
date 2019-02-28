from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 

warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec 

# Reads ‘alice.txt’ file 
sample = open('20k/dev.txt', 'r') 
s = sample.read() 


# Replaces escape character with space 
f = s.replace("\n", " ") 

data = [] 

# iterate through each sentence in the file 
for i in sent_tokenize(f): 
	temp = [] 
	
	# tokenize the sentence into words 
	for j in word_tokenize(i): 
		if '###' in j or  j in ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']:			
			continue
		temp.append(j.lower()) 

	data.append(temp) 





# # Create CBOW model 
# model1 = Word2Vec(data, min_count = 1, size = 100, window = 5) 

# # Print results 
# print("Cosine similarity between 'patients' and 'therapy' - CBOW : ", model1.similarity('patients', 'therapy')) 
	
# print("Cosine similarity between 'patients' and 'patients' - CBOW : ",  model1.similarity('patients', 'study')) 


# # Create Skip Gram model 
# model2 = Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1) 

# # Print results 
# print("Cosine similarity between 'patients' and 'therapy' - CBOW : ", model2.similarity('patients', 'therapy')) 
	
# print("Cosine similarity between 'patients' and 'patients' - CBOW : ",  model2.similarity('patients', 'study'))