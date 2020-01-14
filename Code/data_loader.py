from nltk.tokenize import TweetTokenizer
import collections, os, re, traceback, unicodedata, string, pdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

class training_dataset():

	def __init__(self, 	dataset_name, 
		conv_to_3_class = True, 
		abs_len 		= 35,#34,30, 
		maxlen  		= 180,#178,130, 
		maxlen_word 	= 70,#69,25 
		): 
		input_folder 			=  self.get_input_folder(dataset_name)
		# text file containing the data
		self.data_path_train	= input_folder+'train_clean.txt' 
		self.data_path_val 		= input_folder+'dev_clean.txt'
		self.data_path_test 	= input_folder+'test_clean.txt'
		self.conv_to_3_class 	= conv_to_3_class	# if True, data class reduced  as per  label_transform method
		# The following are to be initilized/ updated from traning data
		self.abs_len			= abs_len			# maximum lines in a abstract
		self.maxlen 			= maxlen			# maximum words in a line
		self.maxlen_word 		= maxlen_word		# maximum characters in a word

		# self.wordlengths		= []

		self.nclasses 			= None 				# number of output classes
		self.idx2word 			= None  			# index to word mapping
		self.word2idx 			= None  			# word to index mapping
		self.char2idx 			= None  			# character to index mapping
		self.label2idx 			= None  			# data label to index mapping
		self.vocsize 			= None  			# vocubulary size as per training data
		self.charsize 			= None  			# alphabet size as per training data

	def get_input_folder(self, dataset):
		switcher = { 
			u'pubmed_non_rct'	: u'../PubMedData/output/', 
			u'pubmed_rct'		: u'../PubmedRCT/20k_num_as_sign/',
			u'arxiv'			: u'../arxiv_final/', 
			u'IEEE_TLT'			: u'../IEEE_final/TLT/',
			u'IEEE_TPAMI'		: u'../IEEE_final/TPAMI/',
			u'merged'			: u'../Merged/',
		}
		return switcher.get(dataset, u'')  

	def label_transform(self, label):
		switcher = { 
			u'0'			: u'BACKGROUND+OBJECTIVE', 
			u'1'			: u'DESCRIPTION',
			u'2'			: u'OBSERVATION+CONCLUSIONS', 
			u'BACKGROUND'	: u'BACKGROUND+OBJECTIVE', 
			u'OBJECTIVE'	: u'BACKGROUND+OBJECTIVE', 
			u'METHODS'		: u'DESCRIPTION',
			u'RESULTS'		: u'OBSERVATION+CONCLUSIONS',  
			u'CONCLUSIONS'	: u'OBSERVATION+CONCLUSIONS',
		}
		return switcher.get(label, u'UNDEFINED')  

	def unicodeToAscii(self, s):
		# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
		return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn'
			and c in string.ascii_letters
		)

	def __define_data_dicts(self, structured_training_data, labels):
		words   	= set()
		charcounts 	= collections.Counter()
		for abstract in structured_training_data:
			sent_len = 0
			for sentence in abstract:
				word_len 		= 0
				sent_len 		+= 1
				# words_in_sent	= sentence.translate({ord(ch): None for ch in '.,;:()1234567890'}).split()
				for word in sentence:
					words.add(word)				
					word_len +=1
					char_len = 0
					for char in word:
						charcounts[self.unicodeToAscii(char)] += 1
						char_len +=1

					# import pdb; pdb.set_trace()

					if self.maxlen_word < char_len:
						# print('maxlen_word '+str(char_len)+ ': '+ word)
						self.maxlen_word = char_len
					# self.wordlengths.append(word_len)
				if self.maxlen < word_len:					
					# print('maxlen '+str(word_len)+ ': '+ sentence+'\n')
					self.maxlen = word_len
			if self.abs_len < sent_len:
				self.abs_len = sent_len
		
		words   = list(words)
		 #for converting words to indices
		self.word2idx    = {w: i+1 for i, w in enumerate(words)}
		chars 		= [charcount[0] for charcount in charcounts.most_common()]
		self.char2idx 	= {c: i+1 for i, c in enumerate(chars)}

		# for converting indices back to words
		self.idx2word    = dict((k,v) for v,k in self.word2idx.items())
		
		self.idx2word[0]     = 'PAD'
		self.idx2word[-1]    = 'unknown'

		self.vocsize =  len(words)+1		
		idx2char  = dict((k,v) for v,k in self.char2idx.items())
		idx2char[0] = 'PAD'
		self.charsize =  max(idx2char.keys()) + 1

		self.label2idx   = {l: i+1 for i, l in enumerate(labels)}
		self.nclasses = len(labels)+1

		return #idx2word,  word2idx, char2idx, self.abs_len, self.maxlen, self.maxlen_word, vocsize, charsize

	def get_data(self, data_path):
		# defauls : abs_len =30, maxlen=130, maxlen_word = 25
		#  data_path = 'data/pubmed_non_rct.txt'
		input_data = list(open(data_path, 'r'))
		# print(data_path)

		tt = TweetTokenizer()
		labels = [u'BACKGROUND', u'OBJECTIVE', u'METHODS', u'RESULTS',  u'CONCLUSIONS', u'0', u'1',u'2']
		#  evantual shape labels = [u'BACKGROUND+OBJECTIVE', u'METHODS', u'RESULTS+CONCLUSIONS']
		metadata 			= []     
		structured_lines 	= [] 
		structured_data 	= []
		structured_labels 	= []
		metadata_buffer 	= ''
		buffered_lines	 	= []       
		buffered_abstract 	= []  
		buffered_labels 	= []  

		for line in input_data:
				line = line.strip()  
				if line.startswith('#'):#, 0,len(line)):				
					if buffered_abstract:
						metadata.append(metadata_buffer+'\n')
						structured_data.append(buffered_abstract)	
						structured_labels.append(buffered_labels)
						structured_lines.append(buffered_lines)
						metadata_buffer 	= ''
						buffered_abstract 	= []  
						buffered_labels 	= []
						buffered_lines	 	= []  

					metadata_buffer += line+'\n'					

				else:					
					if line:                          
						data = tt.tokenize(line)
						label = data[0]
						data =  data[1:] 
						data = [word.lower() for word in data if word.isalnum()]

						if label in labels:

							if self.conv_to_3_class:
								label = self.label_transform(label)	

							if len(data)<=3:
								try:
									if label == buffered_labels[-1]:
										buffered_abstract[-1]+=data # append to previous line
										#label is same
									else : 
										# skip this line
										pass 
								except Exception as e:
									# skip this line
									pass 
							else:
								# pdb.set_trace()
								buffered_lines.append(line.split('\t')[-1])   # saving the text from the line of the abstract
								buffered_abstract.append(data)
								buffered_labels.append(label)
		if self.conv_to_3_class: 							   
			labels = list(set([self.label_transform(label) for label in labels]))

											   
		if self.nclasses is None:		
			self.__define_data_dicts(structured_data, labels) 	
		#data_as_tensor = (X, X_words, Y)
		data_as_tensor  = self.__get_final_data(structured_data, structured_labels)		
		# pdb.set_trace() 			
		return data_as_tensor, structured_lines, metadata

	def __get_final_data(self, structured_data, structured_labels):
		#Defaults : abs_len =30, maxlen=130, maxlen_word = 25
		# (idx2word,  word2idx, char2idx, label2idx, abs_len, maxlen, maxlen_word, vocsize, charsize, nclasses) = data_dicts
		X 		= []
		X_words = []
		Y 		= []
		for abstract in structured_data:
			indexed_abstract 		= []
			indexed_abstract_word 	= []
			for sentence in abstract:
				indexed_sentence 		= []
				indexed_word_sentence 	= []
				# word_in_sent = sentence.translate({ord(ch): None for ch in '.,;:1234567890'}).split()
				for word in sentence:
					try:
						index = self.word2idx[word]
					except:
						index = -1
					indexed_sentence.append(index)
					indexed_word = []
					for char in word:
						try:
							char_index = self.char2idx[char]
						except:
							char_index = -1
						indexed_word.append(char_index)
					indexed_word_sentence.append(indexed_word)
				# padding or pruning indexed_word_sentence as per nessacity
				indexed_word_sentence = pad_sequences(indexed_word_sentence, maxlen=self.maxlen_word)
				indexed_abstract.append(indexed_sentence)
				indexed_abstract_word.append(indexed_word_sentence)

			indexed_abstract  		= pad_sequences(indexed_abstract , maxlen=self.maxlen)
			indexed_abstract_word  	= pad_sequences(indexed_abstract_word , maxlen=self.maxlen)
			X.append(indexed_abstract)
			X_words.append(indexed_abstract_word)

		for labels in structured_labels:
			indexed_labels = []
			for label in labels:
				indexed_labels.append(self.label2idx[label]) 


			Y.append(indexed_labels)

		# print(self.label2idx)
		X 		=  pad_sequences(X, self.abs_len,padding='post', value=0)
		X_words =  pad_sequences(X_words, self.abs_len, padding='post', value=0)
		Y 		=  pad_sequences(Y, self.abs_len, padding='post', value=0)
		# import pdb; pdb.set_trace()
		Y 		= to_categorical(Y, self.nclasses)		
		return (X, X_words, Y)

	def data_fetch(self):
		train, _, _		= self.get_data(self.data_path_train)
		test, _, _		= self.get_data(self.data_path_test)
		try:
			val, _, _		= self.get_data(self.data_path_val)
			return train, val, test 
		except Exception as e:
			return train, None, test 

	def get_sizes(self):
		return (self.abs_len , self.maxlen, self.maxlen_word)



class transfer_learning_dataset():
	"""docstring for ClassName"""
	def __init__(self, oringinal_dataset):
		self.oringinal_dataset 	= oringinal_dataset
		self.idx2label 			= dict((k,v) for v,k in oringinal_dataset.label2idx.items())
		# print(self.idx2label)

	def numeric_label_transform(self, label):
		switcher = {
			u'BACKGROUND+OBJECTIVE'		: u'0',
			u'DESCRIPTION'				: u'1',
			u'OBSERVATION+CONCLUSIONS'	: u'2',
		}
		return switcher.get(label, u'X')

	def get_evaluation_data(self, dataset, filename):
		input_folder = self.oringinal_dataset.get_input_folder(dataset)
		return self.oringinal_dataset.get_data(input_folder+filename) # eval_data,  orig_sentence, metadata	
	
	def calculate_abstract_wise_confidence(self, prediction):
		confidences = []
		abs_len = self.oringinal_dataset.abs_len
		for abstract in prediction:
			line_pred = np.argmax(abstract,axis=1)
			line_conf = np.max(abstract,axis=1)
			sumd_confidence = 0
			line_count = 0
			for i  in range(abs_len):
				if line_pred[i] != 0:
					line_count +=1
					sumd_confidence += line_conf[i] 
			confidences.append(sumd_confidence / line_count)
		return confidences

	def get_confident_prediction_indices(self, confidences, threshold):
		confident_prediction_indices = []
		for i in range(len(confidences)):
			if confidences[i] >= threshold:
				confident_prediction_indices.append(i)
		return confident_prediction_indices
	
	def get_prediction(self, pred, gold):
		catagorical_gold		=  np.zeros(pred.shape[:-1])
		catagorical_pred 		=  np.zeros(pred.shape[:-1])
		pred_confidence 		=  np.zeros(pred.shape[:-1])
		for (i,abstract) in enumerate(gold):
			for (j,label) in enumerate(abstract):
				gold_line_label = np.argmax(label)  
				if gold_line_label != 0:   
					catagorical_gold[i,j] 	= np.argmax(gold[i,j])
					catagorical_pred[i,j] 	= np.argmax(pred[i,j])
					pred_confidence[i,j]	= max(pred[i,j])
		return catagorical_gold, catagorical_pred, pred_confidence

	def update_golden_data(self, pred, gold, threshold):
		new_gold		=  np.zeros(pred.shape)
		# print(pred.shape)
		for (i,abstract) in enumerate(pred):
			for (j,label) in enumerate(abstract):
				# print(pred[i,j], threshold)
				# if pred[i,j] >= threshold: 
				# 	new_gold[i,j] 		= pred[i,j]
				# else:  
				# 	new_gold[i,j] 		= gold[i,j]
				new_gold[i,j] 		= pred[i,j]

		return new_gold

	def build_new_file_with_prediction(self, outfile_path, metadata, orig_data, pred, gold, threshold =90):
		try:
			outfile = open(outfile_path, 'w')
			label = self.update_golden_data(pred, gold, threshold)
			for (i, abstract_metadata) in enumerate(metadata):
				outfile.write(abstract_metadata+'\n')
				for j, line in enumerate(orig_data[i]):
					try:
						label_i_j = self.numeric_label_transform(self.idx2label[ label[i,j] ])
					except KeyError as e:
						label_i_j = u'X'
					outfile.write(label_i_j+'\t  '+orig_data[i][j]+'\n')
				outfile.write('\n\n')
			outfile.close()
			print('File updated with confidant prediction saved :\n'+outfile_path)
		except Exception as e:
			import traceback; traceback.print_exc()
			pdb.set_trace() 



	def undo_onehot(self, val):
		ret = np.zeros(val.shape[0:2])
		for (i,v) in enumerate(val):
			for (j,y) in enumerate(v):   
				ret[i,j] = np.argmax(y)
		return ret

# gold = f(Y_test)


# def f1(val):
#  ret = np.zeros(val.shape[0:2])
#  for (i,v) in enumerate(val):
#   for (j,y) in enumerate(v):   
#    ret[i,j] = np.argmax(y)
#  return ret

# pred = f1(eval_pred)