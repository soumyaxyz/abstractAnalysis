from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract, Flatten, LSTM, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate, Add, Multiply
from keras.utils import plot_model, to_categorical
from keras.models import Model, model_from_json, load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras.backend as K
import numpy as np
import collections, os, re, traceback, pdb


def get_data(data_path, maxlen=200, maxlen_char_word = 40):
	#  data_path = 'data/pubmed_non_rct.txt'
	
	input_data = list(open(data_path, 'r'))

	tt = TweetTokenizer()
	labels = [u'BACKGROUND', u'OBJECTIVE', u'METHODS', u'RESULTS',  u'CONCLUSIONS']
	raw_data = []    
	final_data = []    

	
	for line in input_data:
			line = line.strip()           

			if not line.startswith('###'):#, 0,len(line)):
				if line:                                       
					raw_data.append(line) # The original traning data                                
					data = tt.tokenize(line)
					label = data[0]
					data =  data[1:] 
					data = [word.lower() for word in data if word.isalnum()]

					if label in labels:
						if len(data)<=3:
							if label == final_data[-1][0]:
								final_data[-1][1]+=data
							else : 
								# skip this line
								pass                            
						else :
							final_data.append([label, data]) #
										   

	senc_adr = []
	labels  = set()
	words   = set()
	lines   = []

	vec_senc_adr = []
	vec_senc = []
	vec_adr = []
	
	for data_line in final_data:
		labels.add(data_line[0])
		for word in data_line[1:][0]:
			# import pdb; pdb.set_trace() 
			words.add(word)


	labels  = list(labels)
	words   = list(words)

	word2idx    = {w: i+1 for i, w in enumerate(words)}
	label2idx   = {l: i+1 for i, l in enumerate(labels)}
	
	idx2label   = dict((k,v) for v,k in label2idx.items())
	idx2word    = dict((k,v) for v,k in word2idx.items())
	
	# idx2label[0]    = 'PAD'
	idx2word[0]     = 'PAD'
	
	# vec_senc_adr = []
	vec_sentence = []
	vec_label = []
	tok_senc_adr = []
	char_per_word = []
	char_word = []
	char_senc = []
	# maxlen_char_word = 0
	# a = []
	charcounts = collections.Counter()
	
	for i in final_data:
		vec_sentence.append([word2idx[word] for word in i[1]])
		vec_label.append(label2idx[i[0]]-1)
		# tok_senc_adr.append(i[1])  
		for w in i[1]:  # word in sentence
			for c in w.lower(): # character in word
				char_per_word.append(c)
				charcounts[c] += 1
				
			if len(char_per_word) > 40:
				# a.append(char_per_word)
				char_per_word = char_per_word[:40]
			if len(char_per_word) > maxlen_char_word:
				maxlen_char_word = len(char_per_word)

			char_word.append(char_per_word)
			char_per_word = []

		char_senc.append(char_word)
		char_word = []

	chars = [charcount[0] for charcount in charcounts.most_common()]
	char2idx = {c: i+1 for i, c in enumerate(chars)}

	
	char_word_lex = []
	char_lex = []
	char_word = []
	for senc in char_senc:
		for word in senc:
			for charac in word:
				char_word_lex.append([char2idx[charac]])
			
			char_word.append(char_word_lex)
			char_word_lex = []
			
		char_lex.append(char_word)
		char_word = []
	  
	

	char_per_word = []  
	char_per_senc = [] 
	char_senc = []
	for s in char_lex:
		for w in s:
			for c in w:
				for e in c:
					char_per_word.append(e)
			char_per_senc.append(char_per_word)
			char_per_word = []
		char_senc.append(char_per_senc)
		char_per_senc = []

	maxlen_now = max([len(l) for l in vec_sentence])

	if maxlen_now > maxlen:
		maxlen = maxlen_now # 174
	   
	pad_char_all = []
	for senc in char_senc:
		while len(senc) < maxlen:
			senc.insert(0, [])
		pad_senc = pad_sequences(senc, maxlen=maxlen_char_word)
		pad_char_all.append(pad_senc)
		pad_senc = []


		
	X_words = np.array(pad_char_all)           
   


	idx2char  = dict((k,v) for v,k in char2idx.items())
	idx2char[0] = 'PAD'
	charsize =  max(idx2char.keys()) + 1
	
	
	# print maxlen

	vocsize =  len(words)+1
	nclasses = len(labels)#+1
	
	

	X = pad_sequences(vec_sentence, maxlen=maxlen)
	Y = to_categorical(vec_label)   

	# import pdb; pdb.set_trace()
	return idx2word, idx2label, maxlen, maxlen_char_word, vocsize, charsize, nclasses, X, X_words, Y



def data_fetch(	data_path_train='/users/debarshi/soumya/PubMedData/output/train_clean.txt', 
				data_path_test = '/users/debarshi/soumya/PubMedData/output/test_clean.txt', 
				w2v_glove_300d_path = '/users/debarshi/soumya/abstractAnalysis/ANMLAD/pubmed_adr/word2vec_format_glove.42B.300d.txt',
				embed_dim = 300,
				char_embed_dim = 100
				):

	idx2word, idx2label, maxlen, maxlen_word, vocsize, charsize, nclasses,  X, X_words, Y = get_data(data_path_train)
	# _, _, maxlen_new, maxlen_word_new, _, charsize_test,  _,  X_test, X_test_words, Y_test = get_data(data_path_test, maxlen, maxlen_word)

	# if maxlen_new != maxlen or maxlen_word_new != maxlen_word:
	# 	idx2word, idx2label, maxlen, maxlen_word, vocsize, charsize_train,  nclasses,  X_train, X_train_words, Y_train = get_data(data_path_train, maxlen_new, maxlen_word_new)

	# charsize = max( charsize_train,  charsize_test) 
	data_size = X.shape[0]
	traning_size = int(data_size*.9)


	# try:
	X_train = X[:traning_size,:] 
	X_train_words =X_words[:traning_size,:,:]
	Y_train = Y[:traning_size,:]

	X_test = X[traning_size:,:]  
	X_test_words = X_words[traning_size:,:, :]  
	Y_test = Y[traning_size:,:] 			
	# except Exception as e:
	# 	pdb.set_trace() 


	print('Loading word embeddings...')
	w2v = KeyedVectors.load_word2vec_format(w2v_glove_300d_path, binary=False, unicode_errors='ignore', limit=5000)
	# w2v = KeyedVectors.load_word2vec_format(glove_300d_path, binary=False, unicode_errors='ignore')
	print('word embeddings loading done!')

	data = (idx2word, idx2label, maxlen, maxlen_word, vocsize, charsize, nclasses,  X_train, X_train_words, Y_train, X_test, X_test_words, Y_test, w2v , embed_dim, char_embed_dim)

	return data








def init_embedding_weights(i2w, w2vmodel):
	# Create initial embedding weights matrix
	# Return: np.array with dim [vocabsize, embeddingsize]

	d = 300
	V = len(i2w)
	assert sorted(i2w.keys()) == list(range(V))  # verify indices are sequential

	emb = np.zeros([V,d])
	num_unknownwords = 0
	unknow_words = []
	for i,l in i2w.items():
		if i==0:
			continue
		if l in w2vmodel.vocab:
			emb[i, :] = w2vmodel[l]
		else:
			num_unknownwords += 1
			unknow_words.append(l)
			emb[i] = np.random.uniform(-1, 1, d)
	return emb, num_unknownwords, unknow_words 


def model_basic(data):

	idx2word, idx2label, maxlen, maxlen_word, vocsize,  charsize, nclasses, _, _, _, _,_, _, w2v , embed_dim, char_embed_dim = data
	
	try:
				
		# Build the model
		print('Building the model...')
		
		main_input = Input(shape=[ maxlen], dtype='int32', name='input') # (None, 36)
		char_input = Input(shape=[ maxlen,  maxlen_word], dtype='int32', name='char_input') # (None, 36, 25)
		# print 'passed checkpoint 1: input\n\n'

		embeds, _, _  = init_embedding_weights( idx2word,  w2v)

		# print 'passed checkpoint 2: embedding init\n\n'

		embed = Embedding(input_dim= vocsize, output_dim= embed_dim, input_length= maxlen,
						  weights=[embeds], mask_zero=False, name='embedding', trainable=True)(main_input)

		# embed = Dropout(0.5, name='embed_dropout')(embed)


		char_embed =  Embedding(input_dim=charsize, output_dim=char_embed_dim, embeddings_initializer='lecun_uniform', input_length=[maxlen, maxlen_word], mask_zero=False, name='char_embedding')(char_input)
	 
		char_embed_shape = char_embed.shape
		char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, char_embed_shape[-2], char_embed_dim)))(char_embed)
			
		# pdb.set_trace()	

		biLSTM_char_embed = Bidirectional(LSTM(150, return_sequences=False))(char_embed)


		# fwd_state = GRU(150, return_state=True)(char_embed)[-2]
		# bwd_state = GRU(150, return_state=True, go_backwards=True)(char_embed)[-2]
		# biLSTM_char_embed = Concatenate(axis=-1)([fwd_state, bwd_state])
		char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, char_embed_shape[1], 2 * 150]))(biLSTM_char_embed)		

		# char_embed = Dropout(0.5, name='char_embed_dropout')(char_embed)
		# pdb.set_trace()

		combined_embed = Concatenate( name='Sum')([embed, char_embed])

		# print 'passed checkpoint 3: embedding + dropout layer\n\n'
		biLSTM = Bidirectional(LSTM(64, return_sequences=False))(combined_embed)
		biLSTM = Dropout(0.5)(biLSTM)

		# print 'passed checkpoint 4: biLSTM + dropout\n\n'

		norm = BatchNormalization()(biLSTM)

		# print 'passed checkpoint 5: normailzation\n\n'

		
		feedforword = Dense(nclasses, name='feed_forword')(norm)
		# final_output = CRF(nclasses, learn_mode='marginal')(feedforword)


		final_output = Activation('softmax')(feedforword) # (None, 36, 5) # (None, 36, 5)

		# print 'passed checkpoint 7: Final_classifier\n\n'

		model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	except Exception as e:
		# print 'passed checkpoint E1\n\n'
		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		traceback.print_exc()
		pdb.set_trace() 

def save_model_weight(model):
	# serialize model to JSON
	# model_json = model.to_json()
	# with open("saved_model/model.json", "w") as json_file:
	#     json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("saved_model/model_weights.h5")
	print("Saved model weights to disk")
	# model.save("saved_model/model.h5")
	# print("Saved model to disk.")


def load_model_weight(model):
	try:
		# son_file = open('saved_model/model.json', 'r')
		# loaded_model_json = json_file.read()
		# json_file.close()
		# loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		from pathlib import Path
		saved_model = Path("saved_model/model_weights.h5")
		if saved_model.is_file():
			model.load_weights("saved_model/model_weights.h5")
			print("Loaded model weights from disk")
			# model = load_model(saved_model)
			# print("Loaded model from disk.")
			return model
		else :
			print("Saved weights not found.")
	except Exception as e:
		traceback.print_exc()
		pdb.set_trace() 

		return None


def main(load_model_weight_from_disk= True, save_model_weight_to_disk=True):
	# Definition of some parameters
	try:
		
		seed = 20
		np.random.seed(seed)
		
		NUM_EPOCHS = 10
		BATCH_SIZE = 16

		data =  data_fetch()   # defaults	
		# if load_model_from_disk:
		# 	model =  load_model()
		# else:
		# 	model = None

		# if model == None:
		model = model_basic(data)

		if load_model_weight_from_disk:
			model_weights = load_model_weight(model)
			if model_weights  is not None:
				model = model_weights


		plot_model(model, to_file='model.png', show_shapes=True)


		print('Training...')

		_, _, _, _, _, _, _,  X_train, X_train_words, Y_train, X_test, X_test_words, Y_test, _, _, _ = data



		
		callbacks = [	TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
						EarlyStopping(monitor='val_loss', patience=2)#,
      #        			ModelCheckpoint(filepath='best_model.ckpt', monitor='val_loss', save_best_only=True)
             		]
		# define model
		# model.fit(X_train, Y_train,
		#       batch_size=batch_size,
		#       epochs=nb_epoch,
		#       validation_data=(X_test, Y_test),
		#       shuffle=True,
		#       callbacks=[tensorboard])


		# print 'passed checkpoint 11\n\n'
		history = model.fit([X_train, X_train_words], Y_train, 
						batch_size=BATCH_SIZE, 
						validation_split=0.1,
		     			# validation_data=([X_test, X_test_words], Y_test), 
						epochs=NUM_EPOCHS, 
						callbacks=callbacks)
		# print 'passed checkpoint 9: training\n\n'

		if save_model_weight_to_disk:
			save_model_weight(model)


		# forecast result
		predir = '/users/debarshi/soumya/abstractAnalysis/ANMLAD/pubmed_non_rct/model_output/predictions'
		fileprefix = 'embedding_level_attention_'

		# scores = predict_score(model, [test_lex, pad_test_lex], test_toks, test_y, predir, idx2label, maxlen, fileprefix=fileprefix)


		print('Predicting...')

		pred_probs = model.predict([X_test, X_test_words], verbose=1)
		# test_loss = model.evaluate(x=[test_lex, pad_test_lex], y=test_y, batch_size=1, verbose=1)
		pred = np.argmax(pred_probs, axis=1)
		golden = np.argmax(Y_test, axis=1)

		CM 	= confusion_matrix(golden, pred)
		acc = accuracy_score(golden, pred)
		print ('Accuracy :'+ str(acc) +'\n Confusion_matrix : \n')

		print (CM)
		
		# pdb.set_trace() 

		# print 'passed checkpoint 10: prediction\n\n'
		plot_model(model, to_file='model.png', show_shapes=True)

	except Exception as e:
		# print 'passed checkpoint E1\n\n'
		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		traceback.print_exc()
		pdb.set_trace() 


if __name__== "__main__":
	# main(load_model_weight_from_disk= True, save_model_weight_to_disk=True):
	main(True, True)