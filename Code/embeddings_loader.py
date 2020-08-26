from gensim.models import KeyedVectors
import numpy as np
import traceback, pdb

PRETRAINED_EMBEDDINGS = '/users/soumya/abstractAnalysis/ANMLAD/pubmed_adr/word2vec_format_glove.42B.300d.txt'

class glove_embeddings_loader():

	def __init__(self, embed_top_n = -1):				
		w2v_glove_300d_path = PRETRAINED_EMBEDDINGS
		self.embed_dim 		= 300
		self.char_embed_dim = 25	

		print('Loading word embeddings...')
		if embed_top_n == -1:
			self.w2vmodel = KeyedVectors.load_word2vec_format(w2v_glove_300d_path, binary=False, unicode_errors='ignore')
		else:
			self.w2vmodel = KeyedVectors.load_word2vec_format(w2v_glove_300d_path, binary=False, unicode_errors='ignore', limit=embed_top_n)
		print('word embeddings loading done!')

	def init_weights(self, idx2word):		
		d = 300
		V = len(idx2word) -1  # -1 represents unknown words, thus removed from count

		# pdb.set_trace()
		assert sorted(idx2word.keys())[1:] == list(range(V))  # verify indices are sequential

		emb = np.zeros([V,d])
		num_unknownwords = 0
		unknow_words = []
		for i,l in idx2word.items():
			if i==0:
				continue
			if i==-1 or l not in self.w2vmodel.vocab:
				num_unknownwords += 1
				unknow_words.append(l)
				emb[i] = np.random.uniform(-1, 1, d)			
			else:
				emb[i, :] = self.w2vmodel[l]			
		return emb, num_unknownwords, unknow_words 
