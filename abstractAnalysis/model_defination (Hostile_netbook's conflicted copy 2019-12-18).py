from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from keras_contrib.layers import CRF
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract, Flatten, LSTM, CuDNNLSTM, BatchNormalization, Layer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate, Add, Multiply
from keras.utils import plot_model
from keras.models import Model, model_from_json, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.backend as K
import traceback, pdb
import numpy as np
import model_defination


class abstract_segmentation_model():
	def __init__(self, model_name, dataset, embeddings):
		self.model 			= self.get_model(model_name, dataset, embeddings)
		self.save_location 	= 'saved_model/model_weights'

	def get_model(self, model_name, dataset, embeddings):
		# pdb.set_trace()
		try:
			return getattr(model_defination, model_name)().get_model_defination(dataset, embeddings)
		except AttributeError as e:
			return 	model_initial().get_model_defination(dataset, embeddings)

	def save_weights(self, sufix = ''):
		self.model.save_weights(self.save_location+sufix+".h5")
		# self.model.save('saved_model/model'+sufix+'.h5')
		print("Saved model and model weights to disk")

	def load_weights(self, sufix= ''):
		try:
			from pathlib import Path
			saved_model = Path(self.save_location+sufix+".h5")
			if saved_model.is_file():
				self.model.load_weights(self.save_location+sufix+".h5")
				print("Loaded model weights from disk")
				return #self.model
			else :
				print("Saved weights not found.")
		except Exception as e:
			print("Saved weights incompatable.")
			# traceback.print_exc()
			# pdb.set_trace() 
			return #None

	def get_results(self, pred, gold):
		non_pad_pred 	= []
		non_pad_gold	= []
		for (i,abstract) in enumerate(gold):
			 for (j,label) in enumerate(abstract):
			 	g_l = np.argmax(label)  
			 	if g_l != 0:
			 		p_l = np.argmax(pred[i,j])
			 		non_pad_pred.append(p_l)
			 		non_pad_gold.append(g_l)
		CM 	= confusion_matrix(non_pad_gold, non_pad_pred)
		acc = accuracy_score(non_pad_gold, non_pad_pred)

		return acc, CM, non_pad_pred, non_pad_gold


# class model_stub():
# 	def

## Dernoncourt et al.

class model_Dernoncourt():
	def get_model_defination(self, dataset, embeddings):
		# try:
					
		# Build the model
		print('Building the model...')
		
		main_input = Input(shape=[dataset.abs_len, dataset.maxlen], dtype='int32', name='input') # (None, 36)
		char_input = Input(shape=[dataset.abs_len, dataset.maxlen,  dataset.maxlen_word], dtype='int32', name='char_input') # (None, 36, 25)
		# print 'passed checkpoint 1: input\n\n'
		# pdb.set_trace()
		main_input_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen)))(main_input)

		char_input_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen, dataset.maxlen_word)))(char_input)

		embeds, _, _  = embeddings.init_weights(dataset.idx2word)

		# print 'passed checkpoint 2: embedding init\n\n'

		embed = Embedding(input_dim= dataset.vocsize, output_dim= embeddings.embed_dim, input_length= dataset.maxlen,
						  weights=[embeds], mask_zero=False, name='embedding', trainable=True)(main_input_r)

		embed = Lambda(lambda x: K.reshape(x, shape=[-1, dataset.abs_len, dataset.maxlen, embeddings.embed_dim]))(embed)

		# embed = Dropout(0.5, name='embed_dropout')(embed)


		char_embed =  Embedding(input_dim=dataset.charsize, output_dim=embeddings.char_embed_dim, embeddings_initializer='lecun_uniform', 
						input_length=[dataset.maxlen, dataset.maxlen_word], mask_zero=False, name='char_embedding')(char_input_r)
	 
		char_embed_shape = char_embed.shape
		char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen_word, embeddings.char_embed_dim)))(char_embed)
			
		# pdb.set_trace()


		biLSTM_char_embed = Bidirectional(CuDNNLSTM(embeddings.char_embed_dim, return_sequences=False))(char_embed)


		# fwd_state = GRU(150, return_state=True)(char_embed)[-2]
		# bwd_state = GRU(150, return_state=True, go_backwards=True)(char_embed)[-2]
		# biLSTM_char_embed = Concatenate(axis=-1)([fwd_state, bwd_state])
		char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, dataset.abs_len, char_embed_shape[1], 2 * embeddings.char_embed_dim]))(biLSTM_char_embed)		

		# char_embed = Dropout(0.5, name='char_embed_dropout')(char_embed)
		# pdb.set_trace()

		combined_embed = Concatenate( name='Sum')([embed, char_embed])

		combined_embed = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen, (2*embeddings.char_embed_dim+ embeddings.embed_dim) )))(combined_embed)
		
		
		# biLSTM_embed= Bidirectional(LSTM(64, return_sequences=True))(combined_embed)

		# with_atention = AttentionWithContext()(biLSTM_embed)

		# with_atention = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*64 )))(with_atention)

		biLSTM = Bidirectional(CuDNNLSTM(64, return_sequences=False))(combined_embed)
		biLSTM = Dropout(0.5)(biLSTM)

		biLSTM_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*64)))(biLSTM)


		norm = BatchNormalization()(biLSTM_r)
		feedforward = Dense(dataset.nclasses, name='feed_forword')(norm)

		final_output = CRF(dataset.nclasses, learn_mode='marginal', sparse_target=True)(feedforward)
		# final_output = Activation('softmax')(feedforward) # (None, 36, 5) # (None, 36, 5)

		
		# print 'passed checkpoint 7: Final_classifier\n\n'

		model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		
		# except Exception as e:
		# 	# print 'passed checkpoint E1\n\n'
		# 	# model.summary()
		# 	# plot_model(model, to_file='model.png', show_shapes=True)

		# 	traceback.print_exc()
		# 	pdb.set_trace() 

	
		return model


## Jin et al.
	





class model_Jin():

	def dot_product(x, kernel):
		"""
		Wrapper for dot product operation, in order to be compatible with both
		Theano and Tensorflow
		Args:
			x (): input
			kernel (): weights
		Returns:
		"""
		if K.backend() == 'tensorflow':
			# todo: check that this is correct
			return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
		else:
			return K.dot(x, kernel)

	class Attention(Layer):
		def __init__(self,
					 W_regularizer=None, b_regularizer=None,
					 W_constraint=None, b_constraint=None,
					 bias=True,
					 return_attention=False,
					 **kwargs):
			"""
			Keras Layer that implements an Attention mechanism for temporal data.
			Supports Masking.
			Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
			# Input shape
				3D tensor with shape: `(samples, steps, features)`.
			# Output shape
				2D tensor with shape: `(samples, features)`.
			:param kwargs:
			Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
			The dimensions are inferred based on the output shape of the RNN.
			Note: The layer has been tested with Keras 1.x
			Example:
			
				# 1
				model.add(LSTM(64, return_sequences=True))
				model.add(Attention())
				# next add a Dense layer (for classification/regression) or whatever...
				# 2 - Get the attention scores
				hidden = LSTM(64, return_sequences=True)(words)
				sentence, word_scores = Attention(return_attention=True)(hidden)
			"""
			self.supports_masking = True
			self.return_attention = return_attention
			self.init = initializers.get('glorot_uniform')
			self.W_regularizer = regularizers.get(W_regularizer)
			self.b_regularizer = regularizers.get(b_regularizer)
			self.W_constraint = constraints.get(W_constraint)
			self.b_constraint = constraints.get(b_constraint)
			self.bias = bias
			super(model_Jin.Attention, self).__init__(**kwargs)
		
		def build(self, input_shape):
			assert len(input_shape) == 3
			self.W = self.add_weight((input_shape[-1],),
									 initializer=self.init,
									 name='{}_W'.format(self.name),
									 regularizer=self.W_regularizer,
									 constraint=self.W_constraint)
			if self.bias:
				self.b = self.add_weight((input_shape[1],),
										 initializer='zero',
										 name='{}_b'.format(self.name),
										 regularizer=self.b_regularizer,
										 constraint=self.b_constraint)
			else:
				self.b = None
			
			self.built = True
		
		def compute_mask(self, input, input_mask=None):
			# do not pass the mask to the next layers
			return None
		
		def call(self, x, mask=None):
			eij = model_Jin.dot_product(x, self.W)
			if self.bias:
				eij += self.b
			
			eij = K.tanh(eij)
			a = K.exp(eij)
			# apply mask after the exp. will be re-normalized next
			if mask is not None:
				# Cast the mask to floatX to avoid float64 upcasting in theano
				a *= K.cast(mask, K.floatx())
			# in some cases especially in the early stages of training the sum may be almost zero
			# and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
			# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
			a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
			weighted_input = x * K.expand_dims(a)
			result = K.sum(weighted_input, axis=1)
			if self.return_attention:
				return [result, a]
			return result
		
		def compute_output_shape(self, input_shape):
			if self.return_attention:
				return [(input_shape[0], input_shape[-1]),
						(input_shape[0], input_shape[1])]
			else:
				return input_shape[0], input_shape[-1]

	def get_model_defination(self, dataset, embeddings):
		try:
						
			# Build the model
			print('Building the model...')
			lstm_dim = 64

		### token embedding layer
			
			main_input = Input(shape=[dataset.abs_len, dataset.maxlen], dtype='int32', name='input') # (None, 36)

			# unused
			char_input = Input(shape=[dataset.abs_len, dataset.maxlen,  dataset.maxlen_word], dtype='int32', name='char_input') # (None, 36, 25)
			
			main_input_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen)))(main_input)	

			embeds, _, _  = embeddings.init_weights(dataset.idx2word)		
			embed = Embedding(input_dim= dataset.vocsize, output_dim= embeddings.embed_dim, input_length= dataset.maxlen, weights=[embeds], mask_zero=False, name='embedding', trainable=True)(main_input_r)


			# embed = Lambda(lambda x: K.reshape(x, shape=[-1, dataset.abs_len, dataset.maxlen,  embeddings.embed_dim]))(embed)

			# embed = Lambda(lambda x: K.reshape(x, shape=[-1,  dataset.maxlen,  embeddings.embed_dim]))(embed)

		### sentence encoding layer		

			# pdb.set_trace();
			blstm_layer = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))(embed)
			
			attention_layer = self.Attention()(blstm_layer)
			# attention_layer  = blstm_layer
			
			sentence_encoding_layer = attention_layer# Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim )))(attention_layer)


			# biLSTM = Dropout(0.5)(blstm_layer)

			sentence_encoding_layer = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim)))(attention_layer)



			# biLSTM_embed= Bidirectional(LSTM(64, return_sequences=True))(combined_embed)

			# pdb.set_trace();

		### context enriching layer

			biLSTM = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=False))(sentence_encoding_layer)
			biLSTM = Dropout(0.5)(biLSTM)
			
			biLSTM_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim)))(biLSTM)


			norm = BatchNormalization()(biLSTM_r)
			feedforward = Dense(dataset.nclasses, name='feed_forword')(norm)


		### label sequence optimazion layer

			final_output = CRF(dataset.nclasses, learn_mode='marginal', sparse_target=True)(feedforward)
			# final_output = Activation('softmax')(feedforward) # (None, 36, 5) # (None, 36, 5)

			model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

			# model.summary()
			# plot_model(model, to_file='model.png', show_shapes=True)

			
		except Exception as e:
			# print 'passed checkpoint E1\n\n'
			# model.summary()
			# plot_model(model, to_file='model.png', show_shapes=True)

			traceback.print_exc()
			pdb.set_trace() 

	
		return model