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
from keras_self_attention import SeqSelfAttention
import traceback, pdb
import numpy as np
import model_defination


class abstract_segmentation_model():
	def __init__(self, model_name, dataset, embeddings, flags = None):
		self.model 			= self.get_model(model_name, dataset, embeddings, flags)
		self.model_name		= model_name+'_'+''.join(str(int(i)) for i in flags)
		self.save_location 	= 'saved_model/model_weights'

	def get_model(self, model_name,  dataset, embeddings, flags = None):
		# pdb.set_trace()
		try:
			if flags == None:
				return getattr(model_defination, model_name)().get_model_defination(dataset, embeddings)
			else:
				return getattr(model_defination, model_name)(flags).get_model_defination(dataset, embeddings)
		except AttributeError as e:
			return 	model_Jin_backup().get_model_defination(dataset, embeddings)

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
				print("Saved weights not found.",Path(self.save_location+sufix+".h5"))
		except Exception as e:
			print("Saved weights incompatable.")
			# traceback.print_exc()
			# pdb.set_trace() 
			return #None

	


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
class model_Jin_backup():
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
			
			attention_layer =SeqSelfAttention(attention_activation='sigmoid')(blstm_layer)
			# attention_layer  = blstm_layer
			
			sentence_encoding_layer = attention_layer

			# pdb.set_trace()

			# biLSTM = Dropout(0.5)(blstm_layer)

			# sentence_encoding_layer = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim)))(attention_layer)



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
			
			# model.summary()
			# plot_model(model, to_file='model.png', show_shapes=True)

			traceback.print_exc()
			pdb.set_trace() 

	
		return model


class model_Jin():
	def __init__(self, flags = None):
		try:
			(f_1, f_2, f_3, f_4) =  flags
			self.embedding 	= f_1
			self.encoding 	= f_2
			self.enriching 	= f_3
			self.optimazion	= f_4
		except Exception as e:
			self.embedding 	= True
			self.encoding 	= True
			self.enriching 	= True
			self.optimazion	= True


	def get_model_defination(self, dataset, embeddings ):
		try:
						
			# Build the model
			print('Building the model...')
			lstm_dim = 64

			### token embedding layer		
			# unused , left for compatability
			char_input = Input(shape=[dataset.abs_len, dataset.maxlen,  dataset.maxlen_word], dtype='int32', name='char_input') 			
			if self.embedding:
				main_input = Input(shape=[dataset.abs_len, dataset.maxlen], dtype='int32', name='input') # (None, 35, 180)
				main_input_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen)))(main_input)		
				embeds, _, _  = embeddings.init_weights(dataset.idx2word)		
				embed = Embedding(input_dim= dataset.vocsize, output_dim= embeddings.embed_dim, input_length= dataset.maxlen, weights=[embeds], mask_zero=False, name='embedding', trainable=True)(main_input_r)
				token_embedding_layer = embed
			else:
				main_input = Input(shape=[dataset.abs_len, dataset.maxlen], dtype='float32', name='input') # (None, 35, 180)
				main_input_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.maxlen, 1)))(main_input)	
				token_embedding_layer = main_input_r

			### sentence encoding layer		
			if self.encoding:			
				blstm_layer = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))(token_embedding_layer)			
				attention_layer =SeqSelfAttention(attention_activation='sigmoid')(blstm_layer)
				sentence_encoding_layer = attention_layer
			else:
				sentence_encoding_layer = token_embedding_layer			

			### context enriching layer
			if self.enriching:	
				biLSTM = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=False))(sentence_encoding_layer)
				biLSTM = Dropout(0.5)(biLSTM)
				
				biLSTM_r = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim)))(biLSTM)


				norm = BatchNormalization()(biLSTM_r)
				abstract_processing_layer = Dense(dataset.nclasses, name='feed_forword')(norm)
			else:
				abs_layer_in = Lambda(lambda x: K.reshape(x, shape=(-1, dataset.abs_len, 2*lstm_dim*dataset.maxlen)))(sentence_encoding_layer)
				feedforward = Dense(dataset.maxlen, name='feed_forword_1')(abs_layer_in)
				norm = BatchNormalization()(feedforward)
				abstract_processing_layer = Dense(dataset.nclasses, name='feed_forword')(norm)
				
			### label sequence optimazion layer
			if self.optimazion:			
				final_output = CRF(dataset.nclasses, learn_mode='marginal', sparse_target=True)(abstract_processing_layer)
			else:
				final_output = Activation('softmax')(abstract_processing_layer) # (None, 35, 4) 

			model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

			# model.summary()
			# plot_model(model, to_file='model.png', show_shapes=True)

			
		except Exception as e:
			
			# model.summary()
			# plot_model(model, to_file='model.png', show_shapes=True)

			traceback.print_exc()
			pdb.set_trace() 

	
		return model