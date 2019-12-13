import traceback, pdb
import numpy as np
import data_loader, embeddings_loader, model_defination
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras as K
	
# import importlib
# importlib.reload(...)

# pdb.set_trace();
# !import code; code.interact(local=vars())		


def main(num_epoch = 10, embed_top_n =  -1, load_model_weight_from_disk= True, save_model_weight_to_disk=True):
	# Definition of some parameters
	try:
		
		seed = 20
		np.random.seed(seed)
		
		NUM_EPOCHS = num_epoch
		BATCH_SIZE = 4

		dataset 			= data_loader.training_dataset('pubmed_non_rct')
		embeddings 			= embeddings_loader.glove_embeddings_loader(embed_top_n)
		train, val, test 	= dataset.data_fetch() 

		

		(X_train, X_train_words, Y_train)	= train
		(X_val, X_val_words, Y_val)			= val
		(X_test, X_test_words, Y_test)		= test

		

		# 
		model_name							= 'model_Jin'
		load_file_sufix						= '_3_pubmed_non_rct' 											# model_weights_3_pubmed_non_rct
		save_file_sufix 					= '_'+str(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])




		# if load_model_from_disk:
		# 	model =  load_model()
		# else:
		# 	model = None

		# if model == None:

		abstract_segmentor = model_defination.abstract_segmentation_model(model_name, dataset, embeddings)
		model = abstract_segmentor.model

		if load_model_weight_from_disk:
			abstract_segmentor.load_weights(load_file_sufix)
			# if model_weights  is not None:
			# 	model = model_weights


		plot_model(model, to_file=model_name+save_file_sufix+'.png', show_shapes=True)


		print('Training...')


		# class_weight = {0: 1.,
		# 				1: 5.,
		# 				2: 5.,
		# 				3: 5.}


		
		callbacks = [	TensorBoard(log_dir ='./logs', histogram_freq=0, write_graph=True, write_images=True),
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
						# validation_split=0.1,
		     			validation_data=([X_val, X_val_words], Y_val), 
						epochs=NUM_EPOCHS, 
						callbacks=callbacks) 
						# class_weight=class_weight)
		# model.fit([X_train, X_train_words], Y_train, batch_size=BATCH_SIZE, validation_data=([X_val, X_val_words], Y_val),  epochs=NUM_EPOCHS,  callbacks=callbacks) 
						# class_weight=class_weight)



		# print 'passed checkpoint 9: training\n\n'

		if save_model_weight_to_disk:
			abstract_segmentor.save_weights(save_file_sufix)


		# forecast result
		# predir = '/users/debarshi/soumya/abstractAnalysis/ANMLAD/pubmed_non_rct/model_output/predictions'
		# fileprefix = 'embedding_level_attention_'

		# scores = predict_score(model, [test_lex, pad_test_lex], test_toks, test_y, predir, idx2label, maxlen, fileprefix=fileprefix)


		print('Predicting...')

		pred_probs = model.predict([X_test, X_test_words], verbose=1)
		# test_loss = model.evaluate(x=[test_lex, pad_test_lex], y=test_y, batch_size=1, verbose=1)
		# pred = np.argmax(pred_probs, axis=1)
		# golden = np.argmax(Y_test, axis=1)

		# CM 	= confusion_matrix(golden, pred)
		# acc = accuracy_score(golden, pred)
		acc, CM, pred, gold = abstract_segmentor.get_results(pred_probs, Y_test)

		

		print ('Test Accuracy :'+ str(acc) +'\n Confusion_matrix : \n')

		print (CM)

		print ("base model ready!!")
		# pdb.set_trace()
		TL =  False
		bootstrap = False

		if (TL):
			evaluation_dataset 					= data_loader.secondary_dataset(dataset)
			evaluation_data 					= evaluation_dataset.get_evaluation_data('/users/soumya/arxiv_final/arxiv.txt')  
			# evaluation_data = evaluation_dataset.get_evaluation_data('/users/soumya/IEEE/TLT.txt')
			eval_tensor, orig_data, metadata 	= evaluation_data

			(X_eval, X_eval_words, Y_eval)		= eval_tensor

			# eval_probs = model.predict([X_eval, X_eval_words], verbose=1)	




			X_e_train 		= X_eval[:60]
			X_e_train_words = X_eval_words[:60]
			Y_e_train 		= Y_eval[:60]

			X_e_test 		= X_eval[60:]
			X_e_test_words 	= X_eval_words[60:]
			Y_e_test 		= Y_eval[60:]


			eval_probs = model.predict([X_e_test, X_e_test_words], verbose=1)
			acc_E, CM_E, pred_E, gold_E = abstract_segmentor.get_results(eval_probs, Y_e_test)
			print ('Before retraining \nEval Test Accuracy :'+ str(acc_E)  )

			# print('\n Eval Confusion_matrix : \n')
			# print (CM_E)


			print ('Retraining now\n' )
			model.fit([X_e_train, X_e_train_words], Y_e_train, batch_size=BATCH_SIZE,validation_split=0.1,epochs=10,callbacks=callbacks)



			eval_probs = model.predict([X_e_test, X_e_test_words], verbose=1)
			acc_E, CM_E, pred_E, gold_E = abstract_segmentor.get_results(eval_probs, Y_e_test)
			print ('After retraining \nEval Test Accuracy :'+ str(acc_E)  )
		elif bootstrap:
			evaluation_dataset 					= data_loader.secondary_dataset(dataset)
			evaluation_data = evaluation_dataset.get_evaluation_data('/users/soumya/IEEE/TLT.txt')
			eval_tensor, orig_data, metadata 	= evaluation_data
			(X_eval, X_eval_words, Y_eval)		= eval_tensor

			eval_probs 	= model.predict([X_eval, X_eval_words], verbose=1)



			conf 		= evaluation_dataset.calculate_abstract_wise_confidence(eval_probs)
			idx 		= evaluation_dataset.get_confident_prediction_indices(conf, .85)


			pdb.set_trace()

			X_e_train 		= X_eval[idx]
			X_e_train_words = X_eval_words[idx]
			Y_e_train 		= Y_eval[idx]

			model.fit([X_e_train, X_e_train_words], Y_e_train, batch_size=BATCH_SIZE,validation_split=0.1,epochs=10,callbacks=callbacks)






		# else: # go to debug mode
		# 	pdb.set_trace();
		# 	# !import code; code.interact(local=vars())

		# 	gold, pred, confidence = evaluation_dataset.get_prediction(eval_probs, Y_eval)
		# 	confidence_threshold = .6
		# 	evaluation_dataset.build_new_file_with_prediction('/users/soumya/test_3.txt', metadata, orig_data, pred, gold, confidence_threshold)








		# print ('Eval Accuracy :'+ str(acc_E) +'\n Confusion_matrix : \n')

		# print (CM_E)
		

		# pdb.set_trace() 
		# print 'passed checkpoint 10: prediction\n\n'
		# plot_model(model, to_file='model.png', show_shapes=True)

	except Exception as e:
		# print 'passed checkpoint E1\n\n'
		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		traceback.print_exc()
		pdb.set_trace() 



import sys
args = sys.argv

try:
	num_epoch =  int(args[1])
except Exception as e:
	num_epoch 	= 10
try:
	embed_top_n =  int(args[2])
except Exception as e:
	embed_top_n 	= -1 # all
try:
	load_model_weight_from_disk =  Bool(args[3])
except Exception as e:
	load_model_weight_from_disk 	= True
try:
	save_model_weight_to_disk =  Bool(args[4])
except Exception as e:
	save_model_weight_to_disk 	= True



if __name__== "__main__":
	# main(load_model_weight_from_disk= True, save_model_weight_to_disk=True, embed_top_n =  None):
	main(num_epoch, embed_top_n, load_model_weight_from_disk, save_model_weight_to_disk)



# dataset 			= data_loader('pubmed_non_rct')
# print(dataset.get_sizes())
# train, val, test 	= dataset.data_fetch()
# print('after fetch:')
# print(dataset.get_sizes())

# wl = dataset.wordlengths


# k = 100
# sss = sum(i > k for i in wl)
# per = sss/len(wl)*100
# print( "Less than %.3f%% sentences are longer than " % per +str(k)+" words.")


# # indexes = np.arange(dataset.wordlengths)
# # import matplotlib.pyplot as plt
# # plt.bar(indexes, dataset.wordlengths)
# # plt.savefig('wordlengths_hist.png')



# import pdb; pdb.set_trace()


