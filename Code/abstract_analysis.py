import traceback, pdb
import numpy as np
import data_loader, embeddings_loader, model_defination
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, precision_score, f1_score
import keras as K

def partition(X_eval, X_eval_words, Y_eval, traning_size):
	# pdb.set_trace()
	X_e_train 		= X_eval[-traning_size:]
	X_e_train_words = X_eval_words[-traning_size:]
	Y_e_train 		= Y_eval[-traning_size:]
	return ( X_e_train, X_e_train_words, Y_e_train)

def get_results(pred, gold):
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
	CR 	= None #classification_report(non_pad_gold, non_pad_pred)
	return acc, CM, CR, non_pad_pred, non_pad_gold

def get_prediction_acc(model, X_e_test, X_e_test_words,  Y_e_test):
	eval_pred = model.predict([X_e_test, X_e_test_words], verbose=1)
	acc_E, CM_E, CR_E, pred_E, gold_E = get_results(eval_pred, Y_e_test)	
	print ('After retraining \nEval Test Accuracy :'+ str(acc_E)  )
	return pred_E, gold_E

def get_trained_model(dataset_name, model_name, num_epoch = 10, embed_top_n =  -1, load_file_sufix = '_model_Jin_pubmed_non_rct', batch_size =4, abletive_flags= None):
	try:
		# Definition of some parameters
		load_model_weight_from_disk = False
		save_model_weight_to_disk 	= False
		seed = 20
		np.random.seed(seed)

		BATCH_SIZE 		= 4


		dataset 			= data_loader.training_dataset(dataset_name)
		embeddings 			= embeddings_loader.glove_embeddings_loader(embed_top_n)
		train, val, test 	= dataset.data_fetch() 

		

		(X_train, X_train_words, Y_train)	= train
		if val is not None:
			(X_val, X_val_words, Y_val)			= val
		(X_test, X_test_words, Y_test)		= test		

		
		
		if load_file_sufix is not None:
			load_model_weight_from_disk = True
		else:
			save_model_weight_to_disk = True

		

		# if load_model_from_disk:
		# 	model =  load_model()
		# else:
		# 	model = None

		# if model == None:

		abstract_segmentor = model_defination.abstract_segmentation_model(model_name, dataset, embeddings, abletive_flags)
		model = abstract_segmentor.model


		save_file_sufix = '_'+abstract_segmentor.model_name+'_'+dataset_name+'_'+str(X_train.shape[0])

		if load_model_weight_from_disk:
			abstract_segmentor.load_weights(load_file_sufix)
			# if model_weights  is not None:
			# 	model = model_weights


		plot_model(model, to_file=save_file_sufix+'.png', show_shapes=True)


		print('Training...')


		# class_weight = {0: 1.,
		# 				1: 5.,
		# 				2: 5.,
		# 				3: 5.}
		
		callbacks = [	TensorBoard(log_dir ='./logs', histogram_freq=0, write_graph=True, write_images=True),
						EarlyStopping(monitor='val_loss', patience=2)#,
						# ModelCheckpoint(filepath='best_model.ckpt', monitor='val_loss', save_best_only=True)
					]
		if val is not None:
			history = model.fit([X_train, X_train_words], Y_train, batch_size=	batch_size, validation_data=([X_val, X_val_words], Y_val), epochs=num_epoch,callbacks=callbacks) # class_weight=class_weight)
		else:
			history = model.fit([X_train, X_train_words], Y_train, batch_size=	batch_size, validation_split=0.1, epochs=num_epoch,callbacks=callbacks) # class_weight=class_weight)
						



		# print 'passed checkpoint 9: training\n\n'

		if save_model_weight_to_disk:
			print('Model weights saved in model\\'+save_file_sufix+'.h5')
			abstract_segmentor.save_weights(save_file_sufix)

		



		print('Predicting...')

		pred_probs = model.predict([X_test, X_test_words], verbose=1)
		# test_loss = model.evaluate(x=[test_lex, pad_test_lex], y=test_y, batch_size=1, verbose=1)
		
		acc, CM, CR, pred, gold = get_results(pred_probs, Y_test)

		

		print ('Test Accuracy :'+ str(acc) +'\n Confusion_matrix : \n')
		print (CM)

		print ("base model ready!!")

		return model, dataset

	except Exception as e:
		traceback.print_exc()
		pdb.set_trace()

def evaluate_transfer_learning(trained_model, trained_on_dataset, eval_dataset, eval_test_filename, eval_train_filename, retraning_size,  test_size, retrain_epoch, batch_size):
	try:		
		# pdb.set_trace()
		model 								= trained_model		
		evaluation_dataset 					= data_loader.transfer_learning_dataset(trained_on_dataset)
		evaluation_test_data 				= evaluation_dataset.get_evaluation_data(eval_dataset, eval_test_filename)  
		eval_test, orig_data, metadata 		= evaluation_test_data
		(X_test, X_test_words, Y_test)		= eval_test

		pred_E, gold_E = get_prediction_acc(model, X_test, X_test_words, Y_test)

		#
		print('retraning_size : ',retraning_size)
		print('eval_train_filename : ', eval_train_filename)


		if retraning_size > 0:

			evaluation_train_data 				= evaluation_dataset.get_evaluation_data(eval_dataset, eval_train_filename)  
			eval_train , orig_data, metadata 	= evaluation_train_data
			(X_train, X_train_words, Y_train)	= eval_train
			(X_train, X_train_words, Y_train)	= partition(X_train, X_train_words, Y_train, retraning_size)

			print ('Retraining now\n' )

			callbacks = [	TensorBoard(log_dir ='./logs', histogram_freq=0, write_graph=True, write_images=True),
							EarlyStopping(monitor='val_loss', patience=2)#,
							# ModelCheckpoint(filepath='best_model.ckpt', monitor='val_loss', save_best_only=True)
						]

			# pdb.set_trace()
			model.fit([X_train, X_train_words], Y_train, batch_size=batch_size, validation_split=0.1, epochs=retrain_epoch,callbacks=callbacks)

			pred_E, gold_E = get_prediction_acc(model,  X_test, X_test_words, Y_test)

		return pred_E, gold_E, model

	except Exception as e:
		traceback.print_exc()
		pdb.set_trace() 

def generate_prediction(trained_model, trained_on_dataset, unlabled_dataset, unlabled_filename):
	evaluation_dataset 				= data_loader.transfer_learning_dataset(trained_on_dataset)
	unlabled_data 					= evaluation_dataset.get_evaluation_data(unlabled_dataset, unlabled_filename)  
	unlabled, orig_data, metadata 	= unlabled_data

	(X_test, X_test_words, Y_test)	= unlabled

	eval_pred = trained_model.predict([X_test, X_test_words], verbose=1)

	pred = evaluation_dataset.undo_onehot(eval_pred)
	gold = evaluation_dataset.undo_onehot(Y_test)

	evaluation_dataset.build_new_file_with_prediction('pred_'+unlabled_filename, metadata, orig_data, pred, gold)


def main(retraning_size, 
		generate_baseline 	= False,
		predict_and_save	= False,
		fine_tune_with_pred = False, 
		eval_dataset 		= 'arxiv'):
	# pdb.set_trace()

	
	# flages for abletive study
	f_1  = True						# token embeding layer	
	f_2  = True						# sentence encoding layer
	f_3  = True						# context enriching layer
	f_4  = True						# sequence optimazion layer
	flags = (f_1, f_2, f_3, f_4)
	# flag =''
	# for f in flags:
	# 	flag +=str(int(f))
	flag = str(list(map(int,flags))).replace(', ','')[1:-1]   # same as above
	if flag == '1111':
		flag = ''
	else:
		flag = '_'+flag



	#set configs
	# generate_baseline 	= False
	# predict_and_save	= False
	# fine_tune_with_pred = False  		

	dataset_name 		= 'pubmed_non_rct' 								# initial traning dataset pubmed_non_rct
	# eval_dataset 		= 'arxiv'									# transfer learning dataset
	eval_test_filename	= 'test_clean.txt'								# transfer learning dataset test filename
	eval_train_filename	= 'train_clean.txt'								# transfer learning dataset train filename
	model_name			= 'model_Jin'   								# model_Dernoncourt, model_Jin
	load_file_sufix 	= 'model_Jin'+flag+'_pubmed_non_rct_14999'   	# saved model weights etc     
	batch_size 			= 4
	if predict_and_save:
		unlabled_dataset 	= 'arxiv'									# dataset to be labled
		unlabled_filename	= 'arxiv_rest.txt'							# file to be labled
	if fine_tune_with_pred:
		predicted_dataset 	= 'arxiv'									# dataset from predicted labels
		predicted_train_file= 'arxiv_rest.txt'							# corosponding training filename

	

	if retraning_size < 0: 							# train model on initial dataset
		num_epoch 		= 10
		embed_top_n     =  50000 #-1   #all
		get_trained_model(dataset_name, model_name, num_epoch, embed_top_n,  None, batch_size, flags)


	else:												# evaluate transfer learning using model pre-trained on initial dataset
		num_epoch 		= 2
		embed_top_n		= 5000
		test_size 		= 40
		retrain_epoch 	= 10


		if(generate_baseline):
			dataset 			= data_loader.training_dataset(eval_dataset)
			embeddings 			= embeddings_loader.glove_embeddings_loader(embed_top_n)
			train, val, test 	= dataset.data_fetch() 

			(X_train, X_train_words, Y_train)	= train
			(X_test, X_test_words, Y_test)		= test			

			abstract_segmentor 	= model_defination.abstract_segmentation_model(model_name, dataset, embeddings, flags)
			trained_model 		= abstract_segmentor.model
			trained_on_dataset 	= dataset
		else:
			trained_model, trained_on_dataset = get_trained_model(dataset_name, model_name, num_epoch, embed_top_n,  load_file_sufix, batch_size , flags)

		pred_E, gold_E, finetuned_model = evaluate_transfer_learning(trained_model, trained_on_dataset, eval_dataset, eval_test_filename, eval_train_filename, retraning_size, test_size, retrain_epoch, batch_size)
		np.savetxt(load_file_sufix+'_'+eval_dataset+'_'+str(retraning_size)+'.csv', np.column_stack((pred_E, gold_E)), delimiter=",", fmt='%s')

		if predict_and_save:
			generate_prediction(finetuned_model, trained_on_dataset, unlabled_dataset, unlabled_filename)
		elif fine_tune_with_pred:
			pred_E, gold_E, model = evaluate_transfer_learning(finetuned_model, trained_on_dataset, eval_dataset, eval_test_filename, predicted_train_file, retraning_size, test_size, retrain_epoch, batch_size)
			np.savetxt(load_file_sufix+'_'+eval_dataset+'_trained_on_predicted.csv', np.column_stack((pred_E, gold_E)), delimiter=",", fmt='%s')

	# import code; code.interact(local=vars())

	# pdb.set_trace() 71428 

	
import sys
args = sys.argv



try:
	retraning_size =  int(args[1])
except Exception as e:
	retraning_size 	= 10




if __name__== "__main__":
	
	# main(retraning_size, True, 'arxiv')

	# main(340, eval_dataset = 'arxiv')
	# main(340, eval_dataset = 'IEEE_TLT')
	# main(340, eval_dataset = 'IEEE_TPAMI')
	main(340, eval_dataset = 'merged', predict_and_save	= True)
	# main(340,True)

	# print('\n\n\n\n\n now TPAMI')
	# retraning_sizes = [40,50,80,90,110] 
	# for retraning_size in retraning_sizes:
	# for retraning_size in range(80,120,10):
	# 	main(retraning_size)


