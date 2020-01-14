import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pprint, pdb

# directory ='./'
pp = pprint.PrettyPrinter(indent=4)

target_names = ['pad','BAC', 'TEQ', 'OBS']
target_names_premutated = ['pad',  'OBSERVATION', 'BACKGROUND', 'TECHNIQUE']
acc_list = []

def walk_on(directory):
	# print(directory)
	for root,dirs,files in os.walk(directory):
		for file in files:
			if file.endswith(".csv"):
				# print(file)
				names = file[6:-4].split('_')
				try:
					data =  pd.read_csv(directory+file,names=[ "Pred", "Gold"])
					gold = data['Gold'].values.tolist()
					pred = data['Pred'].values.tolist()
					# CM 	= confusion_matrix(gold, pred)
					acc = accuracy_score(gold, pred)
					acc_list.append(acc)
					CR 	= classification_report(gold, pred, digits=4,labels=[0,1,2,3],target_names=target_names)    
					# pdb.set_trace()
					print( names, acc, '\n')
					# pp.pprint(CR)

				except Exception as e:
					# pass
					raise e

def generate_on(directory):
	# print(directory)
	for root,dirs,files in os.walk(directory):
		for file in files:
			if file.endswith(".csv"):
				# print(file)
				names = file[6:-4].split('_')
				try:
					print(names)
					if names[-2] == 'arxiv' or names[-2] == 'merged':
						f(directory,file, names[-2], [0,1,3,2])
					else:
						f(directory,file,names[-2])
				except Exception as e:
					# pass
					raise e
		
def f(directory,file, name, label_premutation=[0,1,2,3]):
	data =  pd.read_csv(directory+file,names=[ "Pred", "Gold"])
	gold = data['Gold'].values.tolist()	
	pred = data['Pred'].values.tolist()

	CR 	= classification_report(gold, pred, digits=4,labels=label_premutation,target_names=target_names)
	CM = confusion_matrix(gold, pred, labels=label_premutation)
	# pdb.set_trace()
	if CM.shape[0]>3:
		CM = CM[1:,1:]
	# permutation = [1,2,0]
	# CM = CM[permutation] 
	# CM = CM[:,permutation]
	print(file, name)

	pp.pprint(CM)
	pp.pprint(CR)
	# pdb.set_trace()
	CM_norm = np.round(CM *100 / CM.astype(np.float).sum(axis=0),1 )
	pp.pprint(CM)
	sns.set(font_scale=3)
	HM = sns.heatmap(CM_norm.astype(int), annot=CM, cbar=False, cmap='Blues', fmt='d',xticklabels=target_names[1:], yticklabels=target_names[1:])#, linewidths=1, linecolor='black')
	HM.set_yticklabels(HM.get_yticklabels(), rotation = 90, va='center')#, fontsize = 12)
	HM.set_xticklabels(HM.get_xticklabels(), rotation = 0)#, fontsize = 12)
	plt.savefig(name+'_confusion_matrix.png')

	print('accuracy_score : ', accuracy_score(gold, pred),'\n_____')
	# plt.clf()
	plt.show()                                                                                                                                            
	return CR
	# cmap='Blues'


# pp.pprint(f('./20k_arxiv/','model_Jin_arxiv_74.csv'))

# pp.pprint(f('./20k_TLT/','model_Jin_2k_IEEE_TLT_110.csv'))

# pp.pprint(f('./backup_TPAMI/run2/','model_Jin_pubmed_non_rct_14999_IEEE_TPAMI_110.csv'))


# pp.pprint(f('./20k_merged/','model_Jin_2k_merged_194.csv'))

# import code; code.interact(local=vars())	

generate_on('./best_runs/')
# plt.plot([*range(len(acc_list))],acc_list)
# print(([*range(len(acc_list))],'\n',acc_list))
# plt.show()

# pp.pprint(f('./20k_arxiv/','model_Jin_arxiv_trained_on_predicted.csv'))