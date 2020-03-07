import numpy as np, seaborn as sns, pandas as pd
from collections import Counter 
import matplotlib.pyplot as plt
import termplot, pprint, pdb, traceback

	
	

  

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()

#the output file
output = None
if args.output:
	output = args.output
else:
	from os.path import dirname, basename, join
	output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default 
if args.num_aug:
	num_aug = args.num_aug

#how much to change each sentence
alpha = 0.1#default
if args.alpha:
	alpha = args.alpha

def plot_CM(CM, save = False, filename = 'confusion_matrix'):
	pp = pprint.PrettyPrinter(indent=4)
	
	target_names = ['pad','BAC', 'DES', 'OBS']

	# CM = np.round(CM *100 / CM.astype(np.float).sum(axis=0),1 )
	pp.pprint(CM)
	sns.set(font_scale=3)
	HM = sns.heatmap(CM.astype(int), annot=True, cbar=False, cmap='Blues', fmt='d',xticklabels=target_names[1:], yticklabels=target_names[1:])#, linewidths=1, linecolor='black')
	HM.set_yticklabels(HM.get_yticklabels(), rotation = 90)#, fontsize = 12)
	HM.set_xticklabels(HM.get_xticklabels(), rotation = 0)#, fontsize = 12)
	if save:
		plt.savefig(filename+'.png')
	plt.show()

def plot_distribution(line_label, labels, avg_num_of_lines):
	pp = pprint.PrettyPrinter(indent=4)

	normalized_line_label = np.array([line / line.sum() for line in line_label])
	normalized_line_label =  np.transpose(normalized_line_label)
	pp.pprint(line_label)

	colors=['#7f6d5f','#557f2d', '#2d7f5e']
	# pdb.set_trace()
		

	ind = [i/2 for i in range(avg_num_of_lines)]
	barWidth = .5
	for i, line in enumerate(normalized_line_label):
		# print(i,' ',normalized_line_label[i])
		if i==0:
			plt1 =plt.barh(ind, normalized_line_label[i], color=colors[i], edgecolor='white', height=barWidth, label=labels[i] )
		else:
			plt1 =plt.barh(ind, normalized_line_label[i] ,left=np.sum(normalized_line_label[:i], axis=0), color=colors[i], edgecolor='white', height=barWidth, label=labels[i])
	
	# plt.xticks()
	plt.xlim(0,1.04)
	plt.ylim(-1,9)
	plt.yticks([i/2 for i in range(0, 9)],range(1, 10))
	plt.legend(labels, bbox_to_anchor=(1.1, 1.1), ncol=3)
	plt.gca().invert_yaxis()
	# plt.savefig('distribution.png')
	
	plt.show()

	### rewriten with pandas
	# try:			
		# normalized_line_label = np.array([line / line.sum() for line in line_label])
		# df = pd.DataFrame(normalized_line_label, columns=['BACKGROUND','DESCRIPTION','OBSERVATION'] )
		# df.plot(kind='bar', stacked=True)
		# plt.show()
	# except Exception as e:
	# 	traceback.print_exc()
	# 	pdb.set_trace()	

def label_transform(label):
		switcher = { 
			u'0'			: 0, 
			u'1'			: 1,
			u'2'			: 2, 
			u'BACKGROUND'	: 0, 
			u'OBJECTIVE'	: 0, 
			u'METHODS'		: 1,
			u'RESULTS'		: 2,  
			u'CONCLUSIONS'	: 2,
		}
		return switcher.get(label.strip(), 0)   # should be 4

#generate more data with standard augmentation
def count(train_orig): 
	

	labels = ['BACKGROUND','TECHNIQUE','OBSERVATION'] 
	lines = open(train_orig, 'r', encoding="utf8").readlines()
	lines.append('###')

	
	preceding 		= Counter()
	suceeding		= Counter()
	count 			= Counter()
	abstract_size	= Counter()
	avg_num_of_lines = 9
	line_label 		= np.zeros((avg_num_of_lines,3))


	for label in labels:
		preceding[label] = Counter()
		suceeding[label] = Counter()
		for prev_label in labels:
			preceding[label][prev_label] = 0
			suceeding[label][prev_label] = 0
		# print(preceding[label])

	label_list 	= []

	abs_count = 0
	new_abstract = True
	for i, line in enumerate(lines):
		line = line.strip()

		if not line or line.startswith('#'):
			if new_abstract:
				abs_count   += 1
				prev 		= -1
				curr 		= -1
				if label_list:
					# print('_____\n',label_list,'_____')
					scale_factor = avg_num_of_lines/abstract_len
					for j in range(abstract_len):
						# print(j,' > ',int(j*scale_factor) ," : ", label_list[j] )
						line_label[int(j*scale_factor)][label_list[j]] += 1
						# if j*scale_factor >= 8  and label_list[j] == 0:
						# 	print( lines[i+2])
						# 	pp.pprint( lines[i-1])
						# 	print(line_label[int(j*scale_factor)])
						# 	pdb.set_trace()  



					# print('_____\n',label_list,'_____')
				label_list 	= []
				try:					
					# if abstract_size[abstract_len] == 0: 
					# 	print(abstract_len, abstract_size)
					abstract_size[abstract_len] += 1
				except Exception as e:
					pass
				abstract_len = 0
			new_abstract = False
		else: 
			new_abstract = True
			prev 	= curr
			try:
				line_split 	= line.split("\t", 1)
			except Exception as e:
				line_split 	= line.split("", 1)

			try:
				curr 	= int(line_split[0])
			except ValueError as e:
				# try:
				# 	int(label_transform(line_split[0]))
				# except Exception as e:					
				# 	pdb.set_trace()
				curr 	= label_transform(line_split[0])
			

			abstract_len			+=1
			count[labels[curr]] 	+=1

			# print(abstract_len,' > ', curr)
			label_list.append(curr)
			# line_label[]

			if prev > -1:
				preceding[labels[curr]] [labels[prev]]+=1
				suceeding[labels[prev]] [labels[curr]]+=1
					
	# abstract_size[abstract_len] += 1

	plot_distribution(line_label, labels, avg_num_of_lines)

	print('number of abstracts :', abs_count-1)

	# print('abstract_len :',abstract_size)

	# import pdb;pdb.set_trace()

	# bar = np.zeros( max(abstract_size.keys())+1 )
	# for key in abstract_size.keys():
	# 	bar[key] = abstract_size[key]
	# print(bar)
	# termplot.plot(bar, plot_height=10, plot_char='.')
	
	# pdb.set_trace()


	# plt.bar(abstract_size.keys(), abstract_size.values())
	# plt.plot([*abstract_size.keys()], [*abstract_size.values()])

	idx = sorted (abstract_size)
	nun_lines = [ abstract_size[i] for i in idx]
	# pdb.set_trace()	

	# plt.plot(idx, nun_lines  )
	# plt.show()
	wt_avg = sum(idx[i] * nun_lines[i] / sum(nun_lines) for i in range(len(idx)))

	print('average number of lines per abstracts : ', wt_avg)

	print('total number of lines :', sum(count.values()))
	# pdb.set_trace()

	pCM = np.zeros((3,3))
	sCM = np.zeros((3,3))

	print('\ncount of each samples:')
	for i in range(3):
		print (labels[i],' :', count[labels[i]])




	print('\npreceding label for samples:')
	for i in range(3):
		print(labels[i],':',preceding[labels[i]])
		for j in range(3):
			pCM[i][j] = preceding[labels[i]][labels[j]]

	# print('\nsuceeding label for samples:')
	# for i in range(3):
	# 	print(labels[i],':',suceeding[labels[i]])
	# 	for j in range(3):
	# 		sCM[i][j] = suceeding[labels[i]][labels[j]]


	# plot_CM(pCM, save = True, filename = 'preceding_matrix')
	# plot_CM(sCM, save = True, filename = 'suceeding_matrix')  

	
#main function
if __name__ == "__main__":

	#generate augmented sentences and output into a new file
	count(args.input)