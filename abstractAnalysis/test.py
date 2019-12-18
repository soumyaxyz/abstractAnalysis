import data_loader_ , importlib
importlib.reload(data_loader_)       

dataset                     = data_loader_.training_dataset('pubmed_non_rct')   
train, val, test    = dataset.data_fetch(4)  

w2i = dict()
for word in sentence:
	try:
		w2i[word]
	except KeyError as e:
		w2i[word] = len(w2i)+1

