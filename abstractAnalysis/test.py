import data_loader_ , importlib
importlib.reload(data_loader_)       

dataset                     = data_loader_.training_dataset('pubmed_non_rct')   
train, val, test    = dataset.data_fetch()    