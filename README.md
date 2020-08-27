# Abstract segmentation with sparse data
This is the source code for the paper [**Segmenting Scientific Abstracts into Discourse Categories: A  Deep Learning-Based Approach for Sparse Labeled Data**](https://dl.acm.org/doi/abs/10.1145/3383583.3398598) (  [Arxiv preprint](https://arxiv.org/abs/2005.05414)  ), 	presented in JCDL 2020.

##  Data
This repository includes the dataset of segemented  CS abstracts
| Data|Source |Directory|
|:----| ---- |---- |
| `PubMed-non-RCT`<sup>1</sup>	| [non RCT articles from PubMed](https://pubmed.ncbi.nlm.nih.gov/advanced/) |[PubMedData/](https://github.com/soumyaxyz/abstractAnalysis/tree/master/PubMedData)|
|`cs.NI`| [cs.networks subdomain from arxiv.org](https://arxiv.org/list/cs.NI/recent) |[arxiv_final/](https://github.com/soumyaxyz/abstractAnalysis/tree/master/Merged)|
|`cs.TLT` |[IEEE Transactions on Learning Technologies](https://www.computer.org/csdl/journal/lt)|[IEEE_final/TLT/](https://github.com/soumyaxyz/abstractAnalysis/tree/master/IEEE_final/TLT)|
|`cs.TPAMI` |[IEEE Transactions on Transactions on Pattern Analysis and Machine Intelligence](https://www.computer.org/csdl/journal/tp)|[IEEE_final/TPAMI/](https://github.com/soumyaxyz/abstractAnalysis/tree/master/Merged)|
|`cs.combined`|`cs.NI` + `cs.TLT` + `cs.TPAMI` |[Merged/](https://github.com/soumyaxyz/abstractAnalysis/tree/master/Merged)|


<sup>1</sup>  The `PubMed-non-RCT` dataset was too large to be included in this repository. The code to bulid the dataset is provided along with a small sample of data.

## Embeddings
 We utilized the [Common Crawl (42B tokens 300 dimention)](http://nlp.stanford.edu/data/glove.42B.300d.zip) GLOVE embedding in [word2vec format](https://bartoszptak.github.io/gensim/2019/06/14/gensim-convert-glove-to-word2vec.html). 
## Dependencies
-   python 3.5.6
-   tensorflow 1.10.0
-   keras 2.2.4
-   keras-self-attention 0.47.0
-   sklearn 0.20.3





## Usage

1.  Navigate to [Code/](https://github.com/soumyaxyz/abstractAnalysis/blob/master/Code/)
2. Set the `PRETRAINED_EMBEDDINGS` location<sup>2</sup> in line 5 of [Code/embeddings_loader.py](https://github.com/soumyaxyz/abstractAnalysis/blob/master/Code/embeddings_loader.py#L5)

3. Run [`abstract_analysis.py`](https://github.com/soumyaxyz/abstractAnalysis/blob/master/Code/abstract_analysis.py) 
```
    python abstract_analysis.py -h
    usage: abstract_analysis.py [-h] [-b] [-f] [-s]
                                [{arxiv,IEEE_TLT,IEEE_TPAMI,merged}]
                                [retraining_size]

    positional arguments:
      {arxiv,IEEE_TLT,IEEE_TPAMI,merged}
                            The evaluation dataset, default= arxiv
      retraining_size        Data size for fine tuning, default= 340

    optional arguments:
      -h, --help            show this help message and exit
      -b, --generate_baseline
                            For generating baseline without pre training
      -f, --fine_tune_with_pred
                            For evaluating the effect of transfer learning
      -s, --predict_and_save
                            To generate labels for unlabled abstracts,
                            conflicts with -f/--fine_tune_with_pred
   ```
  <sup>2</sup>  This might cause issues with [line endings](http://www.cs.toronto.edu/~krueger/csc209h/tut/line-endings.html#:~:text=Text%20files%20created%20on%20DOS,(%22%5Cn%22)). To solve the issue open and save all files in the local system.  
