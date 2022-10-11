# corrector
Natural language text correction by fine-tuning large text2text-generation language models (huggingface) using synthetic noisy data.

## French texts
Fine-tunning is performed over noisy/clean parallel texts:

* French literary texts downloaded from the project gutenberg site (https://www.gutenberg.org/) 
* French news crawled data (https://data.statmt.org/news-crawl/fr/news.2021.fr.shuffled.deduped.gz)

Noise is injected helped by:
* Morphalou3.1 lexicon (https://hdl.handle.net/11403/morphalou/v3.1/morphalou3.1_formatcsv_toutenun.zip)



