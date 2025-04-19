# BytePairTokenizer
This is an implementation from scratch of the byte pair algorithm first described in 1994 by philip Gage.
These days, it is used to build a vocbular that an machine learning model can use when performing NLP tasks.
My class utilizes NumPy, vectorized functions, and multiprocessing to enhance computational performance. 

The file tokenizer.py includes the class implementing the encoder. 
The file BuildCorpus.py builds a corpus of a given size using some books and wikipedia articles, and runs the tokenizer with a specified number of cores and up to a specified size.

A more detailed explanation can be found in the following medium story:

