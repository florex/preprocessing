# preprocessing
This module transforms raw text resumes into matrices or vectors for training

============Folder structure==============

This module contains 2 main sub-directories
- data
- preprocessing

The sub-directory preprocessing contains the class Preprocessor which is designed to 
proprocess raw text resumes contained in the file (data/resume_samples.txt). 

# Dependencies :
- Anaconda
- numpy
- nltk
- gensim
- spacy

# Execution :
To preprocessed raw text resumes,   

First, change the path to the output_dir in the file preprocessor.py (edit the property output_dir of the class Preprocessor)

Then run the command :

python process_data.py

This operation creates subdatasets corresponding to the resume length in the directory <output_dir> 

The operation creates two additional files :
   - resumes_refs.py : which associates a sample data id to the path of the original resume.
   - resumes_words.py : which represents a dictionary where keys are hashed words'vectors and values are corresponding words


The resulting dataset is in form of a matrix : 

The first column of the matrix is an integer representing the id of the resume.

The last 10 columns represents the classes of the resumes.

The column between the id and the classes represent the flatten form of a resume matrix.

By default, each resume is encoded into a matrix of shape (500,100).

# Cite this :
Jiechieu, K.F.F., Tsopze, N. Skills prediction based on multi-label resume classification using CNN with model predictions explanation. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05302-x
