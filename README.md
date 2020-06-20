# preprocessing
This module transforms raw resumes into matrices or vectors for training


============Folder structure==============

This module contains main 3 sub-directories
- code
- data
- preprocessing

The sub-directory preprocessing contains the class Preprocessor which is designed to 
proprocess raw text resumes contained in the file (data/skills_it.txt). 

Dependences :
- numpy
- nltk
- gensim
- spacy

Execution :
To preprocessed raw text resume, simply run the command 

python process_data.py #inside the the directory deeplearning



This operation creates subdatasets corresponding to the resume length in the directory datasets/<cv_length>/
The operation creates two additional files :
   - resumes_refs.py : associate a sample data id to the path of the original resume.
   - resumes_words.py : dictionary where keys are hashed words'vectors and values are corresponding words


In the resulting matrices, each ligne represent the flatten form of the resume matrix
The first column of the matrix is an integer representing the id of the resume
The last 10 columns represents the classes of the resumes
The column between the id and the classes represent the flatten form of the matrix representing the resume.
By default, each resume is encoded into a matrix of shape (500,100).
