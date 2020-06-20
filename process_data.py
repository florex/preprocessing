__author__ = 'Ibaaslabs'
from preprocessing.PreProcessor import Preprocessor
p = Preprocessor("data/resume_samples.txt","data/classes_saved.txt","data/skills.txt")
#p.transform_data3() # for resume to matrices
#p.transform_data4() # For doc2vec enconding
p.wc()