__author__ = 'Ibaaslabs'
from preprocessing.PreProcessor import Preprocessor
p = Preprocessor("data/resume_samples.txt","data/classes_saved.txt","data/skills.txt")
#p.transform_data3() # transforms resumes to matrices
#p.transform_data4() # transform resume to vectors using doc2vec enconding
p.wc() # computes the min, max and the average word counts of resumes. 
