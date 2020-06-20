__author__ = 'Ibaaslabs'
import nltk
import sys
import codecs
import math
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import  simple_preprocess
from gensim.test.utils import common_texts
import csv
from PIL import Image
from random import shuffle
import json
import csv
import   numpy
import hashlib
import pickle
from nltk.tokenize import RegexpTokenizer
import string
tokenizer = RegexpTokenizer('\/|^\.|\.$|,|;|\(|\)|^\-|\-$|:|;', gaps=True)
import spacy
import nltk
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

stopset = dict(zip(stopwords.words("english"),range(len(stopwords.words("english")))))

class Preprocessor(object):
    def __init__(self, data_x_path, data_y_path, skills_path, generate_models=False, wv_file="data/vectors.kv"):
        print("-----Begin processor initialization-----")
        self.bool = True
        self.known_samples = dict()
        self.ncol = 100
        self.max_cv_count = 4000
        self.cv_length = 700
        self.output_dir = "E:/Thèse/datasets/dws/"+str(self.cv_length)
        self.min_keywords = 20
        self.class_freq_ceil = 10
        self.resume_words = dict()
        self.data_x_file = codecs.open(data_x_path, "rU", encoding='utf-8', errors='ignore')
        self.data_y_file = codecs.open(data_y_path, "rU", encoding='utf-8', errors='ignore')
        #self.wordnet_lemmatizer = WordNetLemmatizer()
        if generate_models :
            self.skills_file = codecs.open(skills_path, "rU", encoding='utf-8', errors='ignore')
            self.tokens_in = set(tokenizer.tokenize(self.skills_file.read()))
            self.tokens_in = [token.lower() for token in self.tokens_in]
            print("number of tokens in before filtering{}".format(len(self.tokens_in)))
            #self.tokens_in = self.filter_token_in()
            #print ("number of tokens in after filtering{}".format(len(self.tokens_in)))
            #self.token_in_coverage()
            self.tokens_in = dict(zip(self.tokens_in, range(len(self.tokens_in))))
            print("Outputs initialization...")
            self.y_dict, self.tokens_out = self.init_outputs()
            #print (self.y_dict)
            #print(self.tokens_out)
            self.tokens_out = self.filter_tokens_out(100)
            self.tokens_out = dict(zip(self.tokens_out, range(len(self.tokens_out))))
            #print(self.tokens_out)
            print("Word vectors initialization...")
            self.wv = self.cv_to_matrix()
            self.wv.init_sims()
            #self.saveDic(self.tokens_in, "data/tokens_in.txt")
            self.saveDic(self.tokens_out, "data/tokens_out.txt")
            #self.saveDicAsModel(self.tokens_in, "data/tokens_in.json")
            self.saveDicAsModel(self.tokens_out, "data/tokens_out.json")
            self.saveDicAsModel(self.y_dict, "data/y_dict.json")
            #self.wv = self.cv_to_matrix()
            print("Computing TF_IDF")
            self.tf_idf = self.computeTFIDF()
        else :
            self.tokens_in = self.loadDicModel("data/tokens_in.json")
            self.tokens_out = self.loadDicModel("data/tokens_out.json")
            self.y_dict = self.loadDicModel("data/y_dict.json")
            self.wv = KeyedVectors.load(wv_file, mmap='r')
            self.wv.init_sims()
            #self.tf_idf = self.computeTFIDF()
            #self.computeTF()
            self.tf_idf = self.loadDicModel("tf_idf.json")

        #self.data_matrix = numpy.empty((0,len(self.tokens_out)+len(self.tokens_in)+1), dtype=numpy.uint16)
        #self.min,self.max = self.get_min_max()
        #self.flatten_matrix = numpy.empty((0,self.cv_length + len(self.tokens_out)+1), dtype=numpy.float32)
        self.flatten_matrix = numpy.empty((0, self.ncol*self.cv_length + len(self.tokens_out) + 1), dtype=numpy.float16)
        self.pickle_datas = []
        self.outputs = []
        self.resume_pkl = "cv_dataset.pkl"
        #self.doc2VecModel = self.build_d2v_model()
        print("-----End processor initialization-----")


    def get_tokens(self, text):
        sents = nltk.sent_tokenize(text)
        tokens = [token.lemma_ for sent in sents for token in nlp(sent.lower())
                  if token.lemma_ not in stopset and len(token.lemma_) > 2
                  and (token.lemma_ is not ":::" or token.lemma_ is not "::::::")]
        tokens = Counter(tokens)
        return tokens

    def filter_tokens_out(self,freq):
        filtered_tokens = [t for t,v in self.tokens_out.items() if v >= freq]
        #print (filtered_tokens)
        return filtered_tokens

    def init_outputs(self):
        y_dict = dict()
        self.data_y_file.seek(0)
        lines = self.data_y_file.read().splitlines()
        tokens_out = []
        for line in lines :
            line_items = line.split(":")
            y_dict.update({line_items[0].lower():line_items[1].split(",")})
            tokens_out+=line_items[1].split(",")
        return y_dict, Counter(tokens_out)

    def min_max_norm(self, x, min, max):
        if max == min :
            return 1
        else :
            return (x-min)*1.0/(max-min)

    def get_input_vec(self, line):
        tokens = self.get_tokens(line)
        if tokens.values() == [] :
            return []

        minimum = min(tokens.values())
        maximum = max(tokens.values())
        input = [0]*len(self.tokens_in)
        for key,value in tokens.iteritems():
            if key in self.tokens_in :
                input[self.tokens_in[key]] = 1
        return input

    def get_output_vec(self, output_items):
        outputs = output_items.split(';')
        output = [0]*len(self.tokens_out.keys())
        tokens = []
        for y in outputs :
            y_l = y.lower()
            if y_l in self.y_dict.keys() :
                #print (y_l)
                tokens +=self.y_dict.get(y_l)
        tokens = set(tokens)
        has_output = False
        for token in tokens:
            if token in self.tokens_out :
                has_output = True
                output[self.tokens_out[token]] = 1
        if has_output :
            return output
        return None

    def get_output_vec2(self, line):
        line_items = line.split(":::")
        outputs = line_items[1].split(';')
        output = [0]*len(self.tokens_out.keys())
        tokens = []
        for y in outputs :
            if y in self.y_dict.keys() :
                tokens +=self.y_dict.get(y)
        tokens = list(set(tokens))

        for token in tokens:
            if token in self.tokens_out :
                return self.tokens_out[token]
        return None

    def transform_data(self):
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        i = 0
        for line in lines :
            print("-----Begin iteration-----")
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    hash = hashlib.md5(line_items[2]).hexdigest()
                    if hash not in self.known_samples :
                        id = line_items[0]
                        self.known_samples.update({hash:True})
                        input_vector = self.get_input_vec(line)

                        output_vector = self.get_output_vec(line)

                        if input_vector==[0]*len(input_vector) or output_vector == [0]*len(output_vector) :
                            continue

                        if self.bool :
                            self.test_transform(output_vector)
                            print (line)
                            self.bool = False
                        input_vector = [i]+input_vector+output_vector
                        i+=1
                        self.data_matrix = numpy.append(self.data_matrix,numpy.array([input_vector],dtype=numpy.uint16), axis=0)
            print("------End iteration-----")
        print (self.data_matrix.shape)
        print (len(self.tokens_in)+len(self.tokens_out))
        #self.clean_attribute()
        print (len(self.tokens_in)+len(self.tokens_out))
        print (self.data_matrix.shape)
        self.saveDic(self.tokens_in,"data/tokens_in.txt")
        self.saveDic(self.tokens_out,"data/tokens_out.txt")
        header = ['id'] + sorted(self.tokens_in,key=self.tokens_in.get)+ sorted(self.tokens_out,key = self.tokens_out.get)
        header = ",".join(header)
        numpy.savetxt("data/matrix.csv", self.data_matrix, fmt="%d", delimiter=",", header=header)


    def transform_data2(self):
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        i = 0
        for line in lines :
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    hash = hashlib.md5(line_items[2]).hexdigest()
                    if hash not in self.known_samples :
                        matrix = numpy.empty((0,self.ncol),dtype=numpy.float32)
                        keys = sorted(self.tokens_in, key=self.tokens_in.get)
                        cpt = 0
                        for word in keys :
                            if word in line_items[2] :
                                if word in self.model.wv :
                                    wordvec = [ (x-self.min)/(self.max-self.min) for x in self.model.wv[word]]
                                    matrix = numpy.append(matrix,numpy.array([wordvec],dtype=numpy.float32), axis=0)
                                else :
                                    matrix = numpy.append(matrix,numpy.array([[0]*self.ncol],dtype=numpy.float32), axis=0)
                            else :
                                matrix = numpy.append(matrix,numpy.array([[0]*self.ncol],dtype=numpy.float32), axis=0)
                        #numpy.savetxt("data/cv_matrix/"+str(i)+".csv", matrix, fmt="%.2f", delimiter=",")
            i+=1


    def transform_data5(self): #no tfidf
        writers = {}
        for k, v in self.tokens_out.items():
            print(k+str(v))
            writers[v] = csv.writer(open(self.output_dir+"/"+k+".csv", 'w', newline=''), delimiter=',')
        i = 1
        test_writer = csv.writer(open(self.output_dir + "/tests.csv", 'w', newline=''), delimiter=',')
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        shuffle(lines)
        print("Lines shuffled")
        i = 1
        max_cpt = 0
        min_cpt = 10000
        test_cpt = 0
        total_cpt_count = 0
        references = {}
        class_counters = [0]*10
        negative_decount = [0]*10
        with open('resume_dataset.csv', 'w', newline='') as fp:
            writer =  csv.writer(fp, delimiter = ',')
            for line in lines :
                #print("Processing line N0"+str(i+1)+"#.....")
                if line is not "::::::" and not line.isspace():
                    line_items = line.split(":::")
                    if len(line_items) == 3 :
                        #hash = hashlib.md5(line_items[2].encode('utf-8')).hexdigest()
                        reference = line_items[0]
                        #if hash not in self.known_samples :
                        #print("Processing cv... : "+reference+"\n")
                        self.known_samples.update({hash: True})
                        matrix = numpy.empty((0,self.ncol),dtype=numpy.float32)
                        #keys = sorted(self.tokens_in, key=self.tokens_in.get)
                        cpt = 0
                        cv_sents = nltk.sent_tokenize(line_items[2])
                        cv_tokens = [token.lemma_ for sent in  cv_sents for token in nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower()))))]
                        shuffle(cv_tokens)
                        cv_tokens = set(cv_tokens)
                        #cv_tokens = set(token.lemma_ for sent in  cv_sents for token in nlp(sent))
                        #shuffle(cv_tokens)
                        output = self.get_output_vec(line_items[1])
                        words = dict()
                        for word in cv_tokens :
                            if word in self.wv and word not in stopset and word in self.tokens_in:
                                #wordvec = [ (x-self.min)/(self.max-self.min) for x in self.model.wv[word]]
                                find = False
                                for index in range(10) :
                                    if output is not None and output[index]==1 :
                                        find = True
                                        break
                                if find :
                                    vect = self.wv.word_vec(word, use_norm=True)
                                    matrix = numpy.append(matrix,numpy.array([vect],dtype=numpy.float32), axis=0)
                                    words.update({cpt:word})
                                    cpt+=1
                                    if cpt == self.cv_length :
                                        break

                        if cpt > max_cpt:
                            max_cpt = cpt
                        if cpt < min_cpt:
                            min_cpt = cpt

                        for j in range(self.cv_length-cpt) :
                            matrix = numpy.append(matrix,numpy.array([[0]*self.ncol],dtype=numpy.float32), axis=0)

                        if output is not None and cpt > self.min_keywords :
                            matrix = numpy.insert(numpy.append(matrix.reshape((1,self.ncol*self.cv_length)),output),0,i)
                            matrix = matrix.reshape(1,1+len(self.tokens_out)+self.ncol*self.cv_length)
                            if test_cpt > 5000 :
                                for index in writers :
                                    if class_counters[index] < self.max_cv_count :
                                        if output[index] == 1 :
                                            negative_decount[index] += 1
                                            class_counters[index]+=1
                                            #print("index = {:d} in matrix = {:.2f} {:s}".format(index, matrix[0,index - 10], line))
                                            writers[index].writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                            self.resume_words.update({i:words})
                                        else :
                                            if negative_decount[index] > 0 :
                                                writers[index].writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                                negative_decount[index] -= 1
                                                self.resume_words.update({i:words})
                            else :
                                test_writer.writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                self.resume_words.update({i: words})
                                test_cpt += 1

                            #self.flatten_matrix = numpy.append(self.flatten_matrix,matrix, axis=0)
                            #self.outputs.append(output)
                            references.update({str(i):reference})
                            total_cpt_count += cpt
                            if i%1000==0 :
                                print("cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} avg_cpt = {:.2f}".format(i,reference, cpt, min_cpt, max_cpt, total_cpt_count*1.0/i))
                            i += 1
                        elif cpt < self.min_keywords :
                            print("-----cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} ".format(i, reference, cpt, min_cpt, max_cpt))
            fp.close()
        print(class_counters)
        refs = json.dumps(references,indent=4)
        f = open(self.output_dir+"/"+"resumes_refs.json", "w")
        f.write(refs)
        f.close()
        refs = json.dumps(self.resume_words, indent=4)
        f = open(self.output_dir+"/"+"resumes_words.json", "w")
        f.write(refs)
        f.close()
        #print (self.outputs)
        #header = ['id'] + sorted(self.tokens_in, key=self.tokens_in.get) + sorted(self.tokens_out,key=self.tokens_out.get)
        #header = ",".join(header)
        #numpy.savetxt("resume_dataset.csv", self.flatten_matrix, fmt="%.2f", delimiter=",",header=header)

    def data_analysis(self):
        writers = {}
        i = 1
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        print("Lines shuffled")
        i = 1
        max_cpt = 0
        test_cpt = 0
        min_cpt = 10000
        total_cpt_count = 0
        references = {}
        class_counters = [0]*10
        negative_decount = [0]*10
        max_wc = 0
        min_wc = 5000
        total_wc = 0
        with open('resume_dataset.csv', 'w', newline='') as fp:
            for line in lines :
                #print("Processing line N0"+str(i+1)+"#.....")
                if line is not "::::::" and not line.isspace():
                    line_items = line.split(":::")
                    if len(line_items) == 3 :
                        #hash = hashlib.md5(line_items[2].encode('utf-8')).hexdigest()
                        reference = line_items[0]
                        #if hash not in self.known_samples :
                        #print("Processing cv... : "+reference+"\n")
                        self.known_samples.update({hash: True})
                        matrix = numpy.empty((0,self.ncol),dtype=numpy.float32)
                        #keys = sorted(self.tokens_in, key=self.tokens_in.get)
                        cpt = 0
                        cv_sents = nltk.sent_tokenize(line_items[2])
                        cv_tokens = [token.lemma_ for sent in  cv_sents for token in nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower()))))]
                        wc = len(cv_tokens)

                        if wc > max_wc :
                            max_wc = wc
                        if wc < min_wc :
                            min_wc = wc

                        output = self.get_output_vec(line_items[1])
                        words = dict()
                        for word in cv_tokens :
                            if word in self.wv and word not in stopset:
                                #wordvec = [ (x-self.min)/(self.max-self.min) for x in self.model.wv[word]]
                                find = False
                                for index in range(10) :
                                    if output is not None and output[index]==1 : #and word in self.tf_idf[index] and self.tf_idf[index][word] >= 0.3:
                                        find = True
                                        break
                                if find :
                                    cpt+=1

                        if cpt > max_cpt:
                            max_cpt = cpt
                        if cpt < min_cpt:
                            min_cpt = cpt

                        total_cpt_count += cpt
                        total_wc += wc
                        if i%1000==0 :
                            print("cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} avg_cpt = {:.2f}".format(i,reference, cpt, min_cpt, max_cpt, total_cpt_count*1.0/i))
                            print("cv{:d}#{:s} current_wc = {:d} min_wc = {:d} max_wc = {:d} avg_wc = {:.2f}".format(
                                    i, reference, wc, min_wc, max_wc, total_wc * 1.0 / i))

                        i += 1
            print("cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} avg_cpt = {:.2f}".format(i, reference,
                                                                                                         cpt, min_cpt,
                                                                                                         max_cpt,
                                                                                                         total_cpt_count * 1.0 / i))
            print("cv{:d}#{:s} current_wc = {:d} min_wc = {:d} max_wc = {:d} avg_wc = {:.2f}".format(
                i, reference, wc, min_wc, max_wc, total_wc * 1.0 / i))

            fp.close()

    def wc(self):
        wc_filtered = {}
        wc_non_filtered = {}
        i = 1
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        print("Lines shuffled")
        i = 1
        max_cpt = 0
        test_cpt = 0
        min_cpt = 10000
        total_cpt_count = 0
        references = {}
        class_counters = [0]*10
        negative_decount = [0]*10
        max_wc = 0
        min_wc = 5000
        total_wc = 0
        with open('resume_dataset.csv', 'w', newline='') as fp:
            for line in lines :
                #print("Processing line N0"+str(i+1)+"#.....")
                if line is not "::::::" and not line.isspace():
                    line_items = line.split(":::")
                    if len(line_items) == 3 :
                        #hash = hashlib.md5(line_items[2].encode('utf-8')).hexdigest()
                        reference = line_items[0]
                        #if hash not in self.known_samples :
                        #print("Processing cv... : "+reference+"\n")
                        self.known_samples.update({hash: True})
                        matrix = numpy.empty((0,self.ncol),dtype=numpy.float32)
                        #keys = sorted(self.tokens_in, key=self.tokens_in.get)
                        cpt = 0
                        cv_sents = nltk.sent_tokenize(line_items[2])
                        cv_tokens = [token.lemma_ for sent in  cv_sents for token in nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower()))))]
                        wc = len(cv_tokens)

                        if wc > max_wc :
                            max_wc = wc
                        if wc < min_wc :
                            min_wc = wc

                        wc_non_filtered.update({reference:wc})

                        output = self.get_output_vec(line_items[1])
                        words = dict()
                        for word in cv_tokens :
                            if word in self.wv and word not in stopset:
                                #wordvec = [ (x-self.min)/(self.max-self.min) for x in self.model.wv[word]]
                                find = False
                                for index in range(10) :
                                    if output is not None and output[index]==1 : #and word in self.tf_idf[index] and self.tf_idf[index][word] >= 0.3:
                                        find = True
                                        break
                                if find :
                                    cpt+=1
                        wc_filtered.update({reference: cpt})
                        if cpt > max_cpt:
                            max_cpt = cpt
                        if cpt < min_cpt:
                            min_cpt = cpt

                        total_cpt_count += cpt
                        total_wc += wc
                        if i%1000==0 :
                            print("cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} avg_cpt = {:.2f}".format(i,reference, cpt, min_cpt, max_cpt, total_cpt_count*1.0/i))
                            print("cv{:d}#{:s} current_wc = {:d} min_wc = {:d} max_wc = {:d} avg_wc = {:.2f}".format(
                                    i, reference, wc, min_wc, max_wc, total_wc * 1.0 / i))

                        i += 1
            refs = json.dumps(wc_filtered, indent=4)
            f = open("wc_stop_word_fil.json", "w")
            f.write(refs)
            f.close()
            refs = json.dumps(wc_non_filtered, indent=4)
            f = open("wc_no_fil.json", "w")
            f.write(refs)
            f.close()

            fp.close()
    def transform_data3(self):
        writers = {}
        for k, v in self.tokens_out.items():
            print(k+str(v))
            writers[v] = csv.writer(open(self.output_dir+"/"+k+".csv", 'w', newline=''), delimiter=',')
        i = 1
        test_writer = csv.writer(open(self.output_dir+"/tests.csv", 'w', newline=''), delimiter=',')
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        shuffle(lines)
        print("Lines shuffled")
        i = 1
        max_cpt = 0
        test_cpt = 0
        min_cpt = 10000
        total_cpt_count = 0
        references = {}
        class_counters = [0]*10
        negative_decount = [0]*10
        with open('resume_dataset.csv', 'w', newline='') as fp:
            writer =  csv.writer(fp, delimiter = ',')
            for line in lines :
                #print("Processing line N0"+str(i+1)+"#.....")
                if line is not "::::::" and not line.isspace():
                    line_items = line.split(":::")
                    if len(line_items) == 3 :
                        #hash = hashlib.md5(line_items[2].encode('utf-8')).hexdigest()
                        reference = line_items[0]
                        #if hash not in self.known_samples :
                        #print("Processing cv... : "+reference+"\n")
                        self.known_samples.update({hash: True})
                        matrix = numpy.empty((0,self.ncol),dtype=numpy.float32)
                        #keys = sorted(self.tokens_in, key=self.tokens_in.get)
                        cpt = 0
                        cv_sents = nltk.sent_tokenize(line_items[2])
                        cv_tokens = [token.lemma_ for sent in  cv_sents for token in nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower()))))]
                        shuffle(cv_tokens)
                        cv_tokens = set(cv_tokens)
                        #cv_tokens = set(token.lemma_ for sent in  cv_sents for token in nlp(sent))
                        #shuffle(cv_tokens)
                        output = self.get_output_vec(line_items[1])
                        words = dict()
                        for word in cv_tokens :
                            if word in self.wv and word not in stopset:
                                #wordvec = [ (x-self.min)/(self.max-self.min) for x in self.model.wv[word]]
                                find = False
                                for index in range(10) :
                                    if output is not None and output[index]==1 and word in self.tf_idf[index] and self.tf_idf[index][word] >= 0.3:
                                        find = True
                                        break
                                if find :
                                    vect = self.wv.word_vec(word, use_norm=True)
                                    matrix = numpy.append(matrix,numpy.array([vect],dtype=numpy.float32), axis=0)
                                    words.update({cpt:word})
                                    cpt+=1
                                    if cpt == self.cv_length :
                                        break

                        if cpt > max_cpt:
                            max_cpt = cpt
                        if cpt < min_cpt:
                            min_cpt = cpt

                        for j in range(self.cv_length-cpt) :
                            matrix = numpy.append(matrix,numpy.array([[0]*self.ncol],dtype=numpy.float32), axis=0)

                        if output is not None and cpt > self.min_keywords :
                            matrix = numpy.insert(numpy.append(matrix.reshape((1,self.ncol*self.cv_length)),output),0,i)
                            matrix = matrix.reshape(1,1+len(self.tokens_out)+self.ncol*self.cv_length)
                            if test_cpt > 5000 :
                                for index in writers :
                                    if class_counters[index] < self.max_cv_count :
                                        if output[index] == 1 :
                                            negative_decount[index] += 1
                                            class_counters[index]+=1
                                            #print("index = {:d} in matrix = {:.2f} {:s}".format(index, matrix[0,index - 10], line))
                                            writers[index].writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                            self.resume_words.update({i:words})
                                        else :
                                            if negative_decount[index] > 0 :
                                                writers[index].writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                                negative_decount[index] -= 1
                                                self.resume_words.update({i:words})
                            else :
                                test_writer.writerow(['{:.3f}'.format(x) for x in matrix.flatten()])
                                self.resume_words.update({i: words})
                                test_cpt += 1
                            #self.flatten_matrix = numpy.append(self.flatten_matrix,matrix, axis=0)
                            #self.outputs.append(output)
                            references.update({str(i):reference})
                            total_cpt_count += cpt
                            if i%1000==0 :
                                print("cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} avg_cpt = {:.2f}".format(i,reference, cpt, min_cpt, max_cpt, total_cpt_count*1.0/i))
                            i += 1
                        elif cpt < self.min_keywords :
                            print("-----cv{:d}#{:s} current_cpt = {:d} min_cpt = {:d} max_cpt = {:d} ".format(i, reference, cpt, min_cpt, max_cpt))
            fp.close()
        print(class_counters)
        refs = json.dumps(references,indent=4)
        f = open(self.output_dir+"/"+"resumes_refs.json", "w")
        f.write(refs)
        f.close()
        refs = json.dumps(self.resume_words, indent=4)
        f = open(self.output_dir+"/"+"resumes_words.json", "w")
        f.write(refs)
        f.close()
        #print (self.outputs)
        #header = ['id'] + sorted(self.tokens_in, key=self.tokens_in.get) + sorted(self.tokens_out,key=self.tokens_out.get)
        #header = ",".join(header)
        #numpy.savetxt("resume_dataset.csv", self.flatten_matrix, fmt="%.2f", delimiter=",",header=header)

    def test_pickle_dump(self):
        dataset = pickle.load()
        input, output = dataset
        print (input())
        print (output)

    def test_transform(self,input):
        l = dict()
        l2 = dict()
        l3 = dict()
        for i in range(len(input)):
            if input[i] == 1:
                l.update({i:input[i]})

        print ('\n')
        for k, v in self.tokens_out.iteritems():
            l2.update({v:input[v]})
            if input[v] == 1:
                l3.update({k:v})

        #print l2
        self.saveDic(l2,"data/first_example_bin.txt")
        self.saveDic(l3,"data/first_example.txt")

    def saveDic(self,dic,file_name):
        with open(file_name,"w") as f :
           for k,v in dic.items() :
               f.write(str(k)+"==>"+str(v)+"\n")
           f.close()

    def saveDicAsModel(self,dic,file_name):
        ref = json.dumps(dic)
        with open(file_name,"w") as f :
            f.write(ref)
            f.close()

    def clean_attribute(self):
        unused_attributes = numpy.where(~self.data_matrix.any(axis=0))[0]
        self.data_matrix = numpy.delete(self.data_matrix,unused_attributes,1)
        self.tokens_in = {k:v for k,v in self.tokens_in.iteritems() if v not in unused_attributes}




    
    
    def generate_corpus(self) :
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        i = 0
        for line in lines :
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    hash = hashlib.md5(line_items[2]).hexdigest()
                    if hash not in self.known_samples :
                        with open("data/cv_corpus/"+str(i)+".txt","w") as f :
                            f.write(line_items[2])
                            f.close()
                        i+=1



    def clean(self,tokens):
        excluded = set(string.punctuation)
        excluded_2 = set(['year', 'years', 'etc', "#", "&"])
        import re
        url_pattern = re.compile("^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?")
        output = []
        for token in tokens :
            if token not in excluded and token not in excluded_2 and not url_pattern.match(token):
                for word in tokenizer.tokenize(token) :
                    if len(word) >= 2 :
                        output.append(word)

        return output


    def cv_to_matrix(self):
        corpusdir = 'data/cv_corpus'
        corpa = PlaintextCorpusReader(corpusdir,'.*',encoding='windows-1252')
        print("Preprocessing words....")
        sents = [[token.lemma_ for token in nlp(" ".join(self.clean(sent)).lower()) if token.lemma_ not in stopset] for sent in corpa.sents()]
        print("training word vectors....")
        model = Word2Vec(sents,window=5, size=self.ncol,min_count=1, workers=4)
        fname = get_tmpfile("vectors.kv")
        model.wv.save(fname)
        print("cv_to_matrix model saved")
        return model.wv

    def build_d2v_model(self):
        print("Début de la construction du modèle Doc2Vec")
        corpusdir = 'data/cv_corpus'
        corpa = PlaintextCorpusReader(corpusdir, '.*',encoding='windows-1252')
        print("tokenizing...")
        resumes = [[token.lemma_  for sent in paras for token in nlp(" ".join(self.clean(sent)).lower()) if token.lemma_ not in stopset] for paras in  corpa.paras()]
        #print(resumes[0:3])
        print("tokenization completed")
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(resumes)]
        model = Doc2Vec(documents, vector_size=self.cv_length, window=5, min_count=1, workers=4)
        print("Fin de la construction du modèle Doc2Vec")
        return model

    def transform_data4(self):
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        i = 0
        total = len(lines)
        max_cpt = 0;
        for line in lines :
            #print("Processing line N0"+str(i+1)+"#.....")
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    matrix = numpy.empty((0,self.cv_length),dtype=numpy.float32)
                    cv_sents = nltk.sent_tokenize(line_items[2])
                    doc = [token.lemma_ for sent in cv_sents for token in
                                 nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower()))))]
                    vector = self.doc2VecModel.infer_vector(doc)
                    matrix = numpy.append(matrix, numpy.array([vector], dtype=numpy.float32), axis=0)
                    output = self.get_output_vec(line_items[1])
                    if output is not None :
                        matrix = numpy.insert(numpy.append(matrix.reshape((1,self.cv_length)),output),0,i)
                        matrix = matrix.reshape(1,1+len(self.tokens_out)+self.cv_length)
                        self.flatten_matrix = numpy.append(self.flatten_matrix,matrix, axis=0)
                        #if i==0 :
                        #    print(self.flatten_matrix)
                        self.outputs.append(output)
            if i%1000 == 0 :
                print("Processed lines : "+ str(i)+"/"+str(total))
            i+=1
        #print (self.outputs)
        #header = ['id'] + sorted(self.tokens_in, key=self.tokens_in.get) + sorted(self.tokens_out,key=self.tokens_out.get)
        #header = ",".join(header)
        print(self.flatten_matrix[0:2,:])
        numpy.savetxt("resume_d2v_dataset.csv", self.flatten_matrix, fmt="%.4f", delimiter=",")

    def filter_token_in(self):
        corpusdir = 'data/cv_corpus'
        corpa = PlaintextCorpusReader(corpusdir,'.*',encoding='windows-1252')
        corpa_words = set(token.lemma_ for sent in corpa.sents() for token in nlp(" ".join(sent).lower()) )
        tokens = [t for t in self.tokens_in if t in corpa_words]
        return tokens

    def filter_tokens(self,tokens):
        return [t for t in tokens if t in self.tokens_in]


    def token_in_coverage(self):
        corpusdir = 'data/cv_corpus'
        corpa = PlaintextCorpusReader(corpusdir, '.*',encoding='windows-1252')
        resumes = [[item for sent in paras for item in sent] for paras in corpa.paras()]
        cpt=0
        for resume in resumes :
            resume_text = " ".join(resume)
            resume_sents = nltk.sent_tokenize(resume_text)
            resume_words = set(token.lemma_ for sent in resume_sents for token in nlp(" ".join(sent).lower()))
            if not resume_words.isdisjoint(self.tokens_in) :
                cpt+=1
        coverage = cpt*1.0/len(resumes)
        print("token_in coverage : {}".format(coverage))

    def get_min_max(self):
        min = self.wv["computer"][0]
        max = self.wv["computer"][0]
        for word in self.tokens_in :
            if word in self.wv :
                for i in range(self.ncol) :
                    if self.wv[word][i] > max :
                         max = self.wv[word][i]
                    if self.wv[word][i] < min :
                         min = self.wv[word][i]
        print (max)
        print (min)
        return min,max


    def transform_classes(self):
        pass


    def generate_train_data(self):
        pass


    def clean_classes(self):
        pass


    def export_data(self):
        pass

    def loadDicModel(self, file):
        with open(file) as json_file:
            return json.load(json_file)


    def computeTFIDF(self):
        print("Calculating tf_idf....")
        tf_idf = [0]*10
        df = {}
        tf = [0]*10
        dc = [0]*10
        for i in range(10) :
            tf_idf[i] = dict()
            tf[i] = dict()
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        td = 0
        cpt = 1
        for line in lines :
            #print("Processing line N0"+str(i+1)+"#.....")
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    new_doc = True
                    td += 1
                    output = self.get_output_vec(line_items[1])
                    if output is None :
                        continue
                    for index in range(10):
                        if output[index] == 1:
                            dc[index] += 1

                    #cv_tokens = tokenizer.tokenize(line_items[2])
                    cv_sents = nltk.sent_tokenize(line_items[2])
                    cv_tokens = set(token.lemma_ for sent in cv_sents for token in
                                    nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower())))))
                    for token in cv_tokens :
                        if token not in stopset :
                            for index in range(10):
                                if output[index] == 1 :
                                    tf[index][token]= tf[index].get(token,0) + 1
                            df[token] = df.get(token, 0) + 1
                    if cpt%1000 == 0 :
                        print("cv {:d}/{:d} processed".format(cpt,len(lines)))
            cpt+=1

        print("Finilizing TF-IDF computing")
        for index in range(10) :
            n = 0
            for k,v in tf[index].items() :
                wtf = v*1.0/dc[index]
                widf = math.log((td*1.0+1)/(df[k]+1))
                wtf_idf = wtf*widf
                tf_idf[index].update({k:wtf_idf})
        refs = json.dumps(tf_idf,indent=4)
        f = open("tf_idf.json", "w")
        f.write(refs)
        f.close()
        print("tf_idf computed")
        return tf_idf


    def computeTF(self):
        print("Calculating tf_idf....")
        tf_idf = [0]*10
        df = {}
        tf = [0]*10
        dc = [0]*10
        for i in range(10) :
            tf_idf[i] = dict()
            tf[i] = dict()
        self.data_x_file.seek(0)
        lines = self.data_x_file.read().splitlines()
        td = 0
        cpt = 1
        for line in lines :
            #print("Processing line N0"+str(i+1)+"#.....")
            if line is not "::::::" and not line.isspace():
                line_items = line.split(":::")
                if len(line_items) == 3 :
                    new_doc = True
                    td += 1
                    output = self.get_output_vec(line_items[1])
                    if output is None :
                        continue
                    for index in range(10):
                        if output[index] == 1:
                            dc[index] += 1

                    #cv_tokens = tokenizer.tokenize(line_items[2])
                    cv_sents = nltk.sent_tokenize(line_items[2])
                    cv_tokens = set(token.lemma_ for sent in cv_sents for token in
                                    nlp(" ".join(self.clean(nltk.tokenize.word_tokenize(sent.lower())))))
                    for token in cv_tokens :
                        if token not in stopset :
                            for index in range(10):
                                if output[index] == 1 :
                                    tf[index][token]= tf[index].get(token,0) + 1
                            df[token] = df.get(token, 0) + 1
                    if cpt%1000 == 0 :
                        print("cv {:d}/{:d} processed".format(cpt,len(lines)))
            cpt+=1

        f = open("words_freq.json", "w")
        refs = json.dumps(tf, indent=4)
        f.write(refs)
        f.close()
        print("tf computed")
        """
        print("Finilizing TF-IDF computing")
        for index in range(10) :
            n = 0
            for k,v in tf[index].items() :
                wtf = v*1.0/dc[index]
                widf = math.log((td*1.0+1)/(df[k]+1))
                wtf_idf = wtf*widf
                tf_idf[index].update({k:wtf_idf})
        refs = json.dumps(tf_idf,indent=4)
        f = open("words_freq.json", "w")
        f.write(refs)
        f.close()
        print("tf_idf computed")
        return tf_idf
        """




