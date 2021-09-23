#
# The purpose of this program is to generate dataset for query focused abstractive summary from Reddit dataset
# Input: Dataset from Reddit
# Output: Dataset for query focused abstractive/mixed summary
# The script is based on 'preprocessNewsroom.py '
#

import gc
import glob
import re
import os
import sys
import shutil
import subprocess
import json
from types import new_class
import torch
import random
import math
import shutil
from os.path import join as pjoin
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from others.utils import clean
from others.tokenization import BertTokenizer
from others.logging import logger
from prepro.utils import _get_word_ngrams
from orderingQuery import orderingQuery

# Purpose: to count the number of sentence in summary
# Parameter:  Input -- summary as string
#   			    -- string -- str
#			  Output --
# Return: number of sentence
def numberOfSentence (str1):
	return len(sent_tokenize(str1))

# Purpose: to make the title as list of words
# Parameter:  Input -- title as string
#   			    -- string -- str
#			  Output --
# Return: cleaned title as list of words
def cleanQuery (str1):
	stemmer = WordNetLemmatizer()
	stop_words = stopwords.words('english')
	# Remove all the special characters
	document = re.sub(r'\W', ' ', str(str1))
	# remove all single characters
	document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
	# Remove single characters from the start
	document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
	# Substituting multiple spaces with single space
	document = re.sub(r'\s+', ' ', document, flags=re.I)
	# Removing prefixed 'b'
	document = re.sub(r'^b\s+', '', document)
	# Converting to Lowercase
	document = document.lower()
	# Lemmatization
	document = document.split()
	document = [stemmer.lemmatize(word) for word in document]
	# remove stop words
	document = [w for w in document if not w in stop_words]

	str1= ""
	for ele in document:
		str1 = str1 + ' ' + ele

	return str1[1:len(str1)]

# Purpose: to write query, summary and document in coresponding directory
# Parameter:  Input -- information of data
#				    -- dictionary -- entry
#  				    -- int -- entryNo
#   			    -- str -- type ( "query" or "document" or "summary" )
#			  Output --
# Return:
def writeFile (entry, entryNo, targetFile, types):
	for type in types:
		directory = os.path.join(src_dir, "../dataset/"+targetFile+"/"+type)
		if not os.path.exists(directory):
			os.makedirs(directory)
		filepath = directory+"/"+ str(entryNo) +".txt"
		if os.path.isfile(filepath):
			break
		else:
			fquery = open(filepath, "w")
			if (type == "query"):
				query = cleanQuery(entry['title'])
				fquery.write(query)
			elif (type == "summary"):
				fquery.write(entry["summary"])
			elif (type == "document"):
				fquery.write(entry["content"])
			fquery.close()
			print ("Writing done on "+directory+"/"+ str(entryNo) +".txt")

# Purpose: called from formatToLines (to create source & target)
# Parameter: corpus_type, fileNumber
# Return: source, tgt
def load_json(corpus_type, fileNumber):
    source = []
    tgt = []
    p = os.path.join(src_dir,"../merged_stories_tokenized/"+corpus_type+"/")
    q = str(fileNumber)+".txt.json"
    for sent in json.load(open(p+"document/"+q))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        tokens = [t.lower() for t in tokens]
        source.append(tokens)
    for sent in json.load(open(p+"summary/"+q))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        tokens = [t.lower() for t in tokens]
        tgt.append(tokens)

    print("Source and Target are created for the corpus type: "+ corpus_type + ", File: "+q)
    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    source, query, tgt, sorting_status = orderingQuery (source, tgt, extractive = False)
    return {'src': source, 'qry': query, 'tgt': tgt, 'status': sorting_status}

# Purpose: Format to Simple Json Files
# Parameter: targetFiles -> {'train', 'valid', 'test'}
# Return:
def formatToLines(targetFiles):
	if not os.path.exists(os.path.join(src_dir,"../json_data")):
		os.makedirs(os.path.join(src_dir,"../json_data"))

	# change it to : 'train', 'valid', 'test'
	for corpus_type in targetFiles:
		dataset = []
		p_ct = 0
		numberOfFiles = len(os.listdir(os.path.join(src_dir,"../merged_stories_tokenized/"+corpus_type+"/summary")))
		for fileNumber in range (1, numberOfFiles):
			d = load_json(corpus_type, fileNumber)
			dataset.append(d)
			if (len(dataset) > 2000):
				pt_file = "{:s}.{:s}.{:d}.json".format("../json_data/reddit", corpus_type, p_ct)
				with open(pt_file, 'w') as save:
					save.write(json.dumps(dataset))
					p_ct += 1
					dataset = []
		if (len(dataset) > 0):
				pt_file = "{:s}.{:s}.{:d}.json".format("../json_data/reddit", corpus_type, p_ct)
				with open(pt_file, 'w') as save:
					save.write(json.dumps(dataset))
					p_ct += 1
					dataset = []
	#shutil.rmtree("tokenized")

# Purpose: to tokenize the Reddit dataset using StanfordCoreNLP
# Parameter:  Input -- targetFiles -> {'train', 'valid', 'test'}
#			  Output --
# Return:
def tokenize(targetFiles):
	'''
	targetFiles = {'test','valid','train'}  # "test" or "train" or "valid"
	types = {'document', 'query' , 'summary'}
	'''
	types = {'document', 'summary'}
	for targetFile in targetFiles:
		for type in types:
			directory = os.listdir(os.path.join(src_dir,"../dataset/"+targetFile+"/"+type))
			tokenized_dir = os.path.join(src_dir,"../merged_stories_tokenized/"+targetFile+"/"+type)
			if not os.path.exists(tokenized_dir):
				os.makedirs(tokenized_dir)
			#interrupt if files already exist
			elif os.path.exists(tokenized_dir) and len(os.listdir(tokenized_dir))!=0:
				continue
			with open("mapping_for_corenlp.txt", "w") as f:
				for d in directory:
					if (not d.endswith('txt')):
						continue
					f.write("%s\n" % (os.path.join("../dataset/"+targetFile+"/"+type, d)))
			command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_dir]
			subprocess.call(command)
			#os.remove("mapping_for_corenlp.txt")
	#shutil.rmtree("dataset")

# This class is used by formatToBert function
# taken from https://github.com/nlpyang/PreSumm
class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.min_src_ntokens_per_sent = 5
        self.max_src_ntokens_per_sent = 200
        self.max_src_nsents = 100
        self.min_src_nsents = 3
        self.max_tgt_ntokens = 500
        self.min_tgt_ntokens = 5 

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.max_src_nsents]
        sent_labels = sent_labels[:self.max_src_nsents]

        if ((not is_test) and len(src) < self.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

# taken from https://github.com/nlpyang/PreSumm
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

# taken from https://github.com/nlpyang/PreSumm
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

# Purpose: to make the REDDIT dataset in Binary form
# Parameter:  Input --
#			  Output --
# Return:
def formatToBert(targetFiles):
	if not os.path.exists(os.path.join(src_dir,"../bert_data")):
		os.makedirs(os.path.join(src_dir,"../bert_data"))
	max_src_nsents = 100
	for corpus_type in targetFiles:
		is_test = corpus_type == 'test'
		bert = BertData()

		for json_f in glob.glob(pjoin(src_dir, "../json_data", '*' + corpus_type + '.*.json')):
			jobs = json.load(open(json_f))
			real_name = json_f.split('/')[-1]
			save_file = pjoin(src_dir, "../bert_data", real_name.replace('json', 'bert.pt'))
			datasets = []
			for d in jobs:
				source, tgt = d['src'], d['tgt']
				sent_labels = greedy_selection(source[:max_src_nsents], tgt, 3)
				source = [' '.join(s).lower().split() for s in source]
				tgt = [' '.join(s).lower().split() for s in tgt]
				b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer = True, is_test=is_test)
				if (b_data is None):
					continue
				src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
				b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids, 'src_txt': src_txt, "tgt_txt": tgt_txt}
				datasets.append(b_data_dict)
			torch.save(datasets, save_file)
			print("Writing on bert_data, corpus_type: ", corpus_type, ", Save location: ", save_file)
			datasets = []
			gc.collect()
	#shutil.rmtree("json_data")

def split_data(data):
	train_share = 0.8
	validation_share = 0.1
	print(f'Share of training data: {train_share}\nShare of validation data: {validation_share}\nShare of test data: {1-train_share-validation_share}')
	data_dict = {}
	random.shuffle(data)
	data_dict['train'] = data[:math.floor(train_share*len(data))]
	data_dict['valid'] = data[math.floor(train_share*len(data)):math.floor((train_share + validation_share)*len(data))]
	data_dict['test'] = data[math.floor((train_share + validation_share)*len(data)):]
	#print(len(data_dict['train']),len(data_dict['valid']),len(data_dict['test']))
	return data_dict

# Purpose: to generate query focused abstractive summary from REDDIT dataset
# adapted from preprocessNewsroom.py
# Parameter:  Input --
#				    --
#			  Output --
# Return:
if __name__ == "__main__":
	# working directory
	os.chdir(os.path.dirname(__file__))
	print(os.getcwd())
	
	#replace /path/to/ with the path to where you saved the stanford-corenlp-full-2018-10-05 directory.
	#os.environ['CLASSPATH'] = "/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar"
	#os.system("echo $CLASSPATH")

	src_dir = os.path.dirname(__file__)
	raw_dir = os.path.join(src_dir, "../raw_stories/")
	#data format: several json objects per file, newline characters inside the json objects
	data_raw = os.path.join(raw_dir, "corpus-webis-tldr-17.json")
	data_filtered = os.path.join(raw_dir, "corpus-webis-tldr-17-filtered.json")

	#create empty directories
	new_dirs = True
	if new_dirs == True:
		shutil.rmtree( "../bert_data")
		shutil.rmtree(os.path.join(src_dir, "../json_data"))
		shutil.rmtree(os.path.join(src_dir, "../logs"))
		shutil.rmtree(os.path.join(src_dir, "../merged_stories_tokenized"))
		shutil.rmtree(os.path.join(src_dir, "../urls"))
		shutil.rmtree(os.path.join(src_dir, "../dataset"))

		os.makedirs(os.path.join(src_dir, "../bert_data"))
		os.makedirs(os.path.join(src_dir, "../json_data"))
		os.makedirs(os.path.join(src_dir, "../logs"))
		os.makedirs(os.path.join(src_dir, "../merged_stories_tokenized"))
		os.makedirs(os.path.join(src_dir, "../urls"))
		os.makedirs(os.path.join(src_dir, "../dataset"))

	targetFiles = {'test', 'train', 'valid'}

	#os.remove(data_filtered)
	with open(data_raw, 'r') as infile, open(data_filtered, 'a') as outfile:
		count=0
		data = [json.loads(line) for line in infile]
		for entry in data:
			if count >=2000:
				print(entry)
				break
			elif len(entry['summary'].split())>=24 and len(entry['summary'].split())<=48:
				outfile.write(json.dumps(entry))
				outfile.write('\n')
				count+=1

	with open(data_filtered, 'r') as file:
		data = [json.loads(line) for line in file]
		print(len(data))
		data_dict = split_data(data)
		for targetFile in targetFiles:
			print(targetFile)
			count = 0
			for entry in data_dict[targetFile]:
				writeFile(entry, count, targetFile, {'document', 'summary'}) # not called for 'query'
				count+=1
	tokenize(targetFiles)
	formatToLines(targetFiles)
	formatToBert(targetFiles)