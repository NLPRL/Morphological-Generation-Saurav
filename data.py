from collections import Counter, defaultdict
import codecs

import numpy as np

class DatasetRow:
    # One row of training data
    def __init__(self, root_word, morphed_word, morphed_word_tags,morphed_word_tags_original):

        self.root_word = root_word                
        self.morphed_word = morphed_word               
        self.morphed_word_tags = morphed_word_tags
        self.morphed_word_tags_original = morphed_word_tags_original
        self.both_words = [root_word,morphed_word]    
        self.morphed_word_tags_dict = DatasetRow.get_tag_dict(morphed_word_tags)    

    def __unicode__(self):
    
        ret = u'\t'.join(
                ele for ele in (self.root_word, self.morphed_word, self.morphed_word_tags_original)
                )
        # print 'ret_type:', type(ret)
        return ret

    def recapitalize(self, s):
        if not s: return s
        elif self.source.islower(): return s
        elif self.source.isupper() and len(self.source) > 1: return s.upper()
        else: return s[0].upper() + s[1:]


    @staticmethod
    def get_tag_dict(morphed_word_tags):
        # Eg: feats= 'pos=V,mood=IND,tense=PRS,per=3,num=PL,aspect=IPFV/PFV'
        # returns {'mood': 'IND', 'pos': 'V', 'per': '3', 'num': 'PL', 'tense': 'PRS', 'aspect': 'IPFV/PFV'}
        return dict(pair.split('=') for pair in morphed_word_tags.split(','))

    @staticmethod
    def add_morphed_word_tags_categories(morphed_word_tags):
    
        tags = morphed_word_tags.split(';')
        tags_new = []
        for tag in tags:
            tags_new.append(str(tag.lower()+'='+tag))   
        tag_string_new = ','.join(tags_new)
        # print('Hey2',feat_string_new)
        return tag_string_new

class Dataset:
    def __init__(self, dataset_directory, language, train_size):

        
        # self.data is a list:
        # [
        # {'train': Wordforms for (1, 'train'), 'test-covered': Wordforms for (1, 'test-covered'), 'dev': (1, 'dev')}, --> task1's dict
        # {'train': (2, 'train'), 'test-covered': (2, 'test-covered'), 'dev': (2, 'dev')}, --> task2's dict
        # {'train': (3, 'train'), 'test-covered': (3, 'test-covered'), 'dev': (3, 'dev')}  --> task3's dict
        # ]

        # CHANGED for task in (1,2,3)] ---> for task in (1)]
        # [{'dev': (1, 'dev'), 'train': (1, 'train-high')}]
        def read_data_from_file(dataset_directory,language,mode,train_size):
            if mode == 'train':
                data_file = '%s%s-%s-%s' % (dataset_directory, language, mode, train_size)
            else:
                data_file = '%s%s-%s' % (dataset_directory, language, mode)
            with codecs.open(data_file, 'r', encoding='utf-8') as f:
                data_set = []
                # Rows of training data are list items, each stored as a WordForm object
                for line in f:
                    row = line.strip().split('\t')
                    tags = DatasetRow.add_morphed_word_tags_categories(row[2]).decode('utf-8')
                    row_object = DatasetRow(root_word=row[0], morphed_word=row[1], morphed_word_tags=tags, morphed_word_tags_original=row[2])
                    data_set.append(row_object)
                return data_set

        self.training_dataset = read_data_from_file(dataset_directory, language, 'train', train_size)
        # print self.training_dataset
        self.dev_dataset = read_data_from_file(dataset_directory, language, 'dev', None)
        # print self.dev_dataset
        self.full_dataset = [self.training_dataset,self.dev_dataset]
        # print self.full_dataset
        def populate_character_set(full_dataset):
            character_set = set([])
            for dataset in full_dataset:
                for data_row in dataset:
                    for letter in data_row.root_word:
                        character_set.add(letter)
                    for letter in data_row.morphed_word:
                        character_set.add(letter)
            return character_set
        # Set of all alphabets used in the words and their inflected forms in the train, dev and test data across all tasks (the whole of self.data)
        # Eg: ['a', 'b', 'd', 'e', 'l', 'o', 'p', 'r', 's', 't']
        self.character_set = sorted(populate_character_set(self.full_dataset)) 
        # for i in self.character_set:
        #     print i

        def populate_tags_character_set(full_dataset):
            tags_character_set = set([])
            for dataset in full_dataset:
                for data_row in dataset:
                    for letter in data_row.morphed_word_tags:
                        tags_character_set.add(letter)
            return tags_character_set       

        # Set of all characters used in the feature tags in the train, dev and test data across all tasks (the whole of self.data)
        # Eg: [',', '/', '3', '=', 'a', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']
        self.tags_character_set = sorted(populate_tags_character_set(self.full_dataset))
        # for i in self.tags_character_set:
        #     print i

        # Dictionary to store feature character --> index mapping
        # Eg: feature_idx['/'] = 1
        self.tags_character_to_index = {char: index for index,char in enumerate(self.tags_character_set)}
        # print self.tags_character_to_index
        # Max length of any word form
        self.word_len_max = max(
            len(word) for dataset in self.full_dataset
                      for data_row in dataset
                      for word in data_row.both_words)
        # print self.word_len_max

        # Note: the Morphon model will add padding and beginning/end of
        # sentence symbols, so don't use this map
        #self.alphabet_idx = {c: i for i,c in enumerate(self.alphabet)}

        # feature_values = { feature_name1:set of values, feature_name2: set of values..}
        # Eg: {pos:[adj,noun,verb], num:[sg,pl], ..}
        
        # Eg total feature_dicts across all wfs=  [{'num': 'PL', 'pos': 'ADJ', 'gen': 'MASC'}, {'num': 'PL', 'pos': 'ADJ', 'gen': 'FEM'}]
        def populate_all_tag_values(full_dataset):
            all_tag_values = defaultdict(set)
            for dataset in full_dataset:
                for data_row in dataset:
                    for tag,value in data_row.morphed_word_tags_dict.items():
                            all_tag_values[tag].add(value)

            all_tag_values = sorted([
                [tag, sorted(values)]
                for tag, values in all_tag_values.items()])
            return all_tag_values
        
        self.all_tag_values = populate_all_tag_values(self.full_dataset)     
        # feature_values = defaultdict(<type 'set'>, {'num': set(['PL']), 'pos': set(['ADJ']), 'gen': set(['FEM', 'MASC'])})
        # feature_values = ['gen', ['FEM', 'MASC']], ['num', ['PL']], ['pos', ['ADJ']]]
        # print self.all_tag_values

        self.all_tag_values_to_index  = {
                tag: {value: index for index,value in enumerate(values)}
                for tag, values in self.all_tag_values}
        # feature_values_idx={'num': {'PL': 0}, 'gen': {'FEM': 0, 'MASC': 1}, 'pos': {'ADJ': 0}}
        # print self.all_tag_values_to_index

        self.all_tags_to_index = {
                tag: index for index,(tag,_) in enumerate(self.all_tag_values)}
        # feature_idx = {'num': 1, 'gen': 0, 'pos': 2}
        # print self.all_tags_to_index

        self.morphed_tags_vec_len = sum(
                len(values) for _,values in self.all_tag_values)
        # print self.morphed_tags_vec_len
        # feature_vector_length  = 4 because, 'pl','fem', 'masc', 'adj' (total number of UNIQUE feature values)

    def get_tag_vector(self, morphed_word_tags_dict):
        return np.array(
                [morphed_word_tags_dict.get(tag) == value
                 for tag,values in self.all_tag_values
                 for value in values],
                dtype=np.int8)

