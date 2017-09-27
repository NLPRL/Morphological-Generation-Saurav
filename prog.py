"""Entry for SIGMORPHON 2016 shared task"""
import io
import codecs
import os
import pickle
import time
import copy
import sys
import pydot
import numpy as np

from keras.models import Model, model_from_yaml
from keras.layers import (Embedding, Input, GRU, Activation, TimeDistributed,
						  Dense, merge, RepeatVector, LSTM, Convolution1D,
						  BatchNormalization, SimpleRNN)
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model


# class FlushCallback(Callback):
# 	def on_epoch_end(self, epoch, logs={}):
# 		print('Epoch %s: loss %s' % (epoch, logs.get('loss')))


from data import Dataset

EXTRA_LENGTH = 4
NUM_SPECIAL_CHARS = 2

class KerasModel:
	def __init__(self, dataset, model_common_name):
		self.dataset = dataset           # MorphonData instance
		self.model_common_name = model_common_name        # path of model files
		self.word_length = dataset.word_len_max + EXTRA_LENGTH
		self.character_set = ['<zero_pad>', '<s>'] + dataset.character_set
		self.start_index_of_alphabets = NUM_SPECIAL_CHARS
		# Each char gets an index, including special symbols 
		self.character_to_index = {char:index for index,char in enumerate(self.character_set)}

	def set_configurations_model(self,optional_LSTMs):
		def smart_merge(vectors, **kwargs):
			return vectors[0] if len(vectors)==1 else merge(vectors, **kwargs)
		dim_root_word_in = self.word_length
		dim_morphed_word_in = self.word_length
		number_of_repeats = self.word_length
		dim_tags_in = self.dataset.morphed_tags_vec_len 
		dim_embedding = len(self.character_set)
		dim_time_dist_layer = len(self.character_set)
			
		root_word_in = Input(shape=(dim_root_word_in,), dtype='int32')
		# print root_word_in
		
		morphed_word_in = Input(shape=(dim_morphed_word_in,), dtype='int32')
		# print morphed_word_in
		tags_in = Input(shape=(dim_tags_in,), dtype='float32')
		# print tags_in
		
		
		tags_and_optional_LSTMs = [tags_in]
		
		root_word_embedding = Embedding(dim_embedding, 64,
							   input_length=dim_root_word_in,
							   W_constraint=maxnorm(2))(root_word_in)
		
		morphed_word_embedding = Embedding(dim_embedding, 64,
							   input_length=dim_morphed_word_in,
							   W_constraint=maxnorm(2))(morphed_word_in)	
		
		if optional_LSTMs: 
			tags_and_optional_LSTMs.extend([
				LSTM(256, return_sequences=False)(root_word_embedding),
				LSTM(256, return_sequences=False, go_backwards=True)(root_word_embedding)])

		tags_and_optional_LSTMs_layer= smart_merge(tags_and_optional_LSTMs, mode='concat')

		embeddings_and_repeats = smart_merge(
			[root_word_embedding,
			 morphed_word_embedding,
			 RepeatVector(number_of_repeats)(tags_and_optional_LSTMs_layer)],
			mode='concat')



		LSTM_layer = LSTM(256, return_sequences=True,
						 dropout_W=0)(embeddings_and_repeats)
		batch_norm_layer = BatchNormalization()(LSTM_layer)

		time_dist_layer = TimeDistributed(Dense(dim_time_dist_layer))(batch_norm_layer)
		outputs = Activation('softmax')(time_dist_layer)

		all_inputs = [root_word_in, tags_in, morphed_word_in]
		self.model = Model(input=all_inputs, output=outputs)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy')


	def retrieve_model_weights(self,model_common_name):
		self.model.load_weights(model_common_name + '.hdf5')


	def get_string_to_num_repr(self, string, tot_length):
		return np.array(  [self.character_to_index[char] for char in string.lower()]
							+ [0]*(tot_length-len(string)),
							dtype=np.int8)


	def output_test_file(self, f, mode):
		# TODO: ensemble
		# retrieves dev'
		if mode == 'dev':
			data_rows = self.dataset.dev_dataset
		elif mode == 'test':
			pass 

		beam_size = 5
		beam_set = [(np.zeros((len(data_rows), self.word_length),
				 dtype=np.int32),
				 np.zeros((len(data_rows),), dtype=np.float32))]
		
		root_word_and_tags_set = [
				
				np.array([self.get_string_to_num_repr(data_row.root_word, self.word_length)
						  for data_row in data_rows],
						 dtype=np.int8),
				
				np.array([self.dataset.get_tag_vector(data_row.morphed_word_tags_dict)
						  for data_row in data_rows])]

		for char_pos in range(self.word_length):
			print('Beam search %d/%d' % (char_pos+1, self.word_length))
			new_candidates=[]
			for candidate_string,log_score in beam_set:
				 
				shifted = np.hstack([np.full((len(data_rows), 1), 1,dtype=np.int32), candidate_string[:,:-1]])
				a = self.model.predict(root_word_and_tags_set + [shifted])
				
				c=a[:,char_pos,:]
				
				b = np.log(c)
				
				new_candidates.append(b + log_score[:,None])
				
			next_beam_set_beams = []
			for i in range(len(data_rows)):
				neighbourhood_log_score = {}
				for beam_num,log_scores in enumerate(new_candidates):
					for c,log_score in enumerate(log_scores[i,:]):
						
						neighbourhood =   tuple(beam_set[beam_num][0][i,:char_pos].tolist()) \
								  + (c,) \
								  + tuple(beam_set[beam_num][0][i,char_pos+1:].tolist())
						
						neighbourhood_log_score[neighbourhood] = max(
								log_score, neighbourhood_log_score.get(neighbourhood, -1e10))
				choices = sorted(neighbourhood_log_score.items(),
								 key=lambda t: -t[1])[:beam_size]
				
				assert len(choices) == beam_size
				next_beam_set_beams.append(list(zip(*choices)))
			
			beam_set = [(
				np.array([neighbourhood[beam_num] for neighbourhood, _ in next_beam_set_beams],
						 dtype=np.int32),
				np.array([log_scores[beam_num] for _, log_scores in next_beam_set_beams],
						 dtype=np.float32))
				for beam_num in range(beam_size)]

		predicted_strings=[]

		for data_row, pred_string_vec in zip(data_rows, beam_set[0][0]):
			# data_row_unicode = data_row.__unicode__().split('\t')
			# root_word = data_row_unicode[0]
			
			root_word = data_row.root_word
			pred_string= ''.join(self.character_set[index]
					   for index in pred_string_vec if index >= NUM_SPECIAL_CHARS) 

			if root_word.isupper(): 
				pred_string_new = pred_string.upper()
			elif root_word[0].isupper(): 
				pred_string_new = pred_string[0].upper() + pred_string[1:]
			else:
				pred_string_new = pred_string
			predicted_strings.append(pred_string_new)

					
		for data_row, pred_string in zip(data_rows, predicted_strings):
			# print 'wf, s\n', unicode(wf), s

			ret = u'\t'.join(
				ele for ele in (data_row.root_word, pred_string, data_row.morphed_word_tags_original)
				)
			
			f.write(ret+'\n')
			


	def calculate_errors(self, mode, f):
		if mode == 'dev':
			data_rows = self.dataset.dev_dataset
		elif mode == 'test':
			pass
		predicted_strings = []
		for line in f:
			pred = line.strip(u'\n').split(u'\t')[1]
			predicted_strings.append(pred)
		wrong = [(pred, data_row.morphed_word, data_row.root_word, data_row.morphed_word_tags)
					 for data_row, pred in zip(data_rows, predicted_strings)
					 if data_row.morphed_word != pred]
		accuracy = (float(len(data_rows)) - float(len(wrong))) / float(len(data_rows))
		
		print('Accuracy: %.3f%%' % (100.0*accuracy))
		print('%d/%d incorrect classifications:' % (len(wrong), len(data_rows)))
		for t in wrong:
			print('  %-20s %-20s (%s, %s)' % t)
		print('-'*72)

		return accuracy



	def fit_model(self,model_common_name):
		
		root_word_and_tags_set = [
				
				np.array([self.get_string_to_num_repr(data_row.root_word, self.word_length)
						  for data_row in self.dataset.training_dataset],
						 dtype=np.int8),
				
				np.array([self.dataset.get_tag_vector(data_row.morphed_word_tags_dict)
						  for data_row in self.dataset.training_dataset])]

		morphed_word_set = np.array([self.get_string_to_num_repr(data_row.morphed_word, self.word_length)
						 for data_row in self.dataset.training_dataset],
						 dtype=np.int8)
		
		# morphed_word_set_one_hot = self.onehot(morphed_word_set)
		morphed_word_set_one_hot = np.zeros(morphed_word_set.shape+(len(self.character_set),), dtype=np.int8)
		# print(x.shape+(length,))
		for i,j in enumerate(morphed_word_set):
			# print(i,row)
			for k,l in enumerate(j):
				# print(j,col)
				morphed_word_set_one_hot[i,k,l] = 1
		
		morphed_word_set_shifted = np.hstack([
			np.full_like(morphed_word_set[:,0:1], 1),
			morphed_word_set[:,:-1]])
			
		fit_in_1 = root_word_and_tags_set + [morphed_word_set_shifted]
		fit_in_2 = morphed_word_set_one_hot
		# print fit_in_1
		# print '------'
		# print fit_in_2
		self.model.fit(
				fit_in_1, fit_in_2,
				batch_size=100,
				nb_epoch= 1000,
				verbose= 2,
				validation_split= 0.1,
				callbacks=[EarlyStopping(patience=10),
						   ModelCheckpoint(model_common_name + '.hdf5',
										   save_best_only=True,
										   verbose=1)])



def train(model_common_name, dataset_directory, language, train_size):
	
	
	dataset = Dataset(dataset_directory, language, train_size)		
	keras_model = KerasModel(dataset, model_common_name)
	
	if train_size in ['medium','high']:
		optional_LSTMs = True
	else:
		optional_LSTMs = False
	keras_model.set_configurations_model(optional_LSTMs)

	keras_model.fit_model(model_common_name)
	
	keras_model.retrieve_model_weights(model_common_name)
		
	# NO TEST DATA YET
	# if not os.path.exists(prefix + '-solution'):
	#     # print('Annotating test set...', flush=True)
	#     with open(prefix + '-solution', 'w', encoding='utf-8') as f:
	#         morphon.write_test(f, 'test-covered')
	if train_size == 'high':
		plot_model(keras_model.model, show_shapes = True, to_file='model_high.png')
	elif train_size == 'medium':
		plot_model(keras_model.model, show_shapes = True, to_file='model_med.png')
	elif train_size == 'low':
		plot_model(keras_model.model, show_shapes = True, to_file='model_low.png')	
	print('Annotating development set...')
	with codecs.open(model_common_name + '-dev', 'w', encoding='utf-8') as f:
		keras_model.output_test_file(f, 'dev')

	print('Development set evaluation (%s)\n' % model_common_name)
	with codecs.open(model_common_name + '-dev', 'r', encoding='utf-8') as f:
		return keras_model.calculate_errors('dev', f)


def task_run():

	dataset_directory = 'datasets/'
	base_config = {
			  'use_bn': True,
			  'use_encoder': False,
			  # 'dropout': 0.5,
			  'n_layers': 1,
			  'n_conv_layers': 0,
			  'gate_name': 'lstm',
			  'hidden_dims': 256,
			  'embedding_dims': 64,
			  'share_embeddings': False }

	

	saved_models_directory = './keras_saved_models/'
	languages = ['hindi']

	with codecs.open(saved_models_directory + 'execution.log', 'a', encoding='utf-8') as execution_log:
		for language in languages:
			for train_size in ['medium']:
				model_for_language = saved_models_directory + 'language=%s--train_size=%s' % (language,train_size) 
				t_start = time.time()
				accuracy = train(model_for_language,dataset_directory,language,train_size)
				run_time = time.time() - t_start
				execution_log.write('Accuracy: %.3f, language: %s, training size:%s\n' % (100.0*accuracy, language, train_size))
				# x = raw_input('Finished %s' % train_size)

if __name__ == '__main__':
	# print(int(sys.argv[1]))
	# print([x for x in sys.argv[2:] if x != 'TESTING'])
	task_run()


