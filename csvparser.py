import numpy as np
import tensorflow as tf
import math as math
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#data
test_data = "ml_test_data.csv"
training_data = "ml_training_data.csv"

#Just to provide arguments for program start up. https://docs.python.org/3/howto/argparse.html here for more info
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()

#file length for the CSV file
def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):  #enumerate(iterable, start=0)
			pass
	return i + 1

#read from csv for tensorflow
def read_from_csv(filename_queue):
	reader = tf.TextLineReader(skip_header_lines = 1)	#skip the header
	_, csv_row = reader.read(filename_queue)
	record_defaults = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
	CUSTOMER, DISTRIBUTOR, CONTRACT_NUMBER, TRANSACTION_TYPE, RAW_NAME1, RAW_NAME2, RAW_NAME3, RAW_ADDRESS1, RAW_ADDRESS2, RAW_CITY, RAW_STATE, RAW_ZIPCODE, TARGET_LOC_ID = tf.decode_csv(csv_row, record_defaults = record_defaults)
	feature = tf.stack([RAW_NAME1, RAW_NAME2, RAW_NAME3, RAW_ADDRESS1, RAW_ADDRESS2, RAW_STATE, RAW_CITY, RAW_ZIPCODE])
	label = tf.stack([TARGET_LOC_ID])	#return a bytes object containing the values, packed according to the format string fmt.
	return feature, label

#input_pipeline https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/train/shuffle_batch for more info
def input_pipeline(batch_size, num_epochs = None):
	filename_queue = tf.train.string_input_producer(training_data, num_epochs = num_epochs, shuffle = True)
	example, label = read_from_csv(filename_queue)
	min_after_dequeue = 10000			#tensorflow example default number
	capacity = min_after_dequeue + 3 * batch_size 					#tensorflow example default number
	example_batch, label_batch = tf.train.shuffle_batch([example,label], batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
	return example_batch, label_batch

file_length = file_len(args.dataset) -1 
# print(file_length)
examples,labels = input_pipeline(file_length, 1)
# print (file_length)
	
#initiate the training
with tf.Session() as sess:
	tf.global_variables_initializer()
	# init_op = tf.group(tf.global_variables_initializer, tf.local_variables_initializer())
	# tf.global_variables_initializer()
	# sess.run(init_op)
	#populating the filename queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	try:
		while not coord.should_stop():			
			example_batch, label_batch = sess.run()
			print(example_batch)
	#when out of range
	except tf.errors.OutOfRangeError:
		print("Done Training, epoch reached")
	finally:
		coord.request_stop()	#stop requests
		
	coord.join(threads)			
	sess.close()				#close the session