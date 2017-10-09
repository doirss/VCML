# VCML: ValueCentric Machine Learning Project

This is a machine learning proof of concept that fuzzy matches location data.

# Usage

This program is run via command line with python>=3.6

## Training

When training the model, run:

	python3.6 VCML.py -train ml_training_data_copy.csv

Which will output an training accuracy on the dataset and stores in training.txt file containing model used, runtime, operations performed and accuracy percentage.

## Testing

To test the results of the current model, run:
	
	py VCML.py -test <test_data.csv> <answer_data.csv>

Which will output various statistics to testing.txt


