# VCML: ValueCentric Machine Learning Project

This is a machine learning proof of concept that fuzzy matches location data.

# Usage

This program is run via command line with python>=3.6

## Training

When training the model, run:

	# py VCML.py -train <training_data.csv>

Which will output a training.txt file containing model used, runtime, and operations performed

## Testing

To test the results of the current model, run:
	
	# py VCML.py -test <test_data.csv> <answer_data.csv>

Which will output various statistics to testing.txt
