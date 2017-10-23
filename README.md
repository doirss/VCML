# VCML: ValueCentric Machine Learning Project

This is a machine learning proof of concept that fuzzy matches location data.

# Usage

This program is run via command line with python>=3.6

## Training

When training the model, run:

	python3.6 VCML.py -train <training_data.csv>

Which will output a training accuracy on the dataset and stores in training.txt file containing model used, runtime, operations performed and accuracy percentage.

## Testing

To test the results of the current model, run:
	
	py VCML.py -test <test_data.csv> <answer_data.csv>

Which will output various statistics to testing.txt

# Installation
## Linux systems
The main dependency for this project is scikit-learn, numpy, and scipy (all through Anaconda); the script below will install everything:

	# Go to home directory
	cd ~

	# You can change what anaconda version you want at 
	# https://repo.continuum.io/archive/
	wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh
	bash Anaconda3-5.0.0-Linux-x86_64.sh -b -p ~/anaconda
	rm Anaconda3-5.0.0-Linux-x86_64.sh
	echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

	# Refresh basically
	source .bashrc

	conda update conda


