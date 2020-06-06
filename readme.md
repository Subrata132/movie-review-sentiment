# Dataset
	Dataset was taken from : http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

	
# Dataset Overview
	Dataset contains 2000 movie reviews. 1000 each for positve and negative reviews.
	Entire dataset the splitted into test and training set. (train-900,test-100 for both positive and negative reviews)
	Splitted data can be found in 'movie reviews' folder 
	
# Vocabolary Creation
	vocabolary is created from the training set only.
	A file named 'vocab.txt' is located in in 'dictionary' folder. 
	This set of words have been used for fitting the tokenizer.
	'create_vocab.py' have been used to create the 'vocab.txt' file
	
# Test and Train 
	To test the model keep 'retrain=False' (sentiment_analyzier.py ,line 159)
	To retrain make 'retrain=True' . It's recomended to change the 'model_name'(sentiment_analyzier.py ,line 167) while retraining.
	A text file named 'review.txt' is used to test the sentiment. 
	
# Model
	'sentNet.hdf5' is located in 'model' folder.
	
# Contact the Author:
	Subrata Biswas
	4th year undergraduate student,
	Department of EEE , 
	Bangladesh University of Engineering & Technology.
	
	Email: subrata.biswas@ieee.org
	LinkedIn : https://www.linkedin.com/in/subrata-biswas-433247142/
	