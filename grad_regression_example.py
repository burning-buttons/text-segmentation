import nltk
import numpy as np
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# download test data
nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.corpus import twitter_samples 
from nltk.corpus import stopwords

def process_text(tweet):
  """Process tweet function.
  Input:
    tweet: a string containing a tweet
  Output:
    tweets_clean: a list of words containing the processed tweet
  """
  stemmer = PorterStemmer()
  stopwords_english = stopwords.words('english')
  # remove stock market tickers like $GE
  tweet = re.sub(r'\$\w*', '', tweet)
  # remove old style retweet text "RT"
  tweet = re.sub(r'^RT[\s]+', '', tweet)
  # remove hyperlinks
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
  # remove hashtags
  # only removing the hash # sign from the word
  tweet = re.sub(r'#', '', tweet)

  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
  tweet_tokens = tokenizer.tokenize(tweet)

  tweets_clean = []
  for word in tweet_tokens:
    if (word not in stopwords_english and word not in string.punctuation):  # remove stopwords and punctuation
      stem_word = stemmer.stem(word)
      tweets_clean.append(stem_word)

  return tweets_clean

def build_freqs(tweets, ys):
  """Build frequencies.
  Input:
    tweets: a list of tweets
    ys: an m x 1 array with the sentiment label of each tweet 0/1
  Output:
    freqs: a dictionary mapping each (word, sentiment) pair to its frequency
  """
  yslist = np.squeeze(ys).tolist()

  freqs = {}
  for y, tweet in zip(yslist, tweets):
    for word in process_text(tweet):
      pair = (word, y)
      if pair in freqs:
        freqs[pair] += 1
      else:
        freqs[pair] = 1

  return freqs

def sigmoid(z): 
  '''
  Input:
      z: rate
  Output:
      h: the sigmoid of z
  '''
  h = 1/(1 + np.exp(-z))
  return h

def gradient_descent(x, y, theta, alpha, num_iters):
  '''
  Input:
    x: matrix of features which is (m,n+1)
    y: corresponding labels of the input matrix x, dimensions (m,1)
    theta: weight vector of dimension (n+1,1)
    alpha: learning rate
    num_iters: number of iterations
  Output:
    J: the final cost
    theta: final weight vector
  '''
  m = len(x)
  for i in range(0, num_iters):
    # get z, the dot product of x and theta
    z = np.dot(x, theta)
    # get the sigmoid of z
    h = sigmoid(z)
    # calculate the cost function
    J = (-1./m)*(np.dot(np.transpose(y), np.log(h)) + np.dot(np.transpose(1-y), np.log(1-h)))
    # update the weights theta
    theta = theta - np.dot(alpha/m, np.dot(np.transpose(x), h-y))

  J = float(J)
  return J, theta

def extract_features(tweet, freqs):
  '''
  Input: 
    tweet: a list of words for one tweet
    freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
  Output: 
    x: a feature vector of dimension (1,3)
  '''
  word_l = process_text(tweet)
  
  x = np.zeros((1, 3)) 
  #bias term is set to 1
  x[0,0] = 1 
  # set features for each word.
  for word in word_l:
    x[0,1] += freqs.get((word, 1.0), 0)
    x[0,2] += freqs.get((word, 0.0), 0)
  assert(x.shape == (1, 3))
  return x

def predict_tweet(tweet, freqs, theta):
  '''
  Input: 
      tweet: a string
      freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
      theta: (3,1) vector of weights
  Output: 
      y_pred: the probability of a tweet being positive or negative
  '''
  # extract the features of the tweet and store it into x
  x = extract_features(tweet, freqs)

  # make the prediction using x and theta
  y_pred = sigmoid(np.dot(x, theta))

  return y_pred

def test_regression(test_x, test_y, freqs, theta):
  """
  Input: 
    test_x: a list of tweets
    test_y: (m, 1) vector with the corresponding labels for the list of tweets
    freqs: a dictionary with the frequency of each pair (or tuple)
    theta: weight vector of dimension (3, 1)
  Output: 
    accuracy: (# of tweets classified correctly) / (total # of tweets)
  """
  y_hat = []

  for tweet in test_x:
    # get the label prediction for the tweet
    y_pred = predict_tweet(tweet, freqs, theta)
    if y_pred > 0.5:
      y_hat.append(1.0)
    else:
      y_hat.append(0)

  accuracy = np.sum([1 for x in (np.asarray(y_hat) == np.squeeze(test_y)) if x == True])/len(test_y)
  return accuracy

# prepare data, split to training/testing sets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# collect the features
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent, params selected experimentally
J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


accuracy = test_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {accuracy:.4f}")


my_tweet = 'This is a sad tweet!'
print(process_text(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')