import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """

    bow = {}
    # TODO: add your code here
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line in vocab:
                if line in bow:
                    bow[line] += 1
                else:
                    bow[line] = 1
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    """count2020 = smooth
    count2016 = smooth

    for doc in training_data:
        if doc['label'] == label_list[0]:
            count2020 += 1
        elif doc['label'] == label_list[1]:
            count2016 += 1
    logprob['2020'] = math.log((count2020) / (len(training_data) + 2))
    logprob['2016'] = math.log((count2016) / (len(training_data) + 2))"""

    for year in label_list:
        logprob[year] = smooth
    for doc in training_data:
        logprob[doc['label']] += 1
    for year in logprob:
        logprob[year] = math.log((logprob[year]) / (len(training_data) + 2))

    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    wordCounts = {word: 0 for word in vocab}
    wordCounts[None] = 0
    n = 0
    sizeVocab = len(vocab)

    for doc in training_data:
        if doc['label'] == label:
            for word in doc['bow']:
                n += doc['bow'][word]
                if word in wordCounts:
                    wordCounts[word] += doc['bow'][word]
                else:
                    wordCounts[word] = doc['bow'][word]
    for word in wordCounts:
        word_prob[word] = math.log((wordCounts[word] + smooth*1) / (n + smooth*(sizeVocab+1)))

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data,label_list)
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, label_list[0])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, label_list[1])

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    prob2016 = model['log prior']['2016']
    prob2020 = model['log prior']['2020']

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line in model['vocabulary']:
                prob2016 += model['log p(w|y=2016)'][line]
                prob2020 += model['log p(w|y=2020)'][line]
            else:
                prob2016 += model['log p(w|y=2016)'][None]
                prob2020 += model['log p(w|y=2020)'][None]
    predictedY = '2016' if max(prob2016,prob2020) == prob2016 else '2020'
    retval['log p(y=2020|x)'] = prob2020
    retval['log p(y=2016|x)'] = prob2016
    retval['predicted y'] = predictedY

    return retval
