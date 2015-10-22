def gender_features(word):
    return {'last_letter': word[-1]}

#print gender_features('Shrek')

import nltk

from nltk.corpus import names
import random
names = ([( name, 'male') for name in names.words('male.txt')]+
         [( name, 'female') for name in names.words('female.txt')])
random.shuffle( names)

featuresets = [( gender_features( n), g) for (n, g) in names]
train_set, test_set = featuresets[ 500:], featuresets[: 500]

#print train_set
#classifier = nltk.NaiveBayesClassifier.train( train_set)
## classifier2 = nltk.MaxentClassifier.train( train_set)

#print classifier.classify( gender_features(' Neo'))
#print classifier.classify( gender_features(' Trinity'))

dataset=[({'score': 50.10},'B'),({'score': 100.04},'B'),({'score': 100.21},'G'), ({'score': 100.07},'B'),
        ({'score': 200.27},'B'),({'score': 200.11}, 'G'),({'score': 200.03}, 'G'),({'score': 300.05}, 'G')]
print dataset

classifier = nltk.NaiveBayesClassifier.train( dataset)
print classifier.classify({'score': 100})
print classifier.classify({'score': 200})

classifier2 = nltk.MaxentClassifier.train( dataset)
print classifier2.classify({'score': 100})

print("score 200")
print classifier2.classify({'score': 200})

print("weights")
print classifier2._weights
print classifier2.show_most_informative_features()
print classifier2.prob_classify({'score': 200.27}).prob('G')
print classifier2.prob_classify({'score': 200.27}).prob('B')

print classifier2.prob_classify({'score': 200.1}).prob('G')
print classifier2.prob_classify({'score': 200.1}).prob('B')

print classifier2.explain({'score': 200.27})