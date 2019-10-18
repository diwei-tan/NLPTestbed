from nltk.tokenize import word_tokenize

#to ensure tokenization of the words and expressions in the reviews
def form_sent(sent):
    return {word: True for word in word_tokenize(sent)}

#initialize some sentiments
s1 = 'This is a good book'
s2 = 'This is some great shit!'
s3 = 'Author clearly knows his stuff, which is some awesome stuff!'
s4 = 'My god this made for some terrible reading'
s5 = 'Author did it again! Great read'
s6 = 'This is a terrible book. Do not read'
s7 = 'This is a bad book'
s8 = 'This is an awesome book'
s9 = 'Must have been ages since I read something so meaningful! Great stuff!'

training_data = [[form_sent(s1),'pos'],[form_sent(s2),'pos'],[form_sent(s3),'pos'],
[form_sent(s4),'neg'],[form_sent(s5),'pos'],[form_sent(s6),'neg'],
[form_sent(s7),'neg'],[form_sent(s8),'pos']]

#import sentiment classifer: Naive Bayes Classifier
from nltk.classify import NaiveBayesClassifier
model = NaiveBayesClassifier.train(training_data)
#Unit test
t1='This book is inspiring!'
t2='Never want to touch this nonsense again'
t3='This book is such a drag...'
print(model.classify(form_sent(t1)))
print(model.classify(form_sent(t2)))
print(model.classify(form_sent(t3)))

test_data = [[form_sent(t1),'pos'],[form_sent(t2),'neg'],[form_sent(t3),'neg']]
from nltk.classify import accuracy
print(accuracy(model, test_data))