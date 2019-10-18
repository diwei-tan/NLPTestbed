#Github testbed for NLP related codes
#Natural Language Tool Kit Exploration
#Requires installation of nltk and matplotlib
#pip install nltk and matplotlib
#import nltk
#nltk.download()

import nltk

#Read some online web html file
import urllib.request
response = urllib.request.urlopen('http://php.net')
html = response.read()

#Use BeautifulSoup to clean page of html tags, etc. to only have words left
#Note: There are other 'types' it can use to clean the read
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(strip=True)
tokens = [t for t in text.split()]
print(type(tokens))
print(type(tokens[1]))

# for key, value in freq.items():
#     print(str(key) + ':' + str(value))

#plot frequency of words
# import matplotlib
# freq.plot(20)

#Has many stop words such as 'the, 'of', 'a', etc. Remove them
from nltk.corpus import stopwords
stops = stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stops:
        clean_tokens.remove(token)
freq = nltk.FreqDist(clean_tokens)
freq.plot(20)

#This section is amount tokenization of text
string = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
print(sent_tokenize(string))
print(word_tokenize(string))

#tokenize other languages
french = "Bonjour M. Adam, comment allez-vous? J'espère que tout va bien. Aujourd'hui est un bon jour."
print(sent_tokenize(french,"french"))

#Wordnet synonyms and antonyms
from nltk.corpus import wordnet
synonyms = []
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

#Antonymns
antonyms = []
for syn in wordnet.synsets('small'):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)

#Word stemming/lemmatization
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 
print(stemmer.stem('stones')) 
print(stemmer.stem('speaking')) 
print(stemmer.stem('bedroom')) 
print(stemmer.stem('jokes')) 
print(stemmer.stem('lisa')) 
print(stemmer.stem('purple')) 
print('----------------------') 
print(lemmatizer.lemmatize('stones')) 
print(lemmatizer.lemmatize('speaking'))
print(lemmatizer.lemmatize('bedroom'))
print(lemmatizer.lemmatize('jokes'))
print(lemmatizer.lemmatize('lisa'))
print(lemmatizer.lemmatize('purple'))