import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from nltk.corpus import stopwords
import json



class WhiskyLemmatizer(WordNetLemmatizer):
    '''
    An extension on the WordNet Lemmatizer with added context for whisky
    '''
        
    def __init__(self):

        # Loading dictionary of lemmatizations
        module_dir = os.path.dirname(os.path.abspath(__file__))
        lemmatizer = os.path.join(module_dir, 'whisky_lemmatizer_dict.json')
        with open(lemmatizer) as json_in:
            corpus = json.load(json_in)
        self.whisky_words = corpus

        # Loading stopwords
        stopwords_json = os.path.join(module_dir, 'stopwords.json')
        with open(stopwords_json) as json_in:
            swords = json.load(json_in)
        self.swords = set(swords["swords"])
        
    
    def lemmatize(self, word):
        # Caches lemmatized words to avoid lookups
        if word in self.whisky_words:
            out = self.whisky_words[word]
        else:
            tag = self.tag(word)
            out = super().lemmatize(word, pos=tag)
            self.whisky_words[word] = out
        return out
    
    def lemmatizeList(self, lst):
        return [self.lemmatize(w) for w in lst]
    
    def whiskySub(self, word):
        if word in self.whisky_words:
            return self.whisky_words[word]
        else:
            return word
    
    def tag(self, word):
        tag = pos_tag([word])
        tag = pos_tag([word])[0][1][0].lower()
        if tag == "v":
            return "v"
        if tag == "j":
            return "a"
        else:
            return "n"
        
    def tokenFilter(self, corpus):
        tokens = word_tokenize(corpus)
        filtered = [w for w in self.lemmatizeList(tokens) if 
                                        (not w in self.swords)]
        return filtered
        
    def __repr__(self):
        return "<WhiskyLemmatizer>"

if __name__ == "__main__":
    lemmatizer = WhiskyLemmatizer()
    while True:
        inp = input("Word to lemmatize: ")
        print(lemmatizer.lemmatize(inp))