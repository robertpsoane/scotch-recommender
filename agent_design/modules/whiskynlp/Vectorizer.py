from .WhiskyLemmatizer import WhiskyLemmatizer
from .GraphKeywordExtraction import GraphKE
import pandas as pd
import numpy as np

class ListFeatureVectorizer:
    def __init__(self, features):
        self.Lemmatizer = WhiskyLemmatizer()
        self.makeList = GraphKE().makeCorpusList
        self.features = features
    
    def fit(self, input_list, norm=True):
        list_vec = []
        
        for doc in input_list:
            tokenized = self.Lemmatizer.tokenFilter(doc)
            docvec = [tokenized.count(w) for w in self.features]
            
            if norm:
                docvec = np.array(docvec)
                nrm = np.linalg.norm(docvec)
                if nrm > 0:
                    docvec = docvec / nrm
                docvec = list(docvec)

            list_vec.append(docvec)
            
        outdf = pd.DataFrame(list_vec, columns=self.features)
        return outdf

    def prepropRemoveNas(self, df, col):
        """
        Function to return dataframe without missing values for given 
        column.  Also returns that column as a list able to be input 
        into fit function.
        """
        new_df = df[pd.isna(df[col]) == False].reset_index()
        lst = list(new_df[col])
        return new_df, lst

    
    def __repr__(self):
        return "<ListFeatureVectorizer>"