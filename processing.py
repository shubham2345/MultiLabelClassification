import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.models import KeyedVectors

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

def transform(data_path, vectorizer=None):
    df = pd.read_csv(data_path)

    if 'train' in data_path:
        # Initialize CountVectorizer for character level tokenization
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 5))

        # Fit and transform the UTTERANCES column
        input_vectorizer = vectorizer.fit_transform(df['UTTERANCES'])
        
        # Convert the sparse matrix to a DataFrame
        input_vectorized_df = pd.DataFrame(input_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())

        # Encode the labels
        df['CORE RELATIONS'] = df['CORE RELATIONS'].apply(lambda x: x.split())
        unique_relations = set()
        df['CORE RELATIONS'].apply(unique_relations.update)
        unique_relations = sorted(unique_relations)

        label_encoded_df = pd.DataFrame(0, index=df.index, columns=unique_relations)
        for index, row in df.iterrows():
            for relation in row['CORE RELATIONS']:
                label_encoded_df.at[index, relation] = 1
        
        return input_vectorized_df, label_encoded_df, vectorizer  
    else:

        input_vectorizer = vectorizer.transform(df['UTTERANCES'])
        input_vectorized_df = pd.DataFrame(input_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
        
        return input_vectorized_df, None  


