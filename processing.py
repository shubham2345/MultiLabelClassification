import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler


# def transform(data_path, vectorizer=None):
def transform(data_path, vectorizer=None, count_vectorizer_scaler=None, word2vec_scaler=None):

    # Initialize scalers for normalization
    count_vectorizer_scaler = MinMaxScaler()
    word2vec_scaler = MinMaxScaler()
    print("hi")
    df = pd.read_csv(data_path)
    


    if 'train' in data_path:
        vectorizer = CountVectorizer()
        # Fit and transform the UTTERANCES column
        input_vectorizer = vectorizer.fit_transform(df['UTTERANCES'])
        # Convert the sparse matrix to a DataFrame for better visualization
        count_vectorizer_df = pd.DataFrame(input_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
        # print(input_vectorized_df)

        # Assuming your sentences are tokenized
        sentences = df['UTTERANCES'].apply(lambda x: x.split()).tolist()

        # Train a Word2Vec model
        word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        # Create a DataFrame for Word2Vec embeddings
        word2vec_vectors = np.array([sentence_to_vector(sentence, word2vec_model) for sentence in df['UTTERANCES']])
        word2vec_df = pd.DataFrame(word2vec_vectors)

        count_vectorizer_df = count_vectorizer_scaler.fit_transform(count_vectorizer_df)
        word2vec_df = word2vec_scaler.fit_transform(word2vec_df)

        # x_train = pd.concat([count_vectorizer_df, word2vec_df], axis=1)
        
        x_train = pd.DataFrame(np.hstack((count_vectorizer_df, word2vec_df)),
                                         columns=[*vectorizer.get_feature_names_out(), *[f'word2vec_{i}' for i in range(word2vec_df.shape[1])]])



        df['CORE RELATIONS'] = df['CORE RELATIONS'].apply(lambda x: x.split())
        unique_relations = set()
        df['CORE RELATIONS'].apply(unique_relations.update)
        unique_relations = sorted(unique_relations)
        # print("unique_relations", unique_relations)

        label_encoded_df = pd.DataFrame(0, index=df.index, columns=unique_relations)
    

        for index, row in df.iterrows():
            for relation in row['CORE RELATIONS']:
                label_encoded_df.at[index, relation] = 1
        
        return x_train, label_encoded_df, vectorizer, count_vectorizer_scaler, word2vec_scaler
        # return x_train, label_encoded_df, vectorizer
        # return word2vec_df, label_encoded_df, vectorizer
    else:
        print("byeeeeeee")
        
        # Fit and transform the UTTERANCES column
        input_vectorizer = vectorizer.transform(df['UTTERANCES'])
        # Convert the sparse matrix to a DataFrame for better visualization
        count_vectorizer_df = pd.DataFrame(input_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
        # print(input_vectorized_df)

        # Assuming your sentences are tokenized
        sentences = df['UTTERANCES'].apply(lambda x: x.split()).tolist()

        # Train a Word2Vec model
        word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        # Create a DataFrame for Word2Vec embeddings
        word2vec_vectors = np.array([sentence_to_vector(sentence, word2vec_model) for sentence in df['UTTERANCES']])
        word2vec_df = pd.DataFrame(word2vec_vectors)

        count_vectorizer_df = count_vectorizer_scaler.fit_transform(count_vectorizer_df)
        word2vec_df = word2vec_scaler.fit_transform(word2vec_df)

        # x_train = pd.concat([count_vectorizer_df, word2vec_df], axis=1)
        # x_train=np.hstack((count_vectorizer_df, word2vec_df))
                # Combine normalized outputs into a DataFrame
        x_train = pd.DataFrame(np.hstack((count_vectorizer_df, word2vec_df)),
                                         columns=[*vectorizer.get_feature_names_out(), *[f'word2vec_{i}' for i in range(word2vec_df.shape[1])]])
    
        # input_vectorizer = vectorizer.transform(df['UTTERANCES'])
        # # Convert the sparse matrix to a DataFrame
        # input_vectorized_df = pd.DataFrame(input_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
        return x_train, None  # No labels for test data
        # return word2vec_df, None
    
    # label_encoded_df = pd.concat([df[['ID', 'UTTERANCES']], label_encoded_df], axis=1)

def sentence_to_vector(sentence, model):
    # print(model.wv)
    words = sentence.split()
    # print(words)
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    # print(word_vectors)
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Return a zero vector if no words found
    return np.mean(word_vectors, axis=0)


def training_validation_dataset(input_vectorized_df, label_encoded_df):
    pass
    
    # labels = encoded_df.iloc[:, 2:]

    # X_train, X_val, y_train, y_val = train_test_split(input_vectorized_df, label_encoded_df, test_size=0.2, random_state=42)


    # training_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    # validation_data = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    # training_data.to_csv('training_data.csv', index=False)
    # validation_data.to_csv('validation_data.csv', index=False)
    # return X_train, X_val, y_train, y_val
    
