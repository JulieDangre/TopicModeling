import numpy as np
import pandas as pd
import os
import gensim
from gensim import corpora, models

# Liste des fichiers dans le répertoire "resources"
files = os.listdir('clean_docs')


def process_lda_model():
    print("LDA MODEL")
    num_topics = 10
    num_words = 10
    lda_model = models.LdaModel(corpus_bow, num_topics=num_topics, id2word=dictionary, passes=10, iterations=2000)

    # Initialize the coherence evaluation model from gensim
    cm = gensim.models.coherencemodel.CoherenceModel(model=lda_model,
                                                     texts=tokenized_corpus,
                                                     coherence='c_v',
                                                     topn=num_words,
                                                     processes=1
                                                     )

    # Get the coherence of each topic and round the values
    coherence = cm.get_coherence_per_topic()
    coherence = [round(score, 3) for score in coherence]

    # Print topics
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=True)

    # Add a placeholder list to hold the final output
    result = []

    # Loop over the topics
    for t in topics:
        # Assign the word string into their own variable
        words = t[1]

        # Split the words
        words = words.split(' + ')

        # Extract the words from the predictions
        words = [w.split('*')[1].strip('"') for w in words]

        # Append the row to the final clean_docs
        result.append(words)

    # Convert the clean_docs into a NumPy array and transpose.
    result = np.vstack(result).transpose()

    # Append the coherence scores (Cv) to the matrix
    result = np.vstack([result, coherence])

    # Convert the clean_docs into a DataFrame
    return pd.DataFrame(result, columns=range(1, num_topics+1))


# Boucle sur tous les fichiers
for file in files:
    # Charger le corpus à partir du fichier
    with open('clean_docs/' + file, "r", encoding="utf-8") as f:
        corpus = f.read().splitlines()

    # Tokenizer le corpus
    tokenized_corpus = [gensim.utils.simple_preprocess(doc) for doc in corpus]

    # Créer un dictionnaire de tous les mots uniques dans le corpus
    dictionary = corpora.Dictionary(tokenized_corpus)

    print("[INFO] Size of dictionary: {}".format(len(dictionary)))

    # Convertir chaque document en une représentation vectorielle
    corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

    result = process_lda_model()

    with open('topic_modeling_docs/processed_' + file, 'w', encoding='utf-8') as f:
        f.write(result.to_latex(na_rep='--', index=False))
