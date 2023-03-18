import re
import string
import spacy
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Liste des fichiers dans le répertoire "resources"
files = os.listdir('resources')


def suppress_empty_words():
    global stop_words, text

    stop_words = set(stopwords.words('french'))
    stop_words.add('illisible')
    filter_text_from_words_list(stop_words)

    stopwords_eng = stopwords.words('english')
    filter_text_from_words_list(stopwords_eng)

    nlp = spacy.load("fr_core_news_sm")
    filter_text_with_spacy(nlp)

    nlp = spacy.load("it_core_news_sm")
    filter_text_with_spacy(nlp)


def filter_text_with_spacy(nlp):
    global text
    doc = nlp(text)
    # Liste des tokens qui ne sont pas des mots vides
    tokens = [token.text.lower() for token in doc if not token.is_stop]
    text = " ".join(tokens)


def filter_text_from_words_list(stop_words):
    global text
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    # Reconstruction du texte filtré
    text = " ".join(filtered_words)


def suppress_months():
    global text
    # Définition d'une expression régulière pour les noms de mois
    mois_regex = r'(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s'

    # Remplace les noms de mois par une chaîne vide
    text = re.sub(mois_regex, '', text)


def suppress_days():
    global text
    # Définition d'une expression régulière pour les noms de jours
    mois_regex = r'(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s'

    # Remplace les noms de mois par une chaîne vide
    text = re.sub(mois_regex, '', text)


def suppress_adverbs_and_others():
    global text
    # Définition d'une expression régulière pour les adverbes de temps et de lieux
    adverbs_regex = r'(aujourd’hui|demain|hier|ici|là|jadis|auparavant|autrefois|bientôt|maintenant|mr|mme|cher|Jules|Iules|jules|trop|peu|bien|mlle|jusqu|rien|beaucoup|frère|val|valer|paul|montpellier|peutêtre|good|bye|pv|mieux|embrasse|adieu)\s'

    # Remplace les adverbes par une chaîne vide
    text = re.sub(adverbs_regex, '', text)


def suppress_vocatives():
    global text
    # Définition d'un pattern pour identifier les vocatifs
    pattern = re.compile(r"(mon|très) cher frère")

    # Remplace les expressions trouvées par une chaîne vide
    text = pattern.sub("", text)


# Boucle sur tous les fichiers
for file in files:
    # Ouvrir le fichier texte
    with open('resources/' + file, 'r', encoding='utf8') as f:
        text = f.read()

    # Suppression des chiffres romains
    text = re.sub(r'\b[IVXLCDM]+\b', '', text)

    # Transformation en minuscules
    text = text.lower()

    # Suppression des chiffres
    text = ''.join([i for i in text if not i.isdigit()])

    # Suppression des ponctuations
    text = text.translate(str.maketrans(" ", " ", string.punctuation))

    # Suppression des mois
    suppress_months()

    # Suppression des jours
    suppress_days()

    # Suppression des adverbes de temps et de lieux
    suppress_adverbs_and_others()

    # Suppression des vocatifs
    suppress_vocatives()

    # Suppression des mots vides
    suppress_empty_words()

    # Suppression des caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text)

    # Suppression des lettres seules
    text = re.sub(r'\b\w{1}\b', '', text)

    # Suppression des espaces multiples
    text = re.sub('\s+', ' ', text)

    with open('clean_docs/clean_' + file, 'w', encoding='utf-8') as f:
        f.write(text)

print('End of text clean process')
