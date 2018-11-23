import nltk
import json
import os
import time

from analysis.nltk_ibm import *
from models.ibm_model import *
from models.phrases import *

def app_check():
    """
    Checks for dataset existence
    """
    print("Checking for dataset ...")
    print("Searching for en_fr.json")
    for f in os.listdir("./data"):
        if str(f).find("data1.json") or str(f).find("data2.json"):
            print("Dataset exists")
            return True
        else:
            print("Dataset doesn't exist")
            return False

def import_data(data=1):
    """
    Reads data from .json files and returns a list of dictionaries.
    """
    try:
        # IBM 1 and EM algorithm implementation
        if data == 1:
            with open('data/data1.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
        elif data == 2:
            # NLTK IBM 1 and IBM 2 and Phrase based extraction implementation.
            with open('data/data2.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
        else:
            with open('data/en_fr.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
    except Exception as e:
        print("Read error. Check file again")
        return False

if __name__=="__main__":
    if app_check():
        ### IBM 1 with EM algorithm implementation ###
        start = time.time()
        en_lex, fr_lex, t = em_algorithm(import_data(data=1), iterations=30)
        alignments = alignment(import_data(data=1), en_lex, fr_lex, t)
        for a in alignments:
            print(a)
        print("Q1 time:", time.time() - start)

        en_lex, fr_lex, t = em_algorithm(import_data(data=2), iterations=30)
        alignments = alignment(import_data(data=2), en_lex, fr_lex, t)
        for a in alignments:
            print(a)
        # Phrase based translation on data2
        start = time.time()
        extract_phrases_and_compute_score(import_data(data=1), alignments)
        print("Q2 time:", time.time() - start)

        en_lex, fr_lex, t = em_algorithm(import_data(data=3), iterations=30)
        alignments = alignment(import_data(data=3), en_lex, fr_lex, t)
        for a in alignments:
            print(a)
        # Phrase based translation on our data
        extract_phrases_and_compute_score(import_data(data=3), alignments)

        # NLTK IBM 1 and IBM 2 implementation
        start = time.time()
        nltk_ibm_one(import_data(data=1))
        nltk_ibm_two(import_data(data=1))
        print("Q3 time: ", time.time() - start)
        
        nltk_ibm_one(import_data(data=1))
        nltk_ibm_two(import_data(data=1))

        nltk_ibm_one(import_data(data=3))
        nltk_ibm_two(import_data(data=3))

    else:
        print("Dataset error. Check again. Ending")
