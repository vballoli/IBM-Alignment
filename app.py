import nltk
import json
import os

from analysis.nltk_ibm import *

def app_check():
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
    try:
        # IBM 1 and EM algorithm implementation
        if data == 1:
            with open('data/data1.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
        else:
            # NLTK IBM 1 and IBM 2 and Phrase based extraction implementation.
            with open('data/data2.json', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data
    except Exception as e:
        print("Read error. Check file again")
        return False

if __name__=="__main__":
    if app_check():
        # Implement IBM 1 and EM algorithm
        # NLTK IBM 1 and IBM 2 implementation
        nltk_ibm_one(import_data(data=2))
        nltk_ibm_two(import_data(data=2))
        # NLTK based Phrase extractions
    else:
        print("Dataset error. Check again. Ending")
