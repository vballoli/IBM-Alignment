from nltk.translate import phrase_based

def extract_phrases_and_compute_score(corpus, alignments):
    """
    Returns phrase probability scores based on nltk.translate.phrase_based module
    """
    print("Inside extraction")
    for i in range(len(corpus)):
        src = corpus[i]['fr']
        tgt = corpus[i]['en']
        alignment = alignments[i]
        phrases = phrase_based.phrase_extraction(src, tgt, alignment)

        phrase_pairs = []
        ref_phrase = []
        for phrase in sorted(phrases):
            phrase_pairs.append([phrase[2], phrase[3]])
            ref_phrase.append(phrase[3])

        for pair in phrase_pairs:
            num = phrase_pairs.count(pair)
            denom = ref_phrase.count(pair[1])
            print(float(num)/denom)
