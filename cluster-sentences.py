#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import operator

__author__ = 'dsbatista'

import codecs
import re
import sys
import StringIO
import pickle
import numpy

from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise
from gensim.models import Word2Vec
from Word2VecWrapper import Word2VecWrapper
from Sentence import Sentence

N_GRAMS_SIZE = 4

#TODO: incluir virgulas na tokenização
TOKENIZER = r'\,|\(|\)|\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.' \
            r'-]+|https?://[^\s]+|\w+'


def load_relationships(data_file):
    relationships = list()
    rel_id = 0
    f_sentences = codecs.open(data_file, encoding='utf-8')
    #f_sentences = codecs.open(data_file)
    for line in f_sentences:
        s = Sentence(line.strip())
        for r in s.relationships:
            rel_id += 1
            relationships.append(r)
    return relationships


def extract_reverb_patterns_ptb(text):

    """
    Extract ReVerb relational patterns
    http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf

    # extract ReVerb patterns:
    # V | V P | V W*P
    # V = verb particle? adv?
    # W = (noun | adj | adv | pron | det)
    # P = (prep | particle | inf. marker)
    """

    # remove the tags and extract tokens
    text_no_tags = re.sub(r"</?[A-Z]+>", "", text)
    tokens = re.findall(TOKENIZER, text_no_tags, flags=re.UNICODE)

    # tag tokens with pos-tagger
    tagged = tagger.tag(tokens)
    patterns = []
    patterns_tags = []
    i = 0
    limit = len(tagged)-1
    tags = tagged

    """
    verb = ['VB', 'VBD', 'VBD|VBN', 'VBG', 'VBG|NN', 'VBN', 'VBP', 'VBP|TO', 'VBZ', 'VP']
    adverb = ['RB', 'RBR', 'RBS', 'RB|RP', 'RB|VBG', 'WRB']
    particule = ['POS', 'PRT', 'TO', 'RP']
    noun = ['NN', 'NNP', 'NNPS', 'NNS', 'NN|NNS', 'NN|SYM', 'NN|VBG', 'NP']
    adjectiv = ['JJ', 'JJR', 'JJRJR', 'JJS', 'JJ|RB', 'JJ|VBG']
    pronoun = ['WP', 'WP$', 'PRP', 'PRP$', 'PRP|VBP']
    determiner = ['DT', 'EX', 'PDT', 'WDT']
    adp = ['IN', 'IN|RP']
    """

    verb = ['verb', 'verb_past']
    word = ['noun', 'adjective', 'adverb', 'determiner', 'pronoun']
    particule = ['preposition']

    """
    conjunction
    electronic
    interjection
    numeral
    punctuation
    symbol

    adjective
    adverb
    determiner
    noun
    preposition
    pronoun
    verb
    verb_past
    """

    while i <= limit:
        tmp = StringIO.StringIO()
        tmp_tags = []

        # a ReVerb pattern always starts with a verb
        if tags[i][1] in verb:
            tmp.write(tags[i][0]+' ')
            t = (tags[i][0], tags[i][1])
            tmp_tags.append(t)
            i += 1

            # V = verb particle? adv? (also capture auxiliary verbs)
            while i <= limit and (tags[i][1] in verb or tags[i][1] in word):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # W = (noun | adj | adv | pron | det)
            while i <= limit and (tags[i][1] in word):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # P = (prep | particle | inf. marker)
            while i <= limit and (tags[i][1] in particule):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # add the build pattern to the list collected patterns
            patterns.append(tmp.getvalue())
            patterns_tags.append(tmp_tags)
        i += 1

    return patterns, patterns_tags


def extract_patterns(text, context):
    # each sentence contains one relationship only
    patterns, patterns_tags = extract_reverb_patterns_ptb(text)

    # detect which patterns contain passive voice
    extracted_patterns = list()
    for pattern in patterns_tags:
        passive_voice = False
        for i in range(0, len(pattern)):
            if pattern[i][1].startswith('v'):
                try:
                    inf = verbs[pattern[i][0]]
                    if inf in ['ser', 'estar', 'ter', 'haver'] and i+2 <= len(pattern)-1:
                        if (pattern[-2][1] == 'verb_past' or pattern[-2][1] == 'adjectiv') and pattern[-1][0] == 'por':
                            passive_voice = True
                except KeyError:
                    continue

        if passive_voice is True:
            p = '_'.join([tag[0] for tag in pattern])
            p += '_PASSIVE_VOICE_'+context
        else:
            p = '_'.join([tag[0] for tag in pattern])
            p += '_'+context

        extracted_patterns.append(p)

    return extracted_patterns


def main():
    """
    global verbs
    print "Loading Label-Delaf"
    verbs_conj = open('verbs/verbs_conj.pkl', "r")
    verbs = pickle.load(verbs_conj)
    verbs_conj.close()
    """

    print "Loading relationships from", sys.argv[3]
    relationships = load_relationships(sys.argv[3])
    print len(relationships), "relationships loaded"

    global tagger
    print "Loading PoS tagger from", sys.argv[1]
    f_model = open(sys.argv[1], "rb")
    tagger = pickle.load(f_model)
    f_model.close()

    print "Loading word2vec model ...\n"
    word2vec = Word2Vec.load_word2vec_format(sys.argv[2], binary=True)

    print "Extracting ReVerb patterns"
    bet_vectors = []
    words = stopwords.words('portuguese')
    for rel in relationships:
        patterns_bet, patterns_bet_tags = extract_reverb_patterns_ptb(rel.between)
        if len(patterns_bet) > 0:
            #print patterns_bet
            pattern = [t[0] for t in patterns_bet_tags[0] if t[0].encode("utf8").lower() not in words]
            #print pattern
            #print "\n"
            if len(pattern) >= 1:
                pattern_vector_bet = Word2VecWrapper.pattern2vector(pattern, word2vec)
                bet_vectors.append(pattern_vector_bet)
            else:
                pattern_vector = numpy.zeros(word2vec.layer1_size)
                bet_vectors.append(pattern_vector)
        else:
            pattern_vector = numpy.zeros(word2vec.layer1_size)
            bet_vectors.append(pattern_vector)

    """
    for i in range(len(relationships)):
        print relationships[i].sentence
        print relationships[i].between
        print bet_vectors[i]
        print "\n"
    """

    # build a matrix with all the pairwise distances between all the vectors
    print "Building distances matrix"
    matrix = pairwise.pairwise_distances(numpy.array(bet_vectors), metric='cosine', n_jobs=-1)

    # perform DBSCAN
    print "Clustering (DBSCAN)"
    db = DBSCAN(eps=0.1, min_samples=5, metric='precomputed')
    db.fit(matrix)

    # aggregate results by label, discard -1 which is noise
    clusters = defaultdict(list)
    for v in range(0, len(bet_vectors)-1):
        label = db.labels_[v]
        if label > -1:
            clusters[label].append(relationships[v])

    print len(db.labels_), " clusters generated "

    f = open("relationships.txt", "w")

    for k in clusters.keys():
        f.write("cluster: " + str(k) + ' ' + str(len(clusters[k]))+'\n')
        for rel in clusters[k]:
            f.write("cluster: " + str(k)+'\n')
            f.write("ent1: " + rel.ent1 + '\n')
            f.write("ent2: " + rel.ent2 + '\n')
            f.write("BEF:  " + rel.before + '\n')
            f.write("BET:  " + rel.between + '\n')
            f.write("AFT:  " + rel.after + '\n')
            f.write(rel.sentence+'\n')
            f.write("==================\n")
    f.close()

    """
    for rel in sorted(clusters.items(), key=lambda k: len(clusters[k]), reverse=True):
        print "ent1", rel.ent1
        print "ent2", rel.ent2
        print rel.between
        print rel.sentence
        print "\n"
    """

if __name__ == "__main__":
    main()
