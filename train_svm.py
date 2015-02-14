#!/usr/bin/env python
# -*- coding: utf-8 -*-
import operator

__author__ = 'dsbatista'

import codecs
import re
import sys
import StringIO
import pickle
import nltk
import sklearn

from nltk import ngrams
from nltk import bigrams
from nltk import trigrams
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cross_validation import KFold
from Sentence import Relationship, Sentence


N_GRAMS_SIZE = 4

# Parameters for relationship extraction from Sentence
MAX_TOKENS_AWAY = 9
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 3

rel_type_id = dict()
rel_type_id['agreement(Arg1,Arg2)'] = 0
rel_type_id['agreement(Arg2,Arg1)'] = 1
rel_type_id['disagreement(Arg1,Arg2)'] = 2
rel_type_id['disagreement(Arg2,Arg1)'] = 3
rel_type_id['founded-by(Arg1,Arg2)'] = 4
rel_type_id['founded-by(Arg2,Arg1)'] = 5
rel_type_id['hold-shares-of(Arg1,Arg2)'] = 6
rel_type_id['hold-shares-of(Arg2,Arg1)'] = 7
rel_type_id['installations-in(Arg1,Arg2)'] = 8
rel_type_id['installations-in(Arg2,Arg1)'] = 9
rel_type_id['located-in(Arg1,Arg2)'] = 10
rel_type_id['located-in(Arg2,Arg1)'] = 11
rel_type_id['member-of(Arg1,Arg2)'] = 12
rel_type_id['member-of(Arg2,Arg1)'] = 13
rel_type_id['merge'] = 14
rel_type_id['other'] = 15
rel_type_id['owns(Arg1,Arg2)'] = 16
rel_type_id['owns(Arg2,Arg1)'] = 17
rel_type_id['studied-at(Arg1,Arg2)'] = 18
rel_type_id['studied-at(Arg2,Arg1)'] = 19
rel_type_id['work-together'] = 20

id_rel_type = {v: k for k, v in rel_type_id.items()}

#TODO: incluir virgulas na tokenização
TOKENIZER = r'\,|\(|\)|\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.' \
            r'-]+|https?://[^\s]+|\w+'


def load_relationships(data_file):
    relationships = list()
    rel_id = 0
    print "Loading relationships from file"
    #f_sentences = codecs.open(data_file, encoding='utf-8')
    f_sentences = codecs.open(data_file)
    for line in f_sentences:
        if not re.match('^relation', line):
            sentence = line.strip()
        else:
            rel_type = line.strip().split(':')[1]
            rel = Relationship(sentence, None, None, None, None, None, None, None, rel_type, rel_id)

            tokens = re.findall(TOKENIZER, rel.before.decode("utf8"), flags=re.UNICODE)
            rel.before = ' '.join(tokens[-CONTEXT_WINDOW:])
            tokens = re.findall(TOKENIZER, rel.after.decode("utf8"), flags=re.UNICODE)
            rel.after = ' '.join(tokens[:CONTEXT_WINDOW])

            rel_id += 1
            relationships.append(rel)

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


def add_patterns(patterns, reverb_patterns):
    for p in patterns:
        if p not in reverb_patterns:
            reverb_patterns.append(p)


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


def extract_ngrams(text, context):
        chrs = ['_' if c == ' ' else c for c in text]
        return [''.join(g) + '_' + context + ' ' for g in ngrams(chrs, N_GRAMS_SIZE)]


def extract_collocations(text, context):
    bigram_measures = nltk.collocations.BigramAssocMeasures(text)
    trigram_measures = nltk.collocations.TrigramAssocMeasures(text)


def extract_bigrams(text, context):
    tokens = re.findall(TOKENIZER, text, flags=re.UNICODE)
    return [gram[0]+'_'+gram[1]+'_'+context for gram in bigrams(tokens)]


def load_clusters(clusters_file):
    return None


def main():
    global verbs
    print "Loading Label-Delaf"
    verbs_conj = open('verbs/verbs_conj.pkl', "r")
    verbs = pickle.load(verbs_conj)
    verbs_conj.close()

    global tagger
    print "Loading PoS tagger from", sys.argv[1]
    f_model = open(sys.argv[1], "rb")
    tagger = pickle.load(f_model)
    f_model.close()

    """
    global word_clusters
    print "Loading Word Clusters from", sys.argv[3]
    word_clusters = load_clusters(sys.arg[3])
    """

    print "Loading relationships from", sys.argv[3]
    relationships = load_relationships(sys.argv[3])
    print len(relationships), " loaded"

    per_class = dict()
    for rel in relationships:
        try:
            per_class[rel.rel_type] += 1
        except KeyError:
            per_class[rel.rel_type] = 1

    for rel in sorted(per_class.items(), key=operator.itemgetter(1), reverse=True):
        print rel

    # TODO: add TF-IDF words as features
    """
    # extracting tf-idf vectors
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords_pt)
    x_train = vectorizer.fit_transform([rel.sentence for rel in relationships])
    """

    # feature extraction
    all_features = set()
    relationship_features = list()

    for rel in relationships:
        rel_features = list()

        # TODO: first extract all ReVerb patterns, then see which belong to each context
        # extract ReVerb patterns from the 3 contexts
        patterns_bef = extract_patterns(rel.before, "BEF")
        patterns_bet = extract_patterns(rel.between, "BET")
        patterns_aft = extract_patterns(rel.after, "AFT")
        for f in patterns_bef:
            rel_features.append(f)
        for f in patterns_bet:
            rel_features.append(f)
        for f in patterns_aft:
            rel_features.append(f)

        # extract n-grams
        bef_grams = extract_ngrams(rel.before, "BEF")
        bet_grams = extract_ngrams(rel.between, "BET")
        aft_grams = extract_ngrams(rel.after, "AFT")
        for f in bef_grams:
            rel_features.append(f)
        for f in bet_grams:
            rel_features.append(f)
        for f in aft_grams:
            rel_features.append(f)

        # extract_bi_grams
        for f in extract_bigrams(rel.before, "BEF"):
            rel_features.append(f)
        for f in extract_bigrams(rel.before, "BET"):
            rel_features.append(f)
        for f in extract_bigrams(rel.before, "AFT"):
            rel_features.append(f)

        # relationships arguments
        args_type = 'arg1_'+rel.arg1type.strip()+'_arg2_'+rel.arg2type.strip()
        rel_features.append(args_type)

        # append features to relationships features collection
        relationship_features.append(rel_features)

        # add the extracted features to a collection containing all the seen features
        for feature in rel_features:
            all_features.add(feature)

        # TODO: word-clusters generated from word2vec over publico.pt 10 years dataset
        # TODO: normalized verbs from each context

    print len(all_features), " features patterns extracted"

    #TF-IDF Vectorizer
    """
    samples_features = []
    sample_class = []
    relationships_by_id = dict()

    for rel in relationships:
        words = x_train[rel.identifier].toarray()[0]
        samples_features.append(words)
        sample_class.append(rel_type_id[rel.rel_type])
        relationships_by_id[rel.identifier] = rel

    print len(samples_features), " samples"
    print len(rel_type_id), " classes"
    """

    # FeatureHasher
    hasher = sklearn.feature_extraction.FeatureHasher(n_features=len(all_features), non_negative=True, input_type='string')
    samples_features = []
    sample_class = []
    relationships_by_id = dict()

    for rel in relationships:
        features = hasher.fit_transform(relationship_features[rel.identifier])
        samples_features.append(features.toarray()[0])
        sample_class.append(rel_type_id[rel.rel_type])
        relationships_by_id[rel.identifier] = rel

    print len(samples_features), " samples"
    print len(rel_type_id), " classes"

    kf = KFold(len(relationships), 2)
    current_fold = 0
    for train_index, test_index in kf:
        print "\nFOLD", current_fold
        train = []
        train_label = []
        test = []
        test_label = []
        test_ids = []

        for index in train_index:
            train.append(samples_features[index])
            train_label.append(sample_class[index])

        for index in test_index:
            test.append(samples_features[index])
            test_label.append(sample_class[index])
            test_ids.append(index)

        print len(set(train_label)), " classes"

        print "Training...."
        clf = svm.SVC(probability=True)
        clf.fit(train, train_label)
        print "Done"

        print "Testing"
        # compare labels with test_label
        results = clf.predict_proba(test)
        assert len(results) == len(test_label)
        print len(results), "samples classified",

        # Results per class, Precision, Recall, F1
        index_rel = 0
        classifications = list()
        for class_prob in results:
            class_prob_lst = list(class_prob)
            scores = []
            for c_index in range(0, len(class_prob_lst)):
                scores.append((id_rel_type[c_index], class_prob_lst[c_index]))

            sorted_by_score = sorted(scores, key=lambda tup: tup[1], reverse=True)
            classified = sorted_by_score[0][0]
            true_label = relationships_by_id[test_ids[index_rel]].rel_type
            classifications.append((true_label, classified))
            index_rel += 1

        classes_to_annotate = list()
        for rel_type in rel_type_id.keys():
            num_instances_of_class = 0
            num_correct_classified = 0
            num_classified = 0
            num_correct = 0
            for classified in classifications:
                true_label = classified[0]
                classification = classified[1]

                if true_label == rel_type:
                    num_instances_of_class += 1
                    if true_label == classification:
                        num_correct_classified += 1

                if classification == rel_type:
                    num_classified += 1
                if true_label == classification:
                    num_correct += 1

            precision = 1.0 if num_classified == 0 else float(num_correct_classified) / float(num_classified)
            recall = 1.0 if num_instances_of_class == 0 else float(num_correct_classified) / float(num_instances_of_class)
            f1 = 0.0 if precision == 0 and recall == 0 else 2.0 * (precision*recall) / (precision + recall)

            print "Results for class \t" + rel_type + "\t"
            print "Training instances : " + str(per_class[rel_type])
            print "Test instances     : " + str(num_instances_of_class)
            print "Classifications    : " + str(num_classified)
            print "Correct            : " + str(num_correct_classified)
            print "Precision : " + str(precision)
            print "Recall : " + str(recall)
            print "F1 : " + str(f1)
            print "\n"

            if num_classified == 0 or num_correct_classified == 0:
                classes_to_annotate.append(rel_type)

        print "Classes to Annotate"
        for rel_type in classes_to_annotate:
            print rel_type, per_class[rel_type]

        # TODO: overall precision, f1, recall

        # To use in Active Learning Scenario
        """
        index_rel = 0
        for class_prob in results:
            print "rel_id:\t", test_ids[index_rel]
            print "sentence:\t", relationships_by_id[test_ids[index_rel]].sentence
            print "class:\t", relationships_by_id[test_ids[index_rel]].rel_type

            class_prob_lst = list(class_prob)
            scores = []
            for c_index in range(0, len(class_prob_lst)):
                scores.append((id_rel_type[c_index], class_prob_lst[c_index]))

            sorted_by_score = sorted(scores, key=lambda tup: tup[1], reverse=True)
            for t in sorted_by_score:
                print t[0], '\t:', t[1]
            print "\n"
            index_rel += 1
        """
        current_fold+=1

if __name__ == "__main__":
    main()