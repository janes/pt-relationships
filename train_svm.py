#!/usr/bin/env python
# -*- coding: utf-8 -*-
from common.Sentence import Sentence, Relationship

__author__ = 'dsbatista'

import codecs
import re
import sys
import pickle
import sklearn
import operator
import fileinput

from collections import defaultdict
from nltk import ngrams
from nltk import bigrams
from sklearn import svm
from sklearn.cross_validation import KFold
from math import log
from common.ReVerbPT import ReverbPT


# Parameters for relationship extraction from Sentence
MAX_TOKENS_AWAY = 9
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 3
N_GRAMS_SIZE = 4

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

TOKENIZER = r'\,|\(|\)|\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.' \
            r'-]+|https?://[^\s]+|\w+'


def load_relationships(data_file):
    tagged = set()
    relationships = list()
    rel_id = 0
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        # read the lines without tagged entities, this is used later
        # to compare with sentences to tag, and assure no training sentences
        # are given to tag
        if not line.startswith("relation:"):
            clean = re.sub(r"</?[A-Z]+>", "", line.strip())
            tagged.add(clean)

        if not re.match('^relation', line):
            sentence = line.strip()
        else:
            rel_type = line.strip().split(':')[1]
            rel = Relationship(sentence, None, None, None, None, None, None, None, rel_type, rel_id)
            tokens = re.findall(TOKENIZER, rel.before, flags=re.UNICODE)
            rel.before = ' '.join(tokens[-CONTEXT_WINDOW:])
            tokens = re.findall(TOKENIZER, rel.after, flags=re.UNICODE)
            rel.after = ' '.join(tokens[:CONTEXT_WINDOW])
            rel_id += 1
            relationships.append(rel)

    return relationships, tagged


def extract_ngrams(text, context):
        chrs = ['_' if c == ' ' else c for c in text]
        return [''.join(g) + '_' + context + ' ' for g in ngrams(chrs, N_GRAMS_SIZE)]


def extract_bigrams(text, context):
    tokens = re.findall(TOKENIZER, text, flags=re.UNICODE)
    return [gram[0]+'_'+gram[1]+'_'+context for gram in bigrams(tokens)]


def load_clusters(clusters_file):
    clusters = dict()
    words = defaultdict(list)
    for line in fileinput.input(clusters_file):
        word, cluster = line.strip().split()
        clusters[word] = int(cluster)
        words[int(cluster)].append(word)
    fileinput.close()
    return clusters, words


def shannon_entropy(probabilities):
    entropy = 0
    for i in probabilities:
        entropy += i * log(i)

    return entropy


def feature_extraction(reverb, clusters_words, relationships, verbs, word_cluster):
    # feature extraction
    all_features = set()
    relationship_features = list()
    for rel in relationships:
        rel_features = list()

        # TODO: extract Pos-tags for each context
        #patterns_bef = extract_patterns(rel.before, "BEF")
        #patterns_bet = extract_patterns(rel.between, "BET")
        #patterns_aft = extract_patterns(rel.after, "AFT")

        # TODO: normalized verbs/substântivos from each context
        # extract all words from the contexts
        #tokens_bef = re.findall(TOKENIZER, rel.before, flags=re.UNICODE)
        #tokens_bet = re.findall(TOKENIZER, rel.between, flags=re.UNICODE)
        #tokens_aft = re.findall(TOKENIZER, rel.after, flags=re.UNICODE)

        #for f in tokens_bef+tokens_bet:
        #    rel_features.append(f)

        #for f in tokens_bet:
        #    rel_features.append(f)

        #for f in tokens_bet+tokens_aft:
        #    rel_features.append(f)

        # extract ReVerb patterns from BET context
        # extract 2 words from BEF and AFT
        patterns_bet = reverb.extract_reverb_patterns_ptb(rel.between)

        for p in patterns_bet:
            verb = p.split('_')[0]
            try:
                inf = verbs[verb]
                if inf not in ['ser', 'estar', 'ter', 'haver', 'ficar', 'ir']:
                    for word in clusters_words[int(word_cluster[verb])]:
                        if word in verbs[verb]:
                            rel_features.append(word)
            except KeyError:
                pass

        for f in patterns_bet:
            rel_features.append(f)

        # extract n-grams of characters
        bef_grams = extract_ngrams(rel.before, "BEF")
        bet_grams = extract_ngrams(rel.between, "BET")
        aft_grams = extract_ngrams(rel.after, "AFT")
        for f in bef_grams:
            rel_features.append(f)
        for f in bet_grams:
            rel_features.append(f)
        for f in aft_grams:
            rel_features.append(f)

        # extract_bi_grams of words
        for f in extract_bigrams(rel.before, "BIGRAMS_BEF"):
            rel_features.append(f)
        for f in extract_bigrams(rel.between, "BIGRAMS_BET"):
            rel_features.append(f)
        for f in extract_bigrams(rel.after, "BIGRAMS_AFT"):
            rel_features.append(f)

        # relationships arguments
        args_type = 'arg1_' + rel.arg1type.strip() + '_arg2_' + rel.arg2type.strip()
        rel_features.append(args_type)

        # append features to relationships features collection
        relationship_features.append(rel_features)

        """
        print rel.arg1type
        print rel.arg2type
        print rel.sentence.encode("utf8")
        print rel_features
        print "\n"
        """

        # add the extracted features to a collection containing all the seen features
        for feature in rel_features:
            all_features.add(feature)

    print len(all_features), " features extracted"

    return all_features, relationship_features


def cross_validation(per_class, relationships, relationships_by_id, sample_class, samples_features):
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
            recall = 1.0 if num_instances_of_class == 0 else float(num_correct_classified) / float(
                num_instances_of_class)
            f1 = 0.0 if precision == 0 and recall == 0 else 2.0 * (precision * recall) / (precision + recall)

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
        current_fold += 1


def train_classifier(sample_class, samples_features):
    print "Training...."
    clf = svm.SVC(C=1.0, probability=True, verbose=False)
    clf.fit(samples_features, sample_class)
    print "Done"
    return clf


def classify(classifier, relationships_pool, hasher):
        all_features, relationship_features = feature_extraction(clusters_words, relationships_pool,
                                                                 verbs, word_cluster)
        samples_features = []

        print "Hashing features"
        for rel in range(len(relationships_pool)):
            features = hasher.fit_transform(relationship_features[rel])
            samples_features.append(features.toarray()[0])

        # calculate classifications for each class
        results = classifier.predict_proba(samples_features)

        # calculate entropy
        index_rel = 0
        for class_prob in results:
            print "sentence:\t", relationships_pool[index_rel].sentence
            entropy = shannon_entropy(list(class_prob))
            print "entropy:\t", entropy

            class_prob_lst = list(class_prob)
            scores = []
            for c_index in range(0, len(class_prob_lst)):
                scores.append((id_rel_type[c_index], class_prob_lst[c_index]))

            sorted_by_score = sorted(scores, key=lambda tup: tup[1], reverse=True)
            for t in sorted_by_score:
                print t[0], '\t:', t[1]
            print "\n"
            index_rel += 1


def main():

    global tagger
    print "Loading PoS tagger from", sys.argv[1]
    f_model = open(sys.argv[1], "rb")
    tagger = pickle.load(f_model)
    f_model.close()
    reverb = ReverbPT(tagger)

    global word_cluster
    global clusters_words
    print "Loading Word Clusters from", sys.argv[2]
    word_cluster, clusters_words = load_clusters(sys.argv[2])

    print "Loading relationships from", sys.argv[3]
    relationships, tagged = load_relationships(sys.argv[3])
    print len(relationships), " loaded"

    global verbs
    print "Loading Label-Delaf"
    verbs_conj = open('verbs/verbs_conj.pkl', "r")
    verbs = pickle.load(verbs_conj)
    verbs_conj.close()



    # print number of samples per class
    per_class = dict()
    for rel in relationships:
        try:
            per_class[rel.rel_type] += 1
        except KeyError:
            per_class[rel.rel_type] = 1

    for rel in sorted(per_class.items(), key=operator.itemgetter(1), reverse=True):
        print rel

    # TODO: add TF-IDF words as features
    # extracting tf-idf vectors
    #stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords_pt)
    #x_train = vectorizer.fit_transform([rel.sentence for rel in relationships])

    all_features, relationship_features = feature_extraction(reverb, clusters_words, relationships, verbs, word_cluster)

    # Feature Hashing

    #TF-IDF Vectorizer
    #samples_features = []
    #sample_class = []
    #relationships_by_id = dict()

    #for rel in relationships:
    #    words = x_train[rel.identifier].toarray()[0]
    #    samples_features.append(words)
    #    sample_class.append(rel_type_id[rel.rel_type])
    #    relationships_by_id[rel.identifier] = rel

    #print len(samples_features), " samples"
    #print len(rel_type_id), " classes"

    #FeatureHasher
    hasher = sklearn.feature_extraction.FeatureHasher(n_features=len(all_features), non_negative=True,
                                                      input_type='string')
    samples_features = []
    sample_class = []
    relationships_by_id = dict()

    print "Hashing features"
    for rel in relationships:
        features = hasher.fit_transform(relationship_features[rel.identifier])
        samples_features.append(features.toarray()[0])
        sample_class.append(rel_type_id[rel.rel_type])
        relationships_by_id[rel.identifier] = rel

    print len(samples_features), " samples"
    print len(rel_type_id), " classes"

    #hashing = HashingVectorizer(non_negative=True, norm=None)
    #tfidf = TfidfTransformer()
    #hashing_tfidf = Pipeline([("hashing", hashing), ("tidf", tfidf)])

    if sys.argv[4] == 'fold':
        cross_validation(per_class, relationships, relationships_by_id, sample_class, samples_features)

    elif sys.argv[4] == 'model':
        classifier = train_classifier(sample_class, samples_features)

        # create a pool of 1.0000 sentences
        relationships_pool = list()
        # read sentences to annotate from file
        f = open(sys.argv[5], 'r')
        ent = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
        for line in f:
            matches = ent.findall(line)
            clean = re.sub(r"</?[A-Z]+>", "", line)
            # make sure the sentence different from the ones in the training set
            if clean not in tagged:
                # restrict number of entities and sentence length
                if len(matches) == 2 and len(clean) <= 200 > 40:
                    sentence = Sentence(line)
                    for rel in sentence.relationships:
                        relationships_pool.append(rel)
                if len(relationships_pool) > 1000:
                    break

        # classify the sentences
        classify(classifier, relationships_pool, hasher)

if __name__ == "__main__":
    main()