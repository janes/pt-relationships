#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import os
import re
import subprocess
import codecs
import sys
import numpy as np

import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split

from nltk.parse import DependencyGraph

from gensim.models import KeyedVectors

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)


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


class Relationship:

    def __init__(self, sentence, ent1, ent2, rel_type, syntaxnet_info=None):
        self.sentence = sentence
        self.rel_type = rel_type
        self.ent1 = ent1
        self.ent2 = ent2
        self.syntaxnet_info = syntaxnet_info
        self.ent1_begin = None
        self.ent1_end = None
        self.ent2_begin = None
        self.ent2_end = None
        self.before_context = None
        self.between_context = None
        self.after_context = None


def load_relationships(data_file):
    """
    Reads relationships from a given file and returns a
    list (rel_type, sentence) tuples

    :param data_file:
    :return:
    """

    relationships = list()
    f_sentences = codecs.open(data_file, encoding='utf-8')

    for line in f_sentences:
        if not line.startswith("relation:"):
            sentence = line.strip()
        else:
            rel_type = line.strip().split(':')[1]
            entities = []
            for m in re.finditer(regex, sentence):
                entities.append(m.group())

            rel = Relationship(sentence, entities[0], entities[1], rel_type)
            relationships.append(rel)

    f_sentences.close()

    return relationships


def get_path_up_to_root(path, tree, node):
    while int(node[6]) != 0:
        node = tree[int(node[6])-1]
        path.append(node)
    return path


def extract_syntactic_path(rel):
    """
    1  ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
    2  FORM: Word form or punctuation symbol.
    3  LEMMA: Lemma or stem of word form.
    4  UPOSTAG: Universal part-of-speech tag.
    5  XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    6  FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    7  HEAD: Head of the current word, which is either a value of ID or zero (0)
    8  DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    9  DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    10 MISC: Any other annotation.
    """

    # join the strings in CONLL format and pass to the DependencyGraph class
    s = ''
    for e in rel.syntaxnet_info:
        s += "\t".join(e) + '\n'

    rel.dependency_graph = DependencyGraph(tree_str=s.decode("utf8"))

    path = [rel.syntaxnet_info[rel.ent1_end]]
    ent1_path = get_path_up_to_root(path, rel.syntaxnet_info, rel.syntaxnet_info[rel.ent1_end])

    path = [rel.syntaxnet_info[rel.ent2_end]]
    ent2_path = get_path_up_to_root(path, rel.syntaxnet_info, rel.syntaxnet_info[rel.ent2_end])

    ent1 = [(x[0], x[1], x[3], x[6], x[7]) for x in ent1_path]
    ent2 = [(x[0], x[1], x[3], x[6], x[7]) for x in ent2_path]

    # find common nodes
    path = [val for val in ent1 if val in ent2]

    # extract the syntactic path between the two named entities
    full_path = []

    # case_1) where the ROOT node is part of the common nodes
    #         add everything from ROOT until entity is found direction ent_1
    #         add everything from ROOT until entity is found direction ent_2
    if path[0][4] == 'ROOT':
        # start at ROOT node
        for node in ent1[:-1]:
            if node[1].strip(",.").decode("utf8") not in rel.ent1_parts:
                full_path.append(node)

        for node in reversed(ent2):
            if node[1].strip(",.").decode("utf8") not in rel.ent2_parts:
                full_path.append(node)

    # case_2) where the ROOT node is not part of the common nodes
    else:
        for node in ent1:
            if node == path[0]:
                if node[1].strip(",.").decode("utf8") not in rel.ent1_parts:
                    full_path.append(node)
                break
            else:
                if node[1].strip(",.").decode("utf8") not in rel.ent1_parts:
                    full_path.append(node)

        found_common = False

        for node in reversed(ent2):
            if found_common:
                if node[1].strip(",.").decode("utf8") not in rel.ent2_parts:
                    full_path.append(node)
            if node == path[0]:
                found_common = True

    rel.syntactic_path = full_path


def get_contexts(rel):
    """
    identifies the BEFORE, BETWEEN and AFTER context in a sentence
    :param rel:
    :return:
    """

    ent1_parts = get_entity_parts(rel.ent1)
    ent2_parts = get_entity_parts(rel.ent2)

    rel.ent1_begin, rel.ent1_end = get_entity_position(ent1_parts, rel)
    rel.ent2_begin, rel.ent2_end = get_entity_position(ent2_parts, rel)

    rel.ent1_parts = ent1_parts
    rel.ent2_parts = ent2_parts

    rel.before_context = rel.syntaxnet_info[0:rel.ent1_begin]
    rel.between_context = rel.syntaxnet_info[rel.ent1_end + 1:rel.ent2_begin]
    rel.after_context = rel.syntaxnet_info[rel.ent2_end + 1:]


def get_entity_position(ent_parts, rel):
    """
    gets an entity position in a sentence as processed by SyntaxNet
    :param ent_parts:
    :param rel:
    :return:
    """

    z = 0
    begin = 0
    end = 0

    for i in range(len(rel.syntaxnet_info)):
        if rel.syntaxnet_info[i][1].strip(",.") == ent_parts[z].encode("utf8"):

            # entity is a single token
            if z == 0 and len(ent_parts) == 1:
                begin = end = i
                break

            if z == 0:
                begin = i
                z += 1

            elif z == len(ent_parts)-1:
                end = i
                break

            else:
                z += 1

    return begin, end


def get_entity_parts(entity):
    clean = re.sub("</?[A-Z]+>", "", entity)
    entity_parts = clean.split()
    return entity_parts


def read_syntaxnet_output(sentences):

    # joint all sentences into a single string
    all_sentences = "\n".join(sentences)

    # redirect std_error to /dev/null
    FNULL = open(os.devnull, 'w')

    process = subprocess.Popen(
        'MODEL_DIRECTORY=/Users/dbatista/Downloads/Portuguese; '
        'cd /Users/dbatista/models/syntaxnet; '
        'echo \'%s\' | syntaxnet/models/parsey_universal/parse.sh '
        '$MODEL_DIRECTORY 2' % all_sentences,
        shell=True,
        universal_newlines=False,
        stdout=subprocess.PIPE,
        stderr=FNULL)

    output = process.communicate()
    processed_sentences = []
    sentence = []

    for line in output[0].split("\n"):
        if len(line) == 0:
            processed_sentences.append(sentence)
            sentence = []
        else:
            word = line.split("\t")
            sentence.append(word)

    for sentence in processed_sentences:
        for line in sentence:
            print line
        print

    print len(processed_sentences)

    f_out = open("processed_sentences.pkl", "w")
    pickle.dump(processed_sentences, f_out)
    f_out.close()


def extract_features(rel):
    """
    Contexts
    ========

    BEFORE:
        - the words before and single word after E1 and E2 respectively

    BETWEEN:
        - the words
        - parts of speech
        - stemming ?
        - the number of words
        - ReVerb pattern

    AFT:
        - the words before and single word after E1 and E2 respectively

    Syntactic Dependencies Trees
    ============================
    - reduce each relation example to the smallest subtree in the parse or
      dependency tree that includes both entities.
      e.g.:
        s = ''
        for e in rel.syntaxnet_info:
            s += "\t".join(e) + '\n'
        graph = DependencyGraph(tree_str=s.decode("utf8"))
        tree = graph.tree()
        tree.pretty_print()
        for t in tree.subtrees():
            t.pretty_print()
    """

    rel.words_between = [x[1] for x in rel.between_context]
    rel.pos_between = [x[3] for x in rel.between_context]
    rel.words_syntactic_path = [x[1] for x in rel.syntactic_path]
    rel.pos_syntactic_path = [x[2] for x in rel.syntactic_path]

    """
    print
    print rel.sentence
    print rel.rel_type
    print
    print "BET:"
    print "nodes_between:", rel.between_context
    print "pos_between:", rel.pos_between
    print "word_beteen:", rel.words_between
    print
    print "Syntactic Path:"
    print "words", rel.words_syntactic_path
    print "syntactic", rel.pos_syntactic_path
    """

    # LABEL-LEX-sw - Portuguese Formalized Lexicon


    # TODO:
    # check if:
    #    verbs are modified by negative polarity adverbs (nunca, jamais, etc.)
    #    nouns are modified by negative determiners (nao, etc.)

    # rel.dependency_graph.tree().pretty_print()

    """
    triples = sorted([t for t in rel.dependency_graph.triples()])
    for t in triples:
        print t
    """


    """
    features:
        1: words_between
        2: pos_between

        1: words_syntactic_path
        2: pos_syntactic_path
    """



    features = []
    for x in range(len(rel.syntactic_path)):
        word = rel.syntactic_path[x][1]
        pos = rel.syntactic_path[x][2]
        syntactic_dep = rel.syntactic_path[x][4]
        features.append((word, pos, syntactic_dep))
    rel.shortest_dep_features = features


def compare_word_classes(a, b):
    return len(set(a).intersection(set(b)))


def shortest_path_kernel(rel_a, rel_b):

    """
    Shortest-Path Dependency Kernel
    https://www.cs.utexas.edu/~ml/papers/spk-emnlp-05.pdf

    x = [x1 x2 x3 x4 x5 x6 x7]
    x1 = {his, PRP, PERSON},
    x2 = {→},
    x3 = {actions, NNS, Noun},
    x4 = {←},
    x5 = { in, IN},
    x6 = {←},
    x7 = {Brcko, NNP, Noun, LOCATION}

    y = [y1 y2 y3 y4 y5 y6 y7]
    y1 = {his, PRP, PERSON},
    y2 = {→},
    y3 = {arrival, NN, Noun},
    y4 = {←},
    y5 = { in, IN},
    y6 = {←},
    y7 = {Beijing, NNP, Noun, LOCATION}

    x1,y1 -> x3
    x2,y2 -> x1
    x3,y3 -> x1
    x4,y4 -> y1
    x5,y5 -> x2
    x6,y6 -> x1
    x7,y7 -> x3

    K(x, y) = 3×1×1×1×2×1×3 = 18

    """
    if len(rel_a.shortest_dep_features) != len(rel_b.shortest_dep_features):
        return 0
    else:
        sim_score = 1
        for x in range(len(rel_a.shortest_dep_features)):
            sim_score *= compare_word_classes(
                rel_a.shortest_dep_features[x],
                rel_b.shortest_dep_features[x]
            )

        """
        print rel_a.shortest_dep_features
        print rel_b.shortest_dep_features
        print sim_score
        print
        """
        return sim_score


def compute_gram_matrix(x_train):

    # compute gram matrix
    gram_matrix = np.zeros(shape=(len(x_train), len(x_train)))

    x = x_train.as_matrix().tolist()

    for i in range(len(x)):
        for j in range(len(x)):
            sim = shortest_path_kernel(x[i], x[j])
            gram_matrix[i, j] = sim
    return gram_matrix


def main():

    # load training data
    relationships = load_relationships(sys.argv[1])

    """
    # process sentences with SyntaxNet for Portuguese
    sentences = [re.sub(r"</?[A-Z]+>", "", x.sentence.strip()) for x in relationships]
    read_syntaxnet_output(sentences)
    """

    # read sentences already processed with SyntaxNet
    f_out = open("processed_sentences.pkl", "r")
    sentences_processed = pickle.load(f_out)

    data = []

    for x in range(0, len(relationships)):
        relationships[x].syntaxnet_info = sentences_processed[x]
        get_contexts(relationships[x])
        extract_syntactic_path(relationships[x])
        extract_features(relationships[x])
        data.append((relationships[x], relationships[x].rel_type))

    data_frame = pd.DataFrame(data, columns=["relationship", "rel_type"])
    x = data_frame['relationship']

    y = []
    for rel_type in data_frame[['rel_type']].values.tolist():
        y.append(rel_type_id[rel_type[0]])
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # train
    train_gram_matrix = compute_gram_matrix(x_train)
    print "train"
    # print train_gram_matrix.shape, type(train_gram_matrix)
    # print y_train.shape, type(y_train)
    svm = SVC(kernel='precomputed', decision_function_shape='ovo')
    svm.fit(train_gram_matrix, y_train)
    # print svm.classes_
    # http://stackoverflow.com/questions/35022270/which-support-vectors-returned-in-multiclass-svm-sklearn

    # test
    print "test"
    # http://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
    test_gram_matrix = compute_gram_matrix(x_test)
    predictions = svm.predict(test_gram_matrix)

    print predictions, len(predictions)
    print y_test, len(y_test)

    # ./process_sentences.py train_data.txt `jot -r 1  0 1083`

if __name__ == "__main__":
    main()
