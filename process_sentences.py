#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import re
import subprocess
import codecs
import sys

from nltk.stem.wordnet import WordNetLemmatizer

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)


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


def extract_features(rel):
    """

    BETWEEN:
        - the words
        - parts of speech

        - prefixes of length 5
        - words between the nominals can be strong indicators for the
        type of relation, using the prefixes of length 5 for the words
        between the nominals provides a kind of stemming

        - extract a coarse-grained part of speech sequence for the words
        between the nominals; a string using the first letter of each
        token’s Treebank POS tag. This feature is motivated by the fact that
        relations such as Member-Collection usually invoke prepositional
        phrases such as:  of, in the, and of various.
        The corresponding POS sequences we extract are: “I”, “I D”, and “I J”.

        - Finally, we also use the number of words between the nominals as a
        feature because relations such as Product-Producer and Entity-Origin
        often have no intervening tokens (e.g., organ builder or Coconut oil).

    BEFORE:
        - the words before and single word after E1 and E2 respectively

    """

    BEF = rel.syntaxnet_info[0:rel.ent1_begin]
    BET = rel.syntaxnet_info[rel.ent1_end+1:rel.ent2_begin]
    AFT = rel.syntaxnet_info[rel.ent2_end+1:]

    print rel.sentence
    print

    for word_info in BEF:
        print word_info[1],
    print

    print

    for word_info in BET:
        print word_info[1],
    print
    print

    for word_info in AFT:
        print word_info[1],
    print
    print


def syntactic_path(sentence):
    """
    1  ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
    2  FORM: Word form or punctuation symbol.
    3  LEMMA: Lemma or stem of word form.
    4  UPOSTAG: Universal part-of-speech tag.
    5  XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    6  FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    7  HEAD: Head of the current word, which is either a value of ID or zero (0).
    8  DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    9  DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    10 MISC: Any other annotation.

    :param sentence:
    :return:
    """


def get_contexts(rel):
    ent1_parts = get_entity_parts(rel.ent1)
    ent2_parts = get_entity_parts(rel.ent2)
    rel.ent1_begin, rel.ent1_end = get_entity_position(ent1_parts, rel)
    rel.ent2_begin, rel.ent2_end = get_entity_position(ent2_parts, rel)
    rel.ent1_parts = ent1_parts
    rel.ent2_parts = ent2_parts
    return rel


def get_entity_position(ent_parts, rel):

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

    """
    # find ROOT verb
    for word in sentence:
        if len(word) == 1:
            continue
        if word[7] == 'ROOT' and word[3] == 'VERB':
            print word
    """


def lemmatize():
    lmtzr = WordNetLemmatizer()
    lmtzr.lemmatize("owns", "v")
    lmtzr.lemmatize("owned", "v")


def main():
    relationships = load_relationships(sys.argv[1])

    """
    sentences = [re.sub(r"</?[A-Z]+>", "", x.sentence.strip())
                 for x in relationships]
    read_syntaxnet_output(sentences)
    """

    f_out = open("processed_sentences.pkl", "r")
    sentences_processed = pickle.load(f_out)

    for x in range(0, len(relationships)):
        relationships[x].syntaxnet_info = sentences_processed[x]

    rel = get_contexts(relationships[int(sys.argv[2])])

    # TODO: write tests
    #  len(both entities) = 1
    #     - ent1: middle of sentence
    #     - ent2: end of sentence -> 830
    #
    #  len(both entities) = 1
    #     - ent1: middle of sentence
    #     - ent2: middle of sentence
    #     - len(BET) = 1
    #                       16
    # len(ent2) = 1
    #     - ent1: begin of sentence -> 10

    # TODO: corrigir entidades
    # <ORG>Instituto de Apoio à Criança</ORG> ( IAC )
    # <ORG>Instituto de Apoio à Criança (IAC)</ORG>

    # ./process_sentences.py train_data.txt `jot -r 1  0 1083`

    extract_features(rel)
    # TODO: para visualizar a dep-tree procurar um CONLL2GraphTree
    """
    for word_info in rel.syntaxnet_info:
        print word_info[0], word_info[1], word_info[3], word_info[6], word_info[7]
    """

if __name__ == "__main__":
    main()
