#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import codecs
import sys
import re

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"


regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)


class Relationship:

    def __init__(self, sentence, ent1, ent2, rel_type):
        self.sentence = sentence
        self.rel_type = rel_type
        self.ent1 = ent1
        self.ent2 = ent2


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
    pass


def read_syntaxnet_output(sentences):

    #print len(sentences)
    #exit(-1)

    all_sentences = "\n".join(sentences[0:1100])

    process = subprocess.Popen(
        """echo '%s'""" % all_sentences,
        shell=True,
        universal_newlines=False,
        stdout=subprocess.PIPE)

    output = process.communicate()

    process = subprocess.Popen(
        'MODEL_DIRECTORY=/Users/dbatista/Downloads/Portuguese; '
        'cd /Users/dbatista/models/syntaxnet; '
        'echo \'%s\' | syntaxnet/models/parsey_universal/parse.sh '
        '$MODEL_DIRECTORY 2' % all_sentences,
        shell=True,
        universal_newlines=False,
        stdout=subprocess.PIPE)

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


    """
    # find ROOT verb
    for word in sentence:
        if len(word) == 1:
            continue
        if word[7] == 'ROOT' and word[3] == 'VERB':
            print word
    """


def main():
    relationships = load_relationships(sys.argv[1])
    sentences = list(set([re.sub(r"</?[A-Z]+>", "", x.sentence.strip())
                          for x in relationships]))
    read_syntaxnet_output(sentences)

if __name__ == "__main__":
    main()
