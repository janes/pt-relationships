#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
import sys

from nltk import DependencyGraph


def read_syntaxnet_output(sentences):

    # joint all sentences into a single string with
    # separating new lines
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

    # subprocess captures an empty new line
    del processed_sentences[-1]

    deps = []
    for sentence in processed_sentences:
        s = ''
        for line in sentence:
            s += "\t".join(line) + '\n'
        deps.append(s)

    for sent_dep in deps:
        graph = DependencyGraph(tree_str=sent_dep.decode("utf8"))
        print "triples"
        for triple in graph.triples():
            print triple
        print
        tree = graph.tree()
        tree.pretty_print()


def main():
    with open(sys.argv[1], 'r') as f:
        data = f.readlines()
        sentences = [x.strip() for x in data]
    read_syntaxnet_output(sentences)


if __name__ == "__main__":
    main()
