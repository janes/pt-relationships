#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk import TreebankWordTokenizer, DependencyGraph
from nltk.parse import MaltParser

__author__ = 'dsbatista'


def main():
    parser = MaltParser(working_dir="/home/dsbatista/maltparser-1.8", mco="dep-parser-model1.mco",
                        additional_java_args=['-Xmx512m'])

    txt = "A explosao, que ocorreu pelas 12h00, destruiu quase por completo um edificio de quatro andares do " \
          "complexo da fabrica Stockline Plastics, localizada no centro de Glasgow."

    sentence = TreebankWordTokenizer().tokenize(txt)

    print sentence

    graph = parser.parse(sentence, verbose=True)

    g = DependencyGraph()

    graph.tree().draw()
    graph.tree().pprint()


if __name__ == "__main__":
    main()