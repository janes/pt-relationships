#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk import TreebankWordTokenizer, DependencyGraph
#from nltk.parse import MaltParser

from malt import MaltParser

__author__ = 'dsbatista'


def main():
    parser = MaltParser(working_dir="/home/dsbatista/maltparser-1.8", mco="dep-parser-model3",
                        additional_java_args=['-Xmx512m'])

    txt = "A explosao, que ocorreu pelas 12h00, destruiu quase por completo um edificio de quatro andares do " \
          "complexo da fabrica Stockline Plastics, localizada no centro de Glasgow."

    txt = "O PS apoiou a candidatura de Rui Soares."

    graph = parser.raw_parse(txt)
    graph.tree().pprint()

    sentence = TreebankWordTokenizer().tokenize(txt)
    print sentence
    graph = parser.parse(sentence, verbose=True)
    #print type(graph)
    #print dir(graph)
    print graph.tree()
    print graph.to_conll(10)
    #graph.tree().pprint()
    #graph.tree().draw()

if __name__ == "__main__":
    main()