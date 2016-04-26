#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import nltk

from nltk import Tree
from nltk.corpus import floresta
from nltk.grammar import ProbabilisticProduction, Nonterminal


def corpus2trees(text):
    """ Parse the corpus and return a list of Trees """
    rawparses = text.split("\n\n")
    trees = []

    for rp in rawparses:
        if not rp.strip():
            continue

        try:
            t = Tree.parse(rp)
            trees.append(t)
        except ValueError:
            logging.error('Malformed parse: "%s"' % rp)
    return trees


def trees2productions(trees):
    """ Transform list of Trees to a list of productions """
    productions = []
    for t in trees:
        productions += t.productions()
    return productions


class PCFGViterbiParser(nltk.ViterbiParser):
    def __init__(self, grammar, trace=0):
        super(PCFGViterbiParser, self).__init__(grammar, trace)

    @staticmethod
    def _preprocess(tokens):
        replacements = {
            "(": "-LBR-",
            ")": "-RBR-",
        }
        for idx, tok in enumerate(tokens):
            if tok in replacements:
                tokens[idx] = replacements[tok]

        return tokens

    @classmethod
    def train(cls, content, root):
        """
        if not isinstance(content, basestring):
            content = content.read()

        trees = corpus2trees(content)
        productions = trees2productions(trees)
        pcfg = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal(root), productions)
        return cls(pcfg)
        """
        pcfg = nltk.grammar.induce_pcfg(nltk.grammar.Nonterminal(root), productions)

    def parse(self, tokens):
        tokens = self._preprocess(list(tokens))
        tagged = nltk.pos_tag(tokens)

        missing = False
        for tok, pos in tagged:
            if not self._grammar._lexical_index.get(tok):
                missing = True
                self._grammar._productions.append(ProbabilisticProduction(Nonterminal(pos), [tok], prob=0.000001))
        if missing:
            self._grammar._calculate_indexes()

        return super(PCFGViterbiParser, self).parse(tokens)


def main():

    sentences = floresta.tagged_sents()
    print len(sentences)
    viterbi_parser = PCFGViterbiParser.train(sentences, root='ROOT')

    #viterbi_parser = PCFGViterbiParser.train(open('corpus.txt', 'r'), root='ROOT')
    #t = viterbi_parser.parse(nltk.word_tokenize('Numerous passing references to the phrase have occurred in movies'))
    t = viterbi_parser.parse(nltk.word_tokenize('O Noam Chomsky é professor de linguística no MIT.'))
    print t
    t.draw()


if __name__ == "__main__":
    main()