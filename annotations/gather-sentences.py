#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import codecs
from Sentence import Sentence

tokenizer =  r'\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.-]+|https?://[^\s]+|\w+'
CONTEXT_WINDOW = 4


class RelationshipType:
    def __init__(self, _name, _arg1, _arg2, _words):
        self.name = _name
        self.args1 = _arg1
        self.args2 = _arg2
        self.words = _words


def read_sentiwords(data):
    words = list()
    f = codecs.open(data, encoding='utf-8')
    for line in f:
        words.append(line.strip())
    f.close()
    return words


def read_seed_words(data):
    relationships = list()
    f = codecs.open(data, encoding='utf-8')
    for line in f:
        if line.startswith('relation'):
            rel = line.split(":")[1].strip()

        if line.startswith('arg1'):
            tmp = line.split(":")[1].split(",")
            args1 = [x.strip() for x in tmp]

        if line.startswith('arg2'):
            tmp = line.split(":")[1].split(",")
            args2 = [x.strip() for x in tmp]

        if line.startswith('words'):
            tmp_words = line.split(":")[1].split(",")
            words = [x.strip() for x in tmp_words]

            # build Relationship object and apend it to the list
            relation = RelationshipType(rel, args1, args2, words)
            relationships.append(relation)
    f.close()
    return relationships


def find_matches(line, relationships, positive=None, negative=None):
    sentence = Sentence(line)

    for rel in sentence.relationships:
        # tokenize words in bef,bet,aft
        bef = re.findall(tokenizer, rel.before .lower(), flags=re.UNICODE)
        bet = re.findall(tokenizer, rel.between.lower(), flags=re.UNICODE)
        aft = re.findall(tokenizer, rel.after.lower(), flags=re.UNICODE)

        if len(bet) >= 9:
            continue

        # conside only a window of size 'tokens_window' tokens
        before_tokens = bef[0 - CONTEXT_WINDOW:]
        after_tokens = aft[:CONTEXT_WINDOW]

        # try to match with any seed words
        tokens = before_tokens + bet + after_tokens

        for r in relationships:
            # check if the arguments type match
            if rel.arg1type in r.args1 and rel.arg2type in r.args2:
                if len(set(tokens).intersection(set(r.words))) > 0:
                    print r.name.encode("utf8")
                    #print " ".join(r.words).encode("utf8")
                    print rel.ent1.encode("utf8")
                    print rel.ent2.encode("utf8")
                    print "sentence:", line.encode("utf8")

        if positive is not None and negative is not None:
            if len(set(tokens).intersection(set(positive))) > 0:
                print "positive:", set(tokens).intersection(set(positive))
                print rel.ent1
                print rel.ent2
                print line.encode("utf8")

            if len(set(tokens).intersection(set(negative))) > 0:
                print "negative:", set(tokens).intersection(set(negative))
                print rel.ent1
                print rel.ent2
                print line.encode("utf8")


def main():
    ent = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    relationships = read_seed_words(sys.argv[1])
    #positive = read_sentiwords("positive_nouns.txt")
    #negative = read_sentiwords("negative_nouns.txt")

    # read already tagged sentences and store in a set
    tagged = set()
    to_tag = set()
    f = codecs.open(sys.argv[2], encoding='utf-8')
    for line in f:
        if not line.startswith("relation:"):
            clean = re.sub(r"</?[A-Z]+>", "", line.strip())
            tagged.add(clean)
    f.close()

    print len(tagged), "annotated sentences"

    # select sentences to tag which are not already tagged
    f = codecs.open(sys.argv[3], encoding='utf-8')
    for line in f:
        matches = ent.findall(line)
        clean = re.sub(r"</?[A-Z]+>", "", line)
        if clean not in tagged:
            # restringir numero de entidades e tamanho da frase
            if len(matches) == 2 and len(clean) <= 200:
                #to_tag.add(line.encode("utf-8"))
                find_matches(line, relationships, None, None)
    f.close()


if __name__ == "__main__":
    main()