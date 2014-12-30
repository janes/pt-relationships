#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

import sys
import codecs
import re
from FeatureExtractor import FeatureExtractor

from Sentence import Relationship, Sentence

rel_type_id = dict()
rel_type_id['agreement(Arg1,Arg2)'] = 1
rel_type_id['agreement(Arg2,Arg1)'] = 2
rel_type_id['disagreement(Arg1,Arg2)'] = 3
rel_type_id['disagreement(Arg2,Arg1)'] = 4
rel_type_id['founded-by(Arg1,Arg2)'] = 5
rel_type_id['founded-by(Arg2,Arg1)'] = 6
rel_type_id['hold-shares-of(Arg1,Arg2)'] = 7
rel_type_id['hold-shares-of(Arg2,Arg1)'] = 8
rel_type_id['installations-in(Arg1,Arg2)'] = 9
rel_type_id['installations-in(Arg2,Arg1)'] = 10
rel_type_id['located-in(Arg1,Arg2)'] = 11
rel_type_id['located-in(Arg2,Arg1)'] = 12
rel_type_id['member-of(Arg1,Arg2)'] = 13
rel_type_id['member-of(Arg2,Arg1)'] = 14
rel_type_id['merge'] = 15
rel_type_id['other'] = 16
rel_type_id['owns(Arg1,Arg2)'] = 17
rel_type_id['owns(Arg2,Arg1)'] = 18
rel_type_id['studied-at(Arg1,Arg2)'] = 19
rel_type_id['studied-at(Arg2,Arg1)'] = 20
rel_type_id['work-together'] = 21

TOKENIZER = r'\,|\(|\)|\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.' \
            r'-]+|https?://[^\s]+|\w+'


def training(data_file, extractor, f):
    sentence = None
    rel_id = 0
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if not re.match('^relation', line):
            sentence = line.strip()
        else:
            rel_type = line.strip().split(':')[1]
            rel = Relationship(sentence, None, None, None, None, None, None, None, rel_type, rel_id)

            # relationship type
            f.write(str(rel_type_id[rel.rel_type]))

            # relationship identifier
            f.write(" 1.0 "+str(rel.identifier))

            # features
            # arguments namespace
            f.write("|arguments ")
            f.write("arg1_"+rel.arg1type.strip())
            f.write(" arg2_"+rel.arg2type.strip())

            # n-grams namespace
            f.write("|n-grams ")
            before_tokens = re.findall(TOKENIZER, rel.before, flags=re.UNICODE)
            between_tokens = re.findall(TOKENIZER, rel.between, flags=re.UNICODE)
            after_tokens = re.findall(TOKENIZER, rel.after, flags=re.UNICODE)
            bef_grams = extractor.extract_ngrams(' '.join(before_tokens), "BEF")
            bet_grams = extractor.extract_ngrams(' '.join(between_tokens), "BET")
            aft_grams = extractor.extract_ngrams(' '.join(after_tokens), "AFT")
            f.write(bef_grams.getvalue().encode("utf8")+" "+bet_grams.getvalue().encode("utf8")+" "+aft_grams.getvalue().encode("utf8")+" ")

            rel_id += 1
            f.write("\n")

    f_sentences.close()


def classify(data_file, extractor, f):
    f_output = codecs.open("sentences_id.txt", "wb", encoding='utf-8')
    rel_id = 0
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if rel_id > 1000:
            break

        # extract relationships from sentence
        sentence = Sentence(line.strip())

        # rel = Relationship(sentence, None, None, None, None, None, None, None, None, rel_id)
        for rel in sentence.relationships:
            f_output.write(str(rel_id)+'\t'+rel.ent1+'\t'+rel.ent2+'\t'+rel.sentence+'\n')
            # relationship type
            f.write(" ")

            # relationship identifier
            f.write(" 1.0 "+str(rel_id))

            # features
            # arguments namespace
            f.write("|arguments ")
            f.write("arg1_"+rel.arg1type.strip())
            f.write(" arg2_"+rel.arg2type.strip())

            # n-grams namespace
            f.write("|n-grams ")
            before_tokens = re.findall(TOKENIZER, rel.before, flags=re.UNICODE)
            between_tokens = re.findall(TOKENIZER, rel.between, flags=re.UNICODE)
            after_tokens = re.findall(TOKENIZER, rel.after, flags=re.UNICODE)
            bef_grams = extractor.extract_ngrams(' '.join(before_tokens), "BEF")
            bet_grams = extractor.extract_ngrams(' '.join(between_tokens), "BET")
            aft_grams = extractor.extract_ngrams(' '.join(after_tokens), "AFT")
            f.write(bef_grams.getvalue().encode("utf8")+" "+bet_grams.getvalue().encode("utf8")+" "+aft_grams.getvalue().encode("utf8")+" ")

            rel_id += 1
            f.write("\n")

    f_sentences.close()
    f_output.close()


def main():
    extractor = FeatureExtractor()
    f_train = open("train.vw", "wb")
    training(sys.argv[1], extractor, f_train)
    f_train.close()
    f_classify = open("classify.vw", "wb")
    classify(sys.argv[2], extractor, f_classify)
    f_classify.close()

if __name__ == "__main__":
    main()