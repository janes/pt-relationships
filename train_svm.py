#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk

__author__ = 'dsbatista'

import codecs
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cross_validation import KFold
from Sentence import Relationship, Sentence


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
    relationships = list()
    rel_id = 0
    print "Loading relationships frmo file"
    f_sentences = codecs.open(data_file, encoding='utf-8')
    for line in f_sentences:
        if not re.match('^relation', line):
            sentence = line.strip()
        else:
            rel_type = line.strip().split(':')[1]
            rel = Relationship(sentence, None, None, None, None, None, None, None, rel_type, rel_id)
            rel_id += 1
            relationships.append(rel)

    return relationships


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
    relationships = load_relationships(sys.argv[1])
    print len(relationships), " loaded"

    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords_pt)
    x_train = vectorizer.fit_transform([rel.sentence for rel in relationships])

    samples_features = []
    sample_class = []
    relationships_by_id = dict()

    for rel in relationships:
        words = x_train[rel.identifier].toarray()[0]
        samples_features.append(words)
        sample_class.append(rel_type_id[rel.rel_type])
        relationships_by_id[rel.identifier] = rel

    print len(samples_features), " samples"
    print len(rel_type_id), " classes"

    kf = KFold(len(relationships), 2)

    fold = 1
    for train_index, test_index in kf:
        print "\nFOLD", fold
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
        print "accuracy:", clf.score(test, test_label)
        print "\n"

        index_rel = 0
        for class_prob in results:
            print "rel_id:\t", test_ids[index_rel]
            print "sentence:\t", relationships_by_id[test_ids[index_rel]].sentence

            class_prob_lst = list(class_prob)
            #scores = []
            print "range", range(len(class_prob_lst))
            for c_index in range(0, len(class_prob_lst)):
                print id_rel_type[c_index], '\t', class_prob_lst[c_index]
                #scores.append((id_rel_type[c_index], class_prob_lst[c_index]))

            """
            sorted_by_score = sorted(scores, key=lambda tup: tup[1])
            for t in sorted_by_score:
                print t[0], '\t:', t[1]
            """

            index_rel += 1
        fold += 1

if __name__ == "__main__":
    main()