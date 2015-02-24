#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk

__author__ = 'dsbatista'

import MySQLdb


def main():
    # connect to the DB
    configuration = eval(open("agatha_configuration.dict").read())
    user = configuration['username']
    passwd = configuration['password']
    host = configuration['hostname']
    db = configuration['database']
    conn = MySQLdb.connect(host, user, passwd, db, use_unicode="True", charset="utf8")
    cursor = conn.cursor()
    query = "SELECT title_ner, body_ner FROM content_named_entities, content WHERE content_named_entities.id = " \
            "content.id AND (content.type = 'news' OR content.type = 'blog')"
    status = cursor.execute(query)

    print "Records feteched", status

    row = cursor.fetchall()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    f = open('sentences_big_corpus.txt', 'w')
    c = 0
    for r in row:
        if c % 50000 == 0:
            print c
        try:
            sentences = sent_tokenizer.tokenize(r[0])
            for s in sentences:
                #print s.encode("utf8").replace('\n', ' ')
                f.write(s.encode("utf8").replace('\n', ' ')+'\n')
            sentences = sent_tokenizer.tokenize(r[1])
            for s in sentences:
                #print s.encode("utf8").replace('\n', ' ')
                f.write(s.encode("utf8").replace('\n', ' ')+'\n')
            c += 1
        except Exception:
            continue


    conn.close()

if __name__ == "__main__":
    main()
