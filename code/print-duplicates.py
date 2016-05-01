#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import fileinput
import codecs

sentences = set()

def main():
    # read files already tagged
    # for each sentence, generate an hash code
    # store hash code in set
    f = codecs.open(sys.argv[1], encoding='utf-8')
    for line in f:
        if not line.startswith("relation") and len(line)>1:            
            clean = re.sub(r"</?[A-Z]+>", "", line)
            if clean in sentences:
                print "dup",clean.encode("utf8")
            else:
                sentences.add(clean)
    f.close()

if __name__ == "__main__":
    main()
