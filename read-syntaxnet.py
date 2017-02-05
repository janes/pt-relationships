#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess


def main():

    sentence = """
    Quase 900 funcionários do Departamento de Estado assinaram memorando que
    critica Trump.
    """

    process = subprocess.Popen(
        'MODEL_DIRECTORY=/Users/dbatista/Downloads/Portuguese; '
        'cd /Users/dbatista/models/syntaxnet; '
        'echo "%s" | syntaxnet/models/parsey_universal/parse.sh '
        '$MODEL_DIRECTORY 2' % sentence,
        shell=True,
        universal_newlines=False,
        stdout=subprocess.PIPE)

    output = process.communicate()

    for line in output[0].split("\n"):
        word = line.split("\t")
        sentence.append(word)

    # find ROOT verb
    for word in sentence:
        if len(word) == 1:
            continue
        if word[7] == 'ROOT' and word[3] == 'VERB':
            print word

if __name__ == "__main__":
    main()