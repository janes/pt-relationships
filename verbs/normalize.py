
l#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import fileinput

verbs = dict()
prepositions = dict()

"""
load verbs and propostions from the Label-Delaf_pt_v4_1.dic.utf8
"""
def loadLabel(data):
    for line in fileinput.input(data):
        if not line.startswith('#'):
            category = line.split('.')[-1]
            if category.startswith('V') and (not category.startswith('.Vm')):
                infitinitve = line.split('.')[0].split(',')[1]
                conjugation = line.split('.')[0].split(',')[0]
                try:
                    tmp = verbs[conjugation]
                    if (len(tmp)>=1):
                        tmp.append(infitinitve)
                        verbs[conjugation.decode("utf8")] = tmp
                    else:
                        tmp = list()
                        tmp.append(infitinitve.decode("utf8"))
                        verbs[conjugation.decode("utf8")] = tmp
                except:
                    tmp = list()
                    tmp.append(infitinitve.decode("utf8"))
                    verbs[conjugation.decode("utf8")] = tmp

            if category.startswith('PREP'):
                normalized = line.split('.')[0].split(',')[1]
                form = line.split('.')[0].split(',')[0]
                prepositions[form] = normalized
    fileinput.close()




def findVerb(word):
    conjugation = (word.split("_")[0])
    try:
        infinite = verbs[conjugation]
        return infinite
    except KeyError:
        return None


def findPreposition(word):
    try:
        normalized = prepositions[word]
        return normalized
    except KeyError:
        return word


def main():

    ambigous = ['foi_a_RVB', 'foi_de_RVB', 'fora_de_RVB', 'foram_de_RVB', 'for_de_RVB', 'foi_mais_de_RVB','fossem_de_RVB','fosse_a_RVB','fossem_em_RVB','fossem_sobre_RVB','fossem_para_RVB','fossem_por_RVB','fossem_de_RVB','fosse_com_RVB','fossem_de_RVB','tendo_a_RVB','tende_a_RVB','adia_em_RVB']

    loadLabel(sys.argv[2])

    f1 = open("patterns.txt",'w')
    f2 = open("not_found.txt",'w')
    f3 = open("ambiguous.txt",'w')

    for line in fileinput.input(sys.argv[1]):
        data = line.split(" ")
        pattern = data[-1].decode("utf8")
        frequency = data[-2].decode("utf8")

        #normalized preposition
        preposition = pattern.split("_")[-2]
        preposition_normalized = findPreposition(preposition)

        #normalize verb
        infinite = findVerb(pattern)
        verb_normalized = infinite

        if infinite is not None:
            if len(verb_normalized)==1:
                try:
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    f1.write(out.encode("utf8"))
                except:
                    print verb_normalized[0]
                    print preposition_normalized
                    print frequency
                    print pattern.strip()

            elif len(verb_normalized)==2:
                found = False

                #ter e tender    -> ter     196 casos (tendo)
                if (verb_normalized[0]=='ter' and verb_normalized[1]=='tender' or verb_normalized[1]=='ter' and verb_normalized[0]=='tender' and pattern not in ambigous):
                    out = 'ter' + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #fossar e ir     -> ir      119 casos
                if (verb_normalized[0]=='fossar' and verb_normalized[1]=='ir'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #estar e estevar -> estar   115 casos
                if (verb_normalized[0]=='estar' and verb_normalized[1]=='estevar'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #ser e seriar    -> ser     113 casos
                if (verb_normalized[0]=='ser' and verb_normalized[1]=='seriar'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #vivar e viver   -> viver    40 casos
                if (verb_normalized[0]=='vivar' and verb_normalized[1]=='viver'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #falar e falir   -> falar    37 casos
                if (verb_normalized[0]=='falar' and verb_normalized[1]=='falir'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #segar e seguir  -> seguir   22 casos
                if (verb_normalized[0]=='segar' and verb_normalized[1]=='seguir'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #podar e poder   -> poder     8 casos
                if (verb_normalized[0]=='podar' and verb_normalized[1]=='poder'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #unar e unir     -> unir      5 casos
                if (verb_normalized[0]=='unar' and verb_normalized[1]=='unir'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #ir e iriar      -> ir        4 casos
                if (verb_normalized[0]=='ir' and verb_normalized[1]=='iriar'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #ver e vestir    -> ver       4 casos
                if (verb_normalized[0]=='ver' and verb_normalized[1]=='vestir'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #estar e estivar -> estar     4 casos
                if (verb_normalized[0]=='estar' and verb_normalized[1]=='estivar'):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #presidiar e presidir -> presidir 4 casos
                if (verb_normalized[0]=='presidiar' and verb_normalized[1]=='presidir'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #adiar	adir	sobre
                if (verb_normalized[0]=='adiar' and verb_normalized[1]=='adir'and pattern not in ambigous):
                    out = verb_normalized[0] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #crer	criar
                if (verb_normalized[0]=='crer' and verb_normalized[1]=='criar'):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                #ir	ser	a	347	foi_a_RVB
                #ir	ser	de	312	foi_de_RVB
                #ir	ser	de	183	fora_de_RVB
                #ir	ser	de	57	foram_de_RVB
                #ir	ser	de	25	for_de_RVB
                #ir	ser	de	15	foi_mais_de_RVB
                #ir	ser	de	1	foi_“_de_RVB

                if (verb_normalized[0]=='ir' and verb_normalized[1]=='ser' and pattern not in ambigous):
                    out = verb_normalized[1] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

                f1.write(out.encode("utf8"))

                if found == False:
                    out = '\t'.join(verb_normalized)
                    out += '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    f3.write(out.encode("utf8"))


            elif len(verb_normalized)==3:
                if (verb_normalized[0]=='fossar' and verb_normalized[1]=='ir' and verb_normalized[2]=='ser' and pattern not in ambigous):
                    out = verb_normalized[2] + '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                    found = True

            else:
                out = '\t'.join(verb_normalized)
                out += '\t' + preposition_normalized + '\t' + frequency + '\t' + pattern.strip() + '\n'
                f3.write(out.encode("utf8"))
        else:
            out = frequency + '\t' + pattern.strip() + '\n'
            f2.write(out.encode("utf8"))
    f1.close()
    f2.close()
    f3.close()
    fileinput.close()

if __name__ == "__main__":
    main()