import csv


import unicodedata
import os



with open('entities.dict') as fin:
    entity2id = dict()
    id2entity = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
        id2entity[int(eid)] = entity

row_1 = []
row_1 = []

with open('output1.csv') as fin:
    for row in csv.reader(fin, delimiter=','):
        if row[0] == '':
            continue
        # print(row)
        row_1 = entity2id[row[1]]
        row_2 = row[2].strip('][').split(', ')
        # print(row_2)
        row_2 = [float(i) for i in row_2]
        print(row_1)
        print(row_2)