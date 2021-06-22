import numpy as np
import random
import sys

mat =[]
FILENAME='darp01.txt'
with open('/Users/kurozumi ryouho/Desktop/benchmark/'+ FILENAME,'r',encoding='utf-8') as fin:
    for line in fin.readlines():
        row = []
        toks = line.split(' ')
        for tok in toks:
            try:
                num = float(tok)
            except ValueError:
                continue
            row.append(num)
        mat.append(row)
print(mat)

Setting_Info = mat.pop(0)
print(Setting_Info)
print(mat)