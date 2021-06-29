import numpy as np
import random
import sys
import math

def distance(x1,x2,y1,y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

def Setting(FILENAME):
    mat=[]
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

    #インスタンスの最初の行（問題設定）を取り出す
    Setting_Info = mat.pop(0) #0:車両数,4:キャパシティ、8:一台当たりの最大移動時間(min)、10:一人あたりの最大移動時間(min)

    # デポの座標を取り出す
    depo_zahyo = np.zeros(2)  # デポ座標配列
    x = mat.pop(-1)
    depo_zahyo[0] = x[1]
    depo_zahyo[1] = x[2]

    request_number = len(mat) - 1

    # 各距離の計算
    c = np.zeros((len(mat), len(mat)), dtype=float, order='C')
    for i in range(len(mat)):
        for j in range(len(mat)):
            c[i][j] = distance(mat[i][1], mat[j][1], mat[i][2], mat[j][2])

    return Setting_Info,request_number,depo_zahyo,c

FILENAME='darp01.txt'

Setting_Info= Setting(FILENAME)[0]

n = Setting(FILENAME)[1] #リクエスト数
m =Setting_Info[0]  #車両数
Q_max =Setting_Info[4]  #車両の最大容量

q=np.zeros(int(m),dtype=int,order='C')
q=q+Q_max   #各車両の容量


depo_zahyo=Setting(FILENAME)[2] #デポの座標

c = np.zeros((n+1,n+1),dtype=float,order='C')
c= Setting(FILENAME)[3] #各ノード間のコスト
print(q)


