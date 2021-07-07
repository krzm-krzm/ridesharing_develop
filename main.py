import numpy as np
import random
import sys
import math

def distance(x1,x2,y1,y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d



def Setting(FILENAME):
    mat=[]
    with open('/home/kurozumi/デスクトップ/benchmark/'+ FILENAME,'r',encoding='utf-8') as fin:
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
    #インスタンスの複数の行（問題設定）を取り出す
    Setting_Info = mat.pop(0) #0:車両数、4:キャパシティ、8:一台あたりの最大移動時間(min)、10:一人あたりの最大移動時間(min)

    #デポの座標を取り出す
    depo_zahyo = np.zeros(2) #デポ座標配列
    x=mat.pop(-1)
    depo_zahyo[0] = x[1]
    depo_zahyo[1] = x[2]

    request_number = len(mat)-1

    #各距離の計算
    c = np.zeros((len(mat),len(mat)),dtype=float,order='C')

    # eがtime_windowの始、lが終
    e = np.zeros(len(mat),dtype=float,order='C')
    l = np.zeros(len(mat),dtype=float,order='C')

    #テキストファイルからtime_windowを格納 & 各ノードの距離を計算し格納
    for i in range(len(mat)):
        e[i] = mat[i][5]
        l[i] =mat[i][6]
        for j in range(len(mat)):
            c[i][j] =distance(mat[i][1],mat[j][1],mat[i][2],mat[j][2])




    return Setting_Info,request_number,depo_zahyo,c,e,l


def initial_sulution(request_node,vehicle_number):
    riyoukyaku_number = np.arange(1,request_node/2)
    route_list = np.zeros((int(vehicle_number),0)
    while True:
        if riyoukyaku_number.size == 0:
            break



FILENAME='darp01.txt'
Setting_Info= Setting(FILENAME)[0]

n = Setting(FILENAME)[1] #depoを除いたノード数
m =Setting_Info[0]  #車両数
Q_max =Setting_Info[4]  #車両の最大容量

q=np.zeros(int(m),dtype=int,order='C')
q=q+Q_max   #各車両の容量


depo_zahyo=Setting(FILENAME)[2] #デポの座標

c = np.zeros((n+1,n+1),dtype=float,order='C')
c= Setting(FILENAME)[3] #各ノード間のコスト
print(q)

e = np.zeros(n+1,dtype=float,order='C')
l = np.zeros(n+1,dtype=float,order='C')
e = Setting(FILENAME)[4]
l = Setting(FILENAME)[5]

print(e)

#initial_solution

route_list = np.empty((0,0))

arr1 = np.arange(1,6)
#  [2]の位置の要素を削除した新しい配列を生成。
arr2 = np.delete(arr1,2)
route_list = np.append(route_list,arr2)
route_list = np.append(route_list,arr2,axis=0)
print(route_list)
