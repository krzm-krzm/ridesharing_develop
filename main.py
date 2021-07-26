import numpy as np
import random
import sys
import math
import time

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
    Setting_Info = mat.pop(0) #0:車両数、4:キャパシティ、8:一台あたりの最大移動時間(min)、9:一人あたりの最大移動時間(min)

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

    #乗り降りの0-1情報を格納
    noriori = np.zeros(len(mat),dtype=int,order='C')
    for i in range(len(mat)):
        noriori[i] = mat[i][4]


    return Setting_Info,request_number,depo_zahyo,c,e,l,noriori


def initial_sulution(request_node,vehicle_number):
    riyoukyaku_number =  np.arange(1,request_node/2+1)
    Route = [[] *1 for i in range(vehicle_number)]
    i =0
    while True:
        if riyoukyaku_number.size == 0:
            break
        if i > vehicle_number-1:
            i = 0
        a = int(np.random.choice(riyoukyaku_number,1))
        Route[i].append(a)
        b = a + int(request_node/2)
        Route[i].append(b)
        riyoukyaku_number = np.delete(riyoukyaku_number,np.where(riyoukyaku_number == a))
        i = i+1

    return Route

def Route_cost(Route,node_cost):
    Route_sum =0
    Route_sum_k = np.zeros(len(Route),dtype=float,order='C')
    for i in range(len(Route)):
        for j in range(len(Route[i])-1):
            Route_sum_k[i] = Route_sum_k[i] + node_cost[Route[i][j]][Route[i][j+1]]
        Route_sum_k[i] = Route_sum_k[i] + node_cost[0][Route[i][0]]
        Route_sum_k[i]  = Route_sum_k[i] + node_cost[0][Route[i][j+1]]
        Route_sum = Route_sum+Route_sum_k[i]

    return Route_sum

def capacity(Route,q,Q_max,noriori):
    capacity_over =0
    for i in range(len(Route)):
        for j in range(len(Route[i])):
            q[i] = q[i] + noriori[Route[i][j]]
            if q[i] >Q_max:
                capacity_over += 1
    return capacity_over

def time_caluculation(Route_k,node_cost,e,d,request_node):
    B = np.zeros(n + 2, dtype=float, order='C') #サービス開始時間（e.g., 乗せる時間、降ろす時間)
    A = np.zeros(n + 2, dtype=float, order='C') #ノード到着時間
    D = np.zeros(n + 2, dtype=float, order='C') #ノード出発時間
    W = np.zeros(n + 2, dtype=float, order='C')  #車両待ち時間
    L = np.zeros(int(request_node/2),dtype=float, order='C') #リクエストiの乗車時間
    for i in range(len(Route_k)):
        if i ==0:
            A[Route_k[i]] =D[i] +node_cost[i][Route_k[i]]
            B[Route_k[i]] = max(e[Route_k[i]],A[Route_k[i]])
            D[Route_k[i]] = B[Route_k[i]] +d
            W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
        else:
            A[Route_k[i]] = D[Route_k[i-1]] +node_cost[Route_k[i-1]][Route_k[i]]
            B[Route_k[i]] = max(e[Route_k[i]], A[Route_k[i]])
            D[Route_k[i]] = B[Route_k[i]] + d
            W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
    A[-1] = D[Route_k[i]] + node_cost[0][Route_k[i]]
    B[-1] = A[-1]
    for i in range(len(Route_k)):
        if Route_k[i] <= request_node/2:
            L[Route_k[i]-1] = B[Route_k[i]+int(request_node/2)] - D[Route_k[i]]
    return A,B,D,W,L

def time_window_penalty(Route_k,B,l):
    sum =0
    for i in range(len(Route_k)):
        a = B[Route_k[i]] - l[Route_k[i]]
        if a > 0:
          sum = sum +a
    a = B[-1] -l[0]
    if a >0:
        sum = sum +a
    return sum

def ride_time_penalty(L,L_max):
    sum =0
    for i in range(len(L)):
        a = L[i] -L_max
        if a >0:
          sum = sum+a
    return sum
FILENAME='darp01.txt'
Setting_Info= Setting(FILENAME)[0]

n = int(Setting(FILENAME)[1]) #depoを除いたノード数
m =int(Setting_Info[0])  #車両数
d = 5 #乗り降りの時間
Q_max =Setting_Info[4]  #車両の最大容量
T_max = Setting_Info[8] #一台当たりの最大移動時間
L_max = Setting_Info[9] #一人あたりの最大移動時間

q=np.zeros(int(m),dtype=int,order='C')



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

Route = initial_sulution(n,m)

print(Route)

noriori = np.zeros(n+1,dtype=int,order='C')
noriori = Setting(FILENAME)[6]

print(noriori)

Route_SUM = Route_cost(Route,c)
print(Route_SUM)

print(T_max)
print(L_max)

q_s = capacity(Route,q,Q_max,noriori)
print(q_s)


A = np.zeros(n+2,dtype=float,order='C')
D =np.zeros(n+2,dtype=float,order='C')

B= time_caluculation(Route[0],c,e,d,n)
print(B[4])

w = time_window_penalty(Route[0],B[1],l)
print(w)

t = ride_time_penalty(B[4],L_max)
print(t)