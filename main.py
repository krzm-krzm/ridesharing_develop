'''
以下グローバル変数→定義しないとダメな変数ですよ
m:車両数
c:ノード間コスト
noriori:ノードにおいての乗り降り[1,-1]変数
e,l:最早出発、最遅出発時間
d:乗り降りの時間(サービス時間)
Q_max:車両の最大容量
T_max:一台当たりの最大移動時間
L_max:一人あたりの最大移動時間
'''
import numpy as np
import random
import sys
import math
import time
import copy

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
    route = [[] *1 for i in range(vehicle_number)]
    i =0
    while True:
        if riyoukyaku_number.size == 0:
            break
        if i > vehicle_number-1:
            i = 0
        a = int(np.random.choice(riyoukyaku_number,1))
        route[i].append(a)
        b = a + int(request_node/2)
        route[i].append(b)
        riyoukyaku_number = np.delete(riyoukyaku_number,np.where(riyoukyaku_number == a))
        i = i+1

    return route

def Route_cost(route):
    Route_sum =0
    Route_sum_k = np.zeros(len(route),dtype=float,order='C')
    for i in range(len(route)):
        for j in range(len(route[i])-1):
            Route_sum_k[i] = Route_sum_k[i] + c[route[i][j]][route[i][j+1]]
        Route_sum_k[i] = Route_sum_k[i] + c[0][route[i][0]]
        Route_sum_k[i]  = Route_sum_k[i] + c[0][route[i][j+1]]
        Route_sum = Route_sum+Route_sum_k[i]

    return Route_sum

def route_k_cost_sum(route_k):
    route_k_sum =0
    for i in range(len(route_k)-1):
        route_k_sum =route_k_sum+c[route_k[i]][route_k[i+1]]
    route_k_sum = route_k_sum +c[0][route_k[0]]
    route_k_sum = route_k_sum + c[0][route_k[i+1]]

    return  route_k_sum


def capacity(route):
    q = np.zeros(int(m), dtype=int, order='C')
    capacity_over =0
    for i in range(len(route)):
        for j in range(len(route[i])):
            q[i] = q[i] + noriori[route[i][j]]
            if q[i] >Q_max:
                capacity_over += 1
    return capacity_over

def capacity_route_k(route_k):
    capacity_over =0
    q =0
    for i in range(len(route_k)):
        q = q+noriori[route_k[i]]
        if q >Q_max:
            capacity_over +=1
    return capacity_over

def time_caluculation(Route_k,request_node):
    B = np.zeros(n + 2, dtype=float, order='C') #サービス開始時間（e.g., 乗せる時間、降ろす時間)
    A = np.zeros(n + 2, dtype=float, order='C') #ノード到着時間
    D = np.zeros(n + 2, dtype=float, order='C') #ノード出発時間
    W = np.zeros(n + 2, dtype=float, order='C')  #車両待ち時間
    L = np.zeros(int(request_node/2),dtype=float, order='C') #リクエストiの乗車時間
    for i in range(len(Route_k)):
        if i ==0:
            A[Route_k[i]] =D[i] +c[i][Route_k[i]]
            B[Route_k[i]] = max(e[Route_k[i]],A[Route_k[i]])
            D[Route_k[i]] = B[Route_k[i]] +d
            W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
        else:
            A[Route_k[i]] = D[Route_k[i-1]] +c[Route_k[i-1]][Route_k[i]]
            B[Route_k[i]] = max(e[Route_k[i]], A[Route_k[i]])
            D[Route_k[i]] = B[Route_k[i]] + d
            W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
    A[-1] = D[Route_k[i]] + c[0][Route_k[i]]
    B[-1] = A[-1]
    for i in range(len(Route_k)):
        if Route_k[i] <= request_node/2:
            L[Route_k[i]-1] = B[Route_k[i]+int(request_node/2)] - D[Route_k[i]]
    return A,B,D,W,L


def time_window_penalty(route_k,b): #論文でのw(s)
    sum =0
    for i in range(len(route_k)):
        a = b[route_k[i]] - l[route_k[i]]
        if a > 0:
          sum = sum +a
    a = b[-1] -l[0]
    if a >0:
        sum = sum +a
    return sum

def ride_time_penalty(L): #論文でのt_s
    sum =0
    for i in range(len(L)):
        a = L[i] -L_max
        if a >0:
          sum = sum+a
    return sum

'''
enumerate関数を使っている
for j,row in enumerate(Route)
jが車両番号、rowは車両jの顧客リストを取得
中のforループで近傍探索で変更するランダムで選ばれた顧客のindex値（あるいはそのものの値）を取得できるまで回している
'''


def neighbourhood(route,requestnode):
    mm = np.arange(len(route))
    i = random.randint(1,requestnode/2)
    for j,row in enumerate(route):
        try:
            u_before = [i,j]
            i_index = row.index(i)
            break
        except ValueError:
            pass
    u_before = np.array(u_before) #車両変更前 U = [顧客番号、車両番号]
    k_new = int(np.random.choice(mm[mm != u_before[1]],size=1))
    u_after = np.array([u_before[0],k_new]) #車両変更後 U = [顧客番号、新たな車両番号]
    neighbour = np.append(u_before,u_after,axis=0).reshape(2,2)
    return neighbour

def newRoute(route,requestnode,neighbour):
    new_route = copy.deepcopy(route)
    for j in range(len(route)):
        try:
            new_route[j].remove(neighbour[0][0])
            new_route[j].remove(neighbour[0][0]+requestnode/2)
            break
        except ValueError:
            pass
    new_route = insert_route(new_route,requestnode,neighbour)
    return new_route

def penalty_sum(route,requestnode):
    parameta = np.zeros(4)
    c_s = Route_cost(route)
    q_s = capacity(route)
    d_s =0
    w_s =0
    t_s =0
    for i in range(len(route)):
        ROUTE_TIME_info= time_caluculation(route[i],requestnode)
        d_s =d_s+ROUTE_TIME_info[1][-1]
        w_s =w_s +time_window_penalty(route[i],ROUTE_TIME_info[1])
        t_s =t_s +ride_time_penalty(ROUTE_TIME_info[4])
    penalty =c_s+keisu[0]*q_s+keisu[1]*d_s+keisu[2]*w_s+keisu[3]*t_s
    parameta[0] = q_s
    parameta[1] = d_s
    parameta[2] = w_s
    parameta[3] =t_s
    return penalty,parameta

def penalty_sum_route_k(route_k,requestnode):
    c_s =route_k_cost_sum(route_k)
    q_s =capacity_route_k(route_k)
    d_s = 0
    w_s = 0
    t_s = 0
    ROUTE_TIME_info = time_caluculation(route_k, requestnode)
    d_s = ROUTE_TIME_info[1][-1]
    w_s = time_window_penalty(route_k, ROUTE_TIME_info[1])
    t_s = ride_time_penalty(ROUTE_TIME_info[4])

    penalty = c_s + keisu[0]*q_s + keisu[1]*d_s + keisu[2]*w_s + keisu[3]*t_s
    return penalty

def insert_route(route,requestnode,neighbor):
    new_route_k =copy.copy(route[neighbor[1][1]])
    insert_number=neighbor[0][0]
    route_k_node = len(route[neighbor[1][1]])

    new_route_k.insert(0,insert_number)
    new_route_k.insert(1,insert_number+int(requestnode/2))
    penalty=penalty_sum_route_k(new_route_k,requestnode)
    check_route = copy.copy(route[neighbor[1][1]])
    for i in range(route_k_node):
        j=i+1
        while j<=4:
            check_route = copy.copy(route[neighbor[1][1]])
            check_route.insert(i,insert_number)
            check_route.insert(j,int(insert_number+requestnode/2))
            check_penalty =penalty_sum_route_k(check_route,requestnode)
            if check_penalty <penalty:
                penalty =check_penalty
                new_route_k =check_route
            j=j+1
    new_route =copy.copy(route)
    new_route[neighbor[1][1]] =copy.copy(new_route_k)
    return new_route

def keisu_update(delta,parameta):
    for i in range(len(parameta)):
        if parameta[i] > 0:
            keisu[i] =keisu[i] * (1+delta)
        else:
            keisu[i] = keisu[i] /(1+delta)

def tabu_update(theta,tabu_list,neighbour):
    for i in range(math.ceil(theta)):
        if tabu_list[i][2] == -1:
            tabu_list[i][0] = neighbour[0][0]
            tabu_list[i][1] = neighbour[0][1]
            tabu_list[i][2] = math.ceil(theta) +1
            break
    for i in range(math.ceil(theta)):
        if tabu_list[i][2] >=0:
            tabu_list[i][2] = tabu_list[i][2]-1


def main():
    initial_Route = initial_sulution(n, m) #初期解生成
    opt = penalty_sum(initial_Route,n)[0]
    test =penalty_sum(initial_Route,n)[0]
    loop =0 #メインのループ回数
    parameta_loop =0 #パラメーター調整と集中化のループ回数(ループ回数は10回)
    delta =0.5
    theta =7.5*math.log10(n/2)
    tabu_list = np.zeros((math.ceil(theta),3))-1
    while True:
        while True:
            Neighbour =neighbourhood(initial_Route,n)   #近傍探索操作[リクエスト番号,以前の車両]→[リクエスト番号,挿入する車両]
            check=0
            for i in range(math.ceil(theta)): #ループ回数をtabu_listのサイズにあわせなければならない
                if tabu_list[i][0] == Neighbour[1][0] and tabu_list[i][1] == Neighbour[1][1] and tabu_list[i][2] >=0: #たぶん間違い
                    check +=1
                if tabu_list[i][0] == Neighbour[0][0] and tabu_list[i][1] == Neighbour[0][1] and tabu_list[i][2] >=0: #たぶん間違い
                    check +=1
            if check ==0:
                break
        NewRoute = newRoute(initial_Route,n,Neighbour)

        if penalty_sum(NewRoute,n)[0] <= opt:
            opt = penalty_sum(NewRoute,n)[0]

        keisu_update(delta,penalty_sum(NewRoute,n)[1])
        tabu_update(theta,tabu_list,Neighbour)

        parameta_loop += 1
        if parameta_loop ==100:
            delta =np.random.uniform(0,0.5)
            theta = np.random.uniform(0,7.5*math.log10(n/2))
        loop +=1
        if loop ==10:
            break

    print(initial_Route)
    print(NewRoute)
    print(test,opt)
    print(penalty_sum(NewRoute,n)[1])
    print(keisu)
    print(tabu_list)


if __name__ =='__main__':
    FILENAME = 'darp01.txt'
    Setting_Info = Setting(FILENAME)[0]

    n = int(Setting(FILENAME)[1])  # depoを除いたノード数
    m = int(Setting_Info[0])  # 車両数
    d = 5  # 乗り降りの時間
    Q_max = Setting_Info[4]  # 車両の最大容量 global変数 capacity関数で使用
    T_max = Setting_Info[8]  # 一台当たりの最大移動時間
    L_max = Setting_Info[9]  # 一人あたりの最大移動時間

    noriori = np.zeros(n + 1, dtype=int, order='C')
    noriori = Setting(FILENAME)[6]  # global変数  capacity関数で使用

    depo_zahyo = Setting(FILENAME)[2]  # デポの座標

    c = np.zeros((n + 1, n + 1), dtype=float, order='C')
    c = Setting(FILENAME)[3]  # 各ノード間のコスト

    e = np.zeros(n + 1, dtype=float, order='C')
    l = np.zeros(n + 1, dtype=float, order='C')
    e = Setting(FILENAME)[4]
    l = Setting(FILENAME)[5]

    keisu =np.ones(4)
    main()

