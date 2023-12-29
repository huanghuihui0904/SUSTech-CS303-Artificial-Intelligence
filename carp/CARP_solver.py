import argparse
import copy
import time
import random
import heapq
from multiprocessing.pool import Pool

import numpy as np
import operator
#------------------------------------全局变量
NAME = ''
VERTICES = 0
DEPOT = 0
REQUIRED_EDGES = 0
Required = []
NON_REQUIRED_EDGES = 0
VEHICLES = 0
CAPACITY = 0
TOTAL_COST_OF_REQUIRED_EDGES = 0
MAX = 99999
# table = []
TIME = 0
SEED = 0
FILEPATH = ''
TABLE=[]
TABLE_COST=[]
DISTANCE=[]
DEMAND=[]
START_TIME=0
SIZE=30
BEST_FEASIBLE_SOLUTION = {'pay': 0, 'best': []}

UBTRAIL = 70
TASKS=[]



def crv(s1, s2):
    ee22 = random.randint(0, len(s2) - 1)
    ee11 = random.randint(0, len(s1) - 1)
    s_x = copy.deepcopy(s1)
    r_1, r_2 = s_x[ee11].copy(), s2[ee22].copy()

    i2 = random.randint(0, len(r_2) - 1)
    r_21, r_22 = r_2[:i2], r_2[i2:]


    i1 = random.randint(0, len(r_1) - 1)
    r_11, r_12 = r_1[:i1], r_1[i1:]


    nrrr = r_11.copy()
    bo=False
    for item in r_22:
        if   (item[1], item[0]) in set(r_11):
            continue
        elif item in set(r_11):
            continue
        elif bo:
            a=1
        else:
            nrrr.append(item)
            if  (item[1], item[0]) not in set(r_12):
                t7=0
                while t7<len(s_x):
                    route=s_x[t7]
                    if item not in set(route) and (item[1], item[0]) not in set(route):
                        t7+=1
                        continue
                    elif item in set(route):
                        t7+=1
                        route.remove(item)
                        break
                    route.remove((item[1], item[0]))
                    t7+=1
                    break
            elif item not in set(r_12):
                t8=0
                while t8<len(s_x):
                    route=s_x[t8]
                    if item not in set(route) and (item[1], item[0]) not in set(route):
                        t8+=1
                        continue
                    elif item in set(route):
                        t8+=1
                        route.remove(item)
                        break
                    route.remove((item[1], item[0]))
                    t8+=1
                    break

    s = set(nrrr)
    ssim = [item for item in r_12 if item not in s and (item[1], item[0]) not in s]
    t9=0
    while t9<len(ssim):
        task=ssim[t9]
        position =  0
        min_cost= float('inf')
        for i in range(len(nrrr)+1):
            ac = TABLE[task[0]][task[1]][1]
            if i == len(nrrr):
                ac += DISTANCE[nrrr[i-1][1]][task[0]]
                ac += DISTANCE[task[1]][DEPOT]
            elif i == 0:
                ac += DISTANCE[DEPOT][task[0]]
                ac += DISTANCE[task[1]][nrrr[i][0]]
            else:
                ac += DISTANCE[nrrr[i - 1][1]][task[0]]
                ac += DISTANCE[task[1]][nrrr[i][0]]
            if ac <= min_cost:
                min_cost, position = ac, i
        nrrr.insert(position, task)
        t9+=1
    s_x[ee11] = nrrr
    s_x = [item for item in s_x if item != []]
    return s_x


def floyd(graph):
    n = len(graph)
    for k in range(1, n):
        graph[k][k] = 0
        for i in range(1, n):
            for j in range(1, n):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    return graph


def printformat(cur_route, cur_cost):
    s = str(cur_route)
    s=s.replace('[','').replace(']','').replace(' ','')

    print('s',s)
    print('q', cur_cost)



def es(s, lmd):

    total_violate = 0
    pay = 0
    t1=0
    while t1<len(s):
        route=s[t1]
        if len(route) == 0:
            t1+=1
            continue
        start = DEPOT
        route_load = 0
        t2=0
        while t2<len(route):
            tk=route[t2]
            route_load += TABLE[tk[0]][tk[1]][1]
            pay += DISTANCE[start][tk[0]]
            pay += TABLE[tk[0]][tk[1]][0]
            start = tk[1]
            t2+=1
        route_violate = max(0, route_load - CAPACITY)
        pay += DISTANCE[start][DEPOT]
        total_violate += route_violate
        t1+=1
    return pay+(lmd*total_violate)

def arg_analysis():
    parser = argparse.ArgumentParser(description="deal with args")
    parser.add_argument("file_name")
    parser.add_argument("-t", type=int)
    parser.add_argument("-s", type=int)
    args = parser.parse_args()
    return args.file_name, args.t, args.s

def read(filepath):
    file = open(filepath,"r",encoding= "utf-8")
    global NAME, VERTICES, DEPOT, Required,REQUIRED_EDGES, REQUIRED_EDGES, NON_REQUIRED_EDGES, VEHICLES, CAPACITY, TOTAL_COST_OF_REQUIRED_EDGES
    for i in range(0,9):
        line = file.readline().split()
        if line[0] == 'NAME':
            NAME = line[2]
        elif line[0] == 'VERTICES':
            VERTICES = int(line[2])
        elif line[0] == 'DEPOT':
            DEPOT = int(line[2])
        elif line[0] == 'REQUIRED':
            REQUIRED_EDGES = int(line[3])
        elif line[0] == 'NON-REQUIRED':
            NON_REQUIRED_EDGES = int(line[3])
        elif line[0] == 'VEHICLES':
            VEHICLES = int(line[2])
        elif line[0] == 'CAPACITY':
            CAPACITY = int(line[2])
        elif line[0] == 'TOTAL':
            TOTAL_COST_OF_REQUIRED_EDGES = int(line[6])

    datatable=[[(MAX,0) for j in range(VERTICES+1)] for j in range(VERTICES+1)]
    datatable_cost = MAX * np.ones((VERTICES + 1, VERTICES + 1),
                            dtype=np.int32)
    datatable_demand=[]
    for i in range(0,REQUIRED_EDGES):
        line = file.readline ().split ()
        x = int(line[0])
        y = int(line[1])
        cost = int(line[2])
        demand = int(line[3])
        datatable[x][y] = (cost,demand)
        datatable[y][x] = (cost, demand)
        datatable_cost[x][y]=cost
        datatable_cost[y][x]=cost
        datatable_demand.append((x,y,cost,demand))
        datatable_demand.append((y, x, cost, demand))
        Required.append((x,y))
        TASKS.append((x,y))
        TASKS.append((y, x))

    for i in range(REQUIRED_EDGES,REQUIRED_EDGES+NON_REQUIRED_EDGES):
        line = file.readline ().split ()
        x = int(line[0])
        y = int(line[1])
        cost = int(line[2])
        demand = int(line[3])
        datatable[x][y] = (cost,demand)
        datatable[y][x] = (cost, demand)
        datatable_cost[x][y] = cost
        datatable_cost[y][x] = cost
        datatable_demand.append((x, y, cost, demand))
        datatable_demand.append((y, x, cost, demand))

    return datatable,datatable_cost,datatable_demand


def cl(s, S):
    for p in S:
        if operator.eq(p, s):
            return True
    return False

def popinit(tasks):
    current_population = []
    while len(current_population)<200:
        individual = ps(tasks)
        if not cl(individual, current_population):
            heapq.heappush(current_population, (es(individual, lmd=0), individual))

    result = []
    sorted(current_population,key=lambda p:p[0])
    for i in range(SIZE):
        item = heapq.heappop(current_population)
        result.append(item[1])
    return result

def ps(tasks):
    candi = tasks.copy()
    S = []
    while True:
        if len(candi)<=0:
            break
        route = []
        start = DEPOT
        load = 0
        while True:
            candidates = []
            zuixiao = float('inf')
            for task in candi:
                distance = DISTANCE[start][task[0]]
                if distance < zuixiao:
                    candidates=[]
                    candidates.append(task)
                    zuixiao = distance
                elif distance == zuixiao:
                    candidates.append(task)
            candidate_count = len(candidates)
            if candidate_count == 0:
                break
            ran = random.randint(0, candidate_count-1)
            chossess = candidates[ran]
            if TABLE[chossess[0]][chossess[1]][1]+load > CAPACITY:
               break
            route.append(chossess)
            candi.remove((chossess[1], chossess[0]))
            candi.remove(chossess)
            start = chossess[1]
            load += TABLE[chossess[0]][ chossess[1]][1]
            if load>=CAPACITY:
                break
        S.append(route)

    return S


def imdinit(tb, S):

    vio = 0
    t1=0
    while t1<len(S):
        route=S[t1]
        load = 0
        t2=0
        while t2<len(route):
            task=route[t2]
            load += TABLE[task[0]][task[1]][1]
            t2+=1
        viorou = max(0, load - CAPACITY)
        vio += viorou
        t1+=1
    totalv =vio

    pay = 0
    t3=0
    while t3<len(S):
        route=S[t3]

        start = DEPOT
        t4=0
        while t4<len(route):

            pay += DISTANCE[start][task[0]]
            pay += DISTANCE[task[0]][task[1]]
            start = task[1]
            t4+=1
        pay += DISTANCE[start][DEPOT]
        t3+=1
    totalc=pay
    lmd = (tb / CAPACITY) * ((tb / totalc) + (totalv / CAPACITY) + 1)
    return lmd





def first(cost_with_ans):
    global BEST_FEASIBLE_SOLUTION
    global pop_t
    a = random.randint(0, len(cost_with_ans) - 1)
    b = random.randint(0, len(cost_with_ans) - 1)
    while a == b:
        b = random.randint(0, len(cost_with_ans) - 1)
    s_x = crv(cost_with_ans[a], cost_with_ans[b])
    lmd = imdinit(BEST_FEASIBLE_SOLUTION['pay'], s_x)
    boo=cl(s_x, pop_t);
    if not boo:
        heapq.heappush(pop_t, (es(s_x, lmd), s_x))

    b1=False
    fin_ed = 0
    for t__route in pop_t[0][1]:
        fin_ed += len(t__route)
    if fin_ed == REQUIRED_EDGES:
        b1=True
    if b1 :
        if pop_t[0][0] < BEST_FEASIBLE_SOLUTION['pay']:
            BEST_FEASIBLE_SOLUTION['pay'] = pop_t[0][0]
            BEST_FEASIBLE_SOLUTION['best'] = pop_t[0][1]




if __name__ == '__main__':

    FILEPATH, TIME, SEED = arg_analysis()
    START_TIME = time.time()

    TABLE,TABLE_COST,DEMAND=read(FILEPATH)
    DISTANCE=floyd(TABLE_COST)


    tasks = copy.deepcopy(TASKS)
    pop = popinit(tasks)
    pop_t = []
    for p in pop:
        heapq.heappush(pop_t, (es(p, 0), p))
    for p in pop:
        b5=False
        fin_ed = 0
        for route in p:
            fin_ed += len(route)
        if fin_ed == REQUIRED_EDGES:
            b5=True
        if b5:
            pay = 0
            for route in p:
                start = DEPOT
                for task in route:
                    pay += DISTANCE[start][task[0]]
                    pay += DISTANCE[task[0]][task[1]]
                    start = task[1]
                pay += DISTANCE[start][DEPOT]
            BEST_FEASIBLE_SOLUTION['pay'] = pay
            BEST_FEASIBLE_SOLUTION['best'] = p
            break

    # print(TIME)
    while True:
        first(pop)

        if TIME - (time.time() - START_TIME) < 10:
            break
    # print
    S = BEST_FEASIBLE_SOLUTION['best']
    pay = 0
    routes = []
    t1=0
    while t1<len(S):
        route=S[t1]
        if len(route) == 0:
            continue
        start = DEPOT
        routes.append(0)
        routes += route
        t2=0
        while t2<len(route):
            task=route[t2]
            pay += DISTANCE[start][task[0]]
            pay += TABLE[task[0]][task[1]][0]
            start = task[1]
            t2+=1
        pay += DISTANCE[start][DEPOT]
        routes.append(0)
        t1+=1
    r = ','.join((str(item).replace(" ", "") for item in routes))

    print("s", r)
    print("q", pay)




