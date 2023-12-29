import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)





# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        ##I add==========================================================================
        self.chessboard = None
        ##chessboard weight
        r=(-150,48,-8,6,-8,-16,3,4,4,0)
        self.chessboard_weight = np.array([[r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]],
                           [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                           [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                           [r[3], r[6], r[8], r[9], r[9], r[8], r[6], r[3]],
                           [r[3], r[6], r[8], r[9], r[9], r[8], r[6], r[3]],
                           [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                           [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                           [r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]]])
        self.UTILITY_THRESHOLD=3
        self.SEARCH_DEPTH=8
        self.INF = 1e+8

        self.mincount = 0
        self.maxcount = 0

        self.mobility_weight = 0.5
        self.frontier_weight = 0.3
        self.stability_weight = 1
        self.position_value_weight = 0.7


    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.chessboard = chessboard
        # ==================================================================
        # Write your algorithm here
        sum_of_blank_points=np.sum(chessboard==COLOR_NONE)
        valid_points=self.find_points(chessboard,self.color)
        if not valid_points:
            return []
        else:
            random.shuffle(valid_points)
            self.candidate_list=valid_points

        self.candidate_list = self.min_max_decision(900, valid_points)

        # if sum_of_blank_points<=self.UTILITY_THRESHOLD:
        #     self.candidate_list=self.min_max_decision(self.UTILITY_THRESHOLD,valid_points)
        # else:
        #     self.candidate_list = self.min_max_decision(self.SEARCH_DEPTH, valid_points)






        # Here is the simplest sample:Random decision
        # idx = np.where(chessboard == COLOR_NONE)
        # idx = list(zip(idx[0], idx[1]))

    def find_points(self, chessboard, color):
        i = 0
        choice=[]
        while i < self.chessboard_size:
            j = 0
            while j < self.chessboard_size:
                ij = 0
                if chessboard[i][j] == 0:
                    # up
                    if ij == 0 and j - 2 >= 0 and chessboard[i][j - 1] == -color:
                        temp = j - 2
                        while temp >= 0:
                            if chessboard[i][temp] == 0:
                                break
                            if chessboard[i][temp] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i,j))
                                ij = 1
                                break
                            temp = temp - 1

                    # down
                    if ij == 0 and j + 2 < self.chessboard_size and chessboard[i][j + 1] == -color:
                        temp = j + 2
                        while temp < self.chessboard_size:
                            if chessboard[i][temp] == 0:
                                break
                            if chessboard[i][temp] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            temp = temp + 1
                    # left
                    if ij == 0 and i - 2 >= 0 and chessboard[i - 1][j] == -color:
                        temp = i - 2
                        while temp >= 0:
                            if chessboard[temp][j] == 0:
                                break
                            if chessboard[temp][j] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            temp = temp - 1
                    # right
                    if ij == 0 and i + 2 < self.chessboard_size and chessboard[i + 1][j] == -color:
                        temp = i + 2
                        while temp < self.chessboard_size:
                            if chessboard[temp][j] == 0:
                                break
                            if chessboard[temp][j] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            temp = temp + 1
                    ##left up
                    if ij == 0 and i - 2 >= 0 and j - 2 >= 0 and chessboard[i - 1][j - 1] == -color:
                        tempi = i - 2
                        tempj = j - 2
                        while tempi >= 0 and tempj >= 0:
                            if chessboard[tempi][tempj] == 0:
                                break
                            if chessboard[tempi][tempj] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            tempi = tempi - 1
                            tempj = tempj - 1
                    ##right up
                    if ij == 0 and i + 2 <self.chessboard_size and j - 2 >= 0 and chessboard[i + 1][j - 1] == -color:
                        tempi = i + 2
                        tempj = j - 2
                        while tempi <self.chessboard_size and tempj >= 0:
                            if chessboard[tempi][tempj] == 0:
                                break
                            if chessboard[tempi][tempj] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            tempi = tempi + 1
                            tempj = tempj - 1
                    ##left down
                    if ij == 0 and i - 2 >= 0 and j + 2 <self.chessboard_size and chessboard[i - 1][j + 1] == -color:
                        tempi = i - 2
                        tempj = j + 2
                        while tempi >= 0 and tempj <self.chessboard_size:
                            if chessboard[tempi][tempj] == 0:
                                break
                            if chessboard[tempi][tempj] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            tempi = tempi - 1
                            tempj = tempj + 1
                    ##right down
                    if ij == 0 and i + 2 <self.chessboard_size and j + 2 <self.chessboard_size and chessboard[i + 1][j + 1] == -color:
                        tempi = i + 2
                        tempj = j + 2
                        while tempi <self.chessboard_size and tempj <self.chessboard_size:
                            if chessboard[tempi][tempj] == 0:
                                break
                            if chessboard[tempi][tempj] == color:
                                self.candidate_list.append((i, j))
                                choice.append((i, j))
                                ij = 1
                                break
                            tempi = tempi + 1
                            tempj = tempj + 1
                j=j+1
            i=i+1
        return choice

    def min_max_decision(self,depth,valid_points):

        alpha=-self.INF
        value_point_list=[]
        if depth==0:
            for point in valid_points:
                update_list = self.update_chessboard(self.chessboard, point, self.color)

                value=self.evaluate_function(point)
                self.revert_chessboard(self.chessboard,point,self.color,update_list)
                value_point_list.append((value,point))
        else:
            depth=depth-1
            for point in valid_points:
                update_list = self.update_chessboard(self.chessboard, point, self.color)
                value= self.min_value_generating(point,-self.color, depth, alpha, self.INF)
                alpha = max(value, alpha)
                self.revert_chessboard(self.chessboard,point,self.color,update_list)
                value_point_list.append((value,point))
        value_point_list.sort(key=lambda elem:elem[0])
        return [elem[1] for elem in value_point_list]







    def evaluate_function(self,point):
        #mobility
        mobility=0
        valid_points = self.find_points(self.chessboard, self.color)
        mobility=len(valid_points)

        #position value
        # frontier

        position_value=0
        frontier=0
        stability=0

        sum_position_value=0;
        counter=0
        i=0
        while i <self.chessboard_size:
            j=0
            while j<self.chessboard_size:
                if self.chessboard[i][j]==self.color:
                    sum_position_value=sum_position_value+self.chessboard_weight[i][j]
                    counter=counter+1
                    for direction in ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)):
                        if i+direction[0]>=0 and i+direction[0]<8 and j+direction[1]>=0 and j+direction[1]<8:
                            if self.chessboard[i+direction[0]][j+direction[1]]==COLOR_NONE:
                                frontier=frontier+1
                                break



                j=j+1
            i=i+1
        # stability
        for position in ((0,0),(0,7),(7,0),(7,7)):
            if  self.chessboard[position[0]][position[1]]==self.color:
                stability=stability+1
        position_value=sum_position_value/counter

        value=self.mobility_weight*mobility+self.position_value_weight*position_value-self.frontier_weight*frontier-self.stability_weight*stability

        print(value)
        return value

    def update_chessboard(self,chessboard, point, player):
        update_list=self.get_update_chessboard_list(chessboard, point, player)
        chessboard[point] = player
        for point in update_list:
            chessboard[point] = player
        return update_list




    def get_update_chessboard_list(self,chessboard, point, player):
        update_list=[]
        i=0
        while i < self.chessboard_size:
            j = 0
            while j < self.chessboard_size:
                if chessboard[i][j]==player:
                    x=i-point[0]
                    y=j-point[1]
                    if x==0 and y==0:
                        continue
                    elif x==0 and y!=0:
                        if y>0:
                            for k in range(1,y):

                                if chessboard[point[0]][point[1]+k]==-player:
                                    update_list.append((point[0],point[1]+k))
                        else:
                            for k in range(1,-y):
                                if chessboard[point[0]][point[1]-k]==-player:
                                    update_list.append((point[0],point[1]-k))
                    elif x!=0 and y==0:
                        if x>0:
                            for k in range(1,x):
                                if chessboard[point[0]+k][point[1]]==-player:
                                    update_list.append((point[0]+k,point[1]))
                        else:
                            for k in range(1,-x):
                                if chessboard[point[0]-k][point[1]]==-player:
                                    update_list.append((point[0]-k,point[1]))
                    elif abs(x)==abs(y):
                        if x>0 and y>0:
                            for k in range(1,x):
                                if chessboard[point[0]+k][point[1]+k]==-player:
                                    update_list.append((point[0]+k,point[1]+k))
                        elif x<0 and y<0:
                            for k in range(1,-x):
                                if chessboard[point[0]-k][point[1]-k]==-player:
                                    update_list.append((point[0]-k,point[1]-k))


                        elif x>0 and y<0:
                            for k in range(1,x):
                                if chessboard[point[0]+k][point[1]-k]==-player:
                                    update_list.append((point[0]+k,point[1]-k))

                        elif x<0 and y>0:
                            for k in range(1,-x):
                                if chessboard[point[0]-k][point[1]+k]==-player:
                                    update_list.append((point[0]-k,point[1]+k))
                j=j+1
            i=i+1
        update_list=list(set(update_list))
        return update_list



    def revert_chessboard(self,chessboard, point, player, update_array):
        chessboard[point] = COLOR_NONE
        for point in update_array:
            chessboard[point[0], point[1]] = -player


    def min_value_generating(self, point, player, depth, alpha, beta):
        self.mincount=self.mincount+1
        print("min_value_generating",self.mincount)
        if depth==0:
            return self.evaluate_function(point)
        value=self.INF
        blank_points_index = np.where(self.chessboard == COLOR_NONE)
        blank_points = zip(blank_points_index[0], blank_points_index[1])
        no_step = True
        if blank_points_index[0].size>0:
            for point in blank_points:
                if self.is_legal_point(self.chessboard, point, player,self.color):
                    no_step=False

                    update_array = self.update_chessboard(self.chessboard, point, self.color)
                    value = min(value, self.max_value_generating(point,-player, depth - 1, alpha, beta))
                    self.revert_chessboard(self.chessboard, point, player, update_array)

                    if value <= alpha:
                        return value
                    beta = min(value, beta)

            if no_step:
                return self.max_value_generating(point,-player, depth - 1, alpha, beta)
        else:
            return self.evaluate_function(point)

        return value

    def max_value_generating(self,point, player, depth, alpha, beta):
        self.maxcount = self.maxcount + 1
        print("max_value_generating", self.maxcount)
        if depth == 0:
            return self.evaluate_function(point)
        value = -self.INF

        blank_points_index = np.where(self.chessboard == COLOR_NONE)
        blank_points = zip(blank_points_index[0], blank_points_index[1])
        no_step = True

        if blank_points_index[0].size > 0:
            for point in blank_points:
                if self.is_legal_point(self.chessboard, point, player,self.color):
                    no_step = False

                    update_array = self.update_chessboard(self.chessboard, point, self.color)
                    value = max(value, self.min_value_generating(point,-player, depth - 1, alpha, beta))
                    self.revert_chessboard(self.chessboard, point, player, update_array)

                    if value >= beta:
                        return value
                    alpha = max(value, alpha)
            if no_step:
                return self.min_value_generating(point,-player, depth - 1, alpha, beta)
        else:
            return self.evaluate_function(point)
        return value


    def is_legal_point(self,chessboard, point, player,color):
        i=point[0]
        j=point[1]
        ij=0
        # up
        if ij == 0 and j - 2 >= 0 and chessboard[i][j - 1] == -color:
            temp = j - 2
            while temp >= 0:
                if chessboard[i][temp] == 0:
                    break
                if chessboard[i][temp] == color:
                    return True

                temp = temp - 1

        # down
        if ij == 0 and j + 2 < self.chessboard_size and chessboard[i][j + 1] == -color:
            temp = j + 2
            while temp < self.chessboard_size:
                if chessboard[i][temp] == 0:
                    break
                if chessboard[i][temp] == color:
                    return True
                temp = temp + 1
        # left
        if ij == 0 and i - 2 >= 0 and chessboard[i - 1][j] == -color:
            temp = i - 2
            while temp >= 0:
                if chessboard[temp][j] == 0:
                    break
                if chessboard[temp][j] == color:
                    return True
                temp = temp - 1
        # right
        if ij == 0 and i + 2 < self.chessboard_size and chessboard[i + 1][j] == -color:
            temp = i + 2
            while temp < self.chessboard_size:
                if chessboard[temp][j] == 0:
                    break
                if chessboard[temp][j] == color:
                    return True
                temp = temp + 1
        ##left up
        if ij == 0 and i - 2 >= 0 and j - 2 >= 0 and chessboard[i - 1][j - 1] == -color:
            tempi = i - 2
            tempj = j - 2
            while tempi >= 0 and tempj >= 0:
                if chessboard[tempi][tempj] == 0:
                    break
                if chessboard[tempi][tempj] == color:
                    return True
                tempi = tempi - 1
                tempj = tempj - 1
        ##right up
        if ij == 0 and i + 2 < self.chessboard_size and j - 2 >= 0 and chessboard[i + 1][j - 1] == -color:
            tempi = i + 2
            tempj = j - 2
            while tempi < self.chessboard_size and tempj >= 0:
                if chessboard[tempi][tempj] == 0:
                    break
                if chessboard[tempi][tempj] == color:
                    return True
                tempi = tempi + 1
                tempj = tempj - 1
        ##left down
        if ij == 0 and i - 2 >= 0 and j + 2 < self.chessboard_size and chessboard[i - 1][j + 1] == -color:
            tempi = i - 2
            tempj = j + 2
            while tempi >= 0 and tempj < self.chessboard_size:
                if chessboard[tempi][tempj] == 0:
                    break
                if chessboard[tempi][tempj] == color:
                    return True
                tempi = tempi - 1
                tempj = tempj + 1
        ##right down
        if ij == 0 and i + 2 < self.chessboard_size and j + 2 < self.chessboard_size and chessboard[i + 1][
            j + 1] == -color:
            tempi = i + 2
            tempj = j + 2
            while tempi < self.chessboard_size and tempj < self.chessboard_size:
                if chessboard[tempi][tempj] == 0:
                    break
                if chessboard[tempi][tempj] == color:
                    return True
                tempi = tempi + 1
                tempj = tempj + 1
        return False