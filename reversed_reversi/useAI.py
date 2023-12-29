import numpy as np

import main

ai=main.AI(8,-1,5)
r=(0,0,0,0,0,0,0,0,0,0)
chessboard = np.array([[r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]],
                           [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                           [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                           [r[3], r[6], r[8], 1, -1, r[8], r[6], r[3]],
                           [r[3], r[6], r[8], -1, 1, r[8], r[6], r[3]],
                           [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                           [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                           [r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]]])

import time

time_start=time.time()

ai.go(chessboard)

time_end=time.time()
print('time cost',time_end-time_start,'s')


