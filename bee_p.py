import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import animation
# 関数の設定
# xはnp.array

#sphere
def func(x):
    return np.sum(x**2)

# 初期設定
N = 100 # 個体数
d = 2 # 次元
TC = np.zeros(N) #更新カウント
lim = 30
xmax = 5
xmin = -5
G = 300 # 繰り返す回数

x = np.zeros((N,d))
for i in range(N):
    x[i] = (xmax-xmin)*np.random.rand(d) + xmin

# ルーレット選択用関数
def roulette_choice(w):
    tot = []
    acc = 0
    for e in w:
        acc += e
        tot.append(acc)

    r = np.random.random() * acc
    for i, e in enumerate(tot):
        if r <= e:
            return i

x_best = x[0] #x_bestの初期化
best = 100

# 繰り返し
fig = plt.figure()
best_value = []
ims = []
for g in range(G):

    centroid = 5*math.sin(0.07*g)

    def func(x):
        return np.sum((x-centroid)**2)

    best_value.append(best)

    z = np.zeros(N)
    for i in range(N):
        z[i] = func(x[i])


    img2 = plt.scatter(centroid,centroid,color='black')
    img1 = plt.scatter(x[:,0],x[:,1],marker='h',color='gold')
    ims.append([img1]+[img2])


    # employee bee step
    for i in range(N):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(N)


        for j in range(d):
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])

        if func(x[i]) > func(v[i]):
            x[i] = v[i]
            TC[i] = 0
        else: TC[i] += 1

    # onlooker bee step
    for i in range(N):
        w = []
        for j in range(N):
            w.append(np.exp(-func(x[j])))
        i = roulette_choice(w)
        for j in range(d):
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
        if func(x[i]) > func(v[i]):
            x[i] = v[i]
            TC[i] = 0
        else: TC[i] += 1



    # scout bee step
    for i in range(N):
        if TC[i] > lim:
            for j in range(d):
                x[i,j] = np.random.rand()*(xmax-xmin) + xmin
            TC[i] = 0

    # 最良個体の発見
    for i in range(N):
        if best > func(x[i]):
            x_best = x[i]
            best = func(x_best)



# 結果
print(x_best,func(x_best))


anim =  animation.ArtistAnimation(fig, ims, interval=70)
rc('animation', html='jshtml')
anim
