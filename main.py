# ,1,,3
import math

import scipy
from terminaltables import AsciiTable
import matplotlib.pyplot as plt


def printTable(name, table_data):
    table1 = AsciiTable(table_data)
    print(name)
    print(table1.table)


data = sorted([
    -1.006, 0.386, -1.223, -0.591, -0.345, 0.157, 0.800, -0.155, -0.379, -1.023,
    1.306, -0.861, 0.303, 0.518, 0.986, 0.788, 0.883, -0.098, -0.242, 1.701,
    1.199, -1.230, -0.730, -1.492, 0.643, -0.577, -0.224, 0.997, -1.165, -0.494,
    -2.577, 2.641, -1.143, -0.086, 2.919, 0.527, 0.297, 0.434, 0.756, 0.172,
    -2.086, -0.904, -1.413, -0.012, -1.248, 1.671, -0.521, -0.025, 1.164, 0.354,
    0.866, -0.005, 0.403, 1.908, 0.448, 0.169, -0.731, -1.189, 0.905, 0.283,
    2.431, 1.409, 0.191, -0.165, 0.889, 0.804, -2.131, -0.754, 1.458, 1.650,
    0.026, 0.885, 0.011, -0.990, -0.104, 0.174, -0.052, -0.182, 1.813, 0.346,
    0.110, 1.757, -0.693, -0.732, 1.073, -1.724, -1.810, 0.947, -1.118, 0.666,
    0.970, 1.140, -1.105, 0.894, 1.547, -0.484, -0.086, -0.066, 0.150, -0.264
])
N = len(data)
intervalsCount = 13
negativeItems = 7
positiveItems = 7

maxItem = max(data)
minItem = min(data)

h = (maxItem - minItem) / intervalsCount

a = round(h * -negativeItems, 3)
b = round(h * positiveItems, 3)

groups = []
dataGrouped = [[]]

last = 0
for i in range(1, negativeItems + 1, 1):
    groups.append((a + h*(i - 1), a + h*i))
    if i == negativeItems:
        last = a + h*i

for i in range(1, positiveItems + 1, 1):
    groups.append((last + h*(i - 1), last + h*i))

groups = sorted(groups)

groupNum = 0
for item in data:
    leftBound = groups[groupNum][0]
    rightBound = groups[groupNum][1]
    if leftBound <= item < rightBound:
        dataGrouped[groupNum].append(item)
    elif groupNum == len(groups) - 1 and item <= rightBound:
        dataGrouped[groupNum].append(item)
    else:
        groupNum += 1
        dataGrouped.append([])
        dataGrouped[groupNum].append(item)

headers = ["Номера интервалов"]
n = ["Число элементов в интервале"]
pArr = ["Относительные частоты"]
pPercents = ["Относительные частоты процентные"]
mediansRow = ["Середины интервалов"]
medians = []
bounds = ["Границы интервалов"]
barTitles = []

for i in range(len(dataGrouped)):
    group = dataGrouped[i]
    bound = groups[i]
    headers.append(i + 1)
    ni = len(group)
    median = (bound[0] + bound[1]) / 2
    bounds.append((round(bound[0], 3), round(bound[1], 3)))
    mediansRow.append(round(median, 3))
    medians.append(median)
    barTitles.append(str((round(bound[0], 2), round(bound[1], 2))))
    n.append(ni)
    pArr.append(ni / N)
    pPercents.append(str(round(ni / N * 100, 1)) + "%")


table_data = [
    headers,
    bounds,
    mediansRow,
    n,
    pArr,
    pPercents
]

printTable("Таблица 1", table_data)

bounds.pop(0)
n.pop(0)
pArr.pop(0)

fig, ax = plt.subplots(figsize=(24,12))
plt.title("Рис. 1")
plt.bar(barTitles, n)

fig, ax = plt.subplots(figsize=(8,4))
plt.title("Рис. 2")


def countElementsLower(bound):
    return len(list(filter(lambda x: x <= bound, data)))


def Laps(x):
    return scipy.stats.norm.cdf(x)


for i in range(len(bounds)):
    bound = bounds[i]
    F = countElementsLower(bound[1]) / N
    plt.plot([bound[0], bound[1]], [F, F])

plt.show()

XPArr = []
XPRow = []
X2PArr = []
X2PRow = []

for i in range(len(medians)):
    median = medians[i]
    p = pArr[i]
    xp = median*p
    x2p = (median**2)*p
    XPArr.append(xp)
    X2PArr.append(x2p)
    XPRow.append(round(xp, 3))
    X2PRow.append(round(x2p, 3))

mediansRow.pop(0)
X = sum(XPArr)
M2 = sum(X2PArr)
XPRow = ["Xi * Pi"] + XPRow + ["X = " + str(round(X, 3))]
X2PRow = ["Xi^2 * Pi"] + X2PRow + ["M2 = " + str(round(M2, 3))]
XiRow = ["Xi"] + mediansRow
PiRow = ["Pi"] + pArr
headers.append("Некоторые результаты")

second_table_data = [
    headers,
    XiRow,
    PiRow,
    XPRow,
    X2PRow
]

printTable("Таблица 2", second_table_data)
S2 = M2 - X**2
print("X = " + str(round(X, 3)) + "; ", "S^2 = M2 - X^2 = " + str(round(S2, 3)))

t = 1.95

interval = (
    X - t * (math.sqrt(S2) / math.sqrt(N)),
    X + t * (math.sqrt(S2) / math.sqrt(N)),
)
print("Доверительный интервал для мат ожидания: ")
print(str(round(interval[0], 3)), "<", "M", "<", str(round(interval[1], 3)))

#TODO: Тут возможно надо будет переделать, я смотрел по таблице какие интервалы надо объединять
#Сначала интервалы по значениям
dataGrouped[0].extend(dataGrouped[1])
dataGrouped[0].extend(dataGrouped[2])
dataGrouped[0].extend(dataGrouped[3])
dataGrouped.remove(dataGrouped[1])
dataGrouped.remove(dataGrouped[1])
dataGrouped.remove(dataGrouped[1])

dataGrouped[8].extend(dataGrouped[9])
dataGrouped[8].extend(dataGrouped[10])
dataGrouped.remove(dataGrouped[10])
dataGrouped.remove(dataGrouped[9])


# А потом интевалы по границам
groups[0] = (groups[0][0], groups[3][1])
groups.remove(groups[1])
groups.remove(groups[1])
groups.remove(groups[1])

groups[8] = (groups[8][0], groups[10][1])
groups.remove(groups[10])
groups.remove(groups[9])

K = len(dataGrouped)
r = K - 3
p = 0.05


zArgs = []
z = []


def calculateP(i, pair):
    fPair = Laps(pair)
    if i == 0:
        fPair = Laps((-math.inf, pair[1]))
    elif i == len(groups) - 1:
        fPair = Laps((pair[0], math.inf))

    return fPair[1] - fPair[0]


X_ = 0
for i in range(K):
    bound = groups[i]
    zArgs.append(((bound[0] - X) / math.sqrt(S2), (bound[1] - X) / math.sqrt(S2)))
    ni = len(dataGrouped[i])
    X_ += (ni**2) / (N * calculateP(i, zArgs[i]))

X_ = round(X_ - N, 3)

print("Xˆ2 =", X_, "; ", X_, "< 12.592 --> гипотеза о нормальном распределении генеральной совокупности не отвергается.")