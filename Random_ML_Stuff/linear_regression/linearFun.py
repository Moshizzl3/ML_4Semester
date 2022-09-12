
ageX = [1, 3, 5, 7, 9, 11, 13]
heightY = [75, 92, 108, 121, 130, 142, 155]


def calcA(x, y):
    resultX = 0
    resultXpow = 0
    resultY = 0
    resultXy = 0

    for n in x:
        resultX += n
        resultXpow += n**2

    for n in y:
        resultY += n

    for index, n in enumerate(y):
        resultXy += n * x[index]

    return ((resultY * resultXpow) - (resultX * resultXy)) / (len(x) * (resultXpow) - resultX**2)


def calcB(x, y):

    resultX = 0
    resultXpow = 0
    resultY = 0
    resultXy = 0

    for n in x:
        resultX += n
        resultXpow += n**2

    for n in y:
        resultY += n

    for index, n in enumerate(y):
        resultXy += n * x[index]

    return ((len(x) * resultXy) - (resultX * resultY)) / (len(x) * resultXpow - resultX**2)


def calcValue(x, dataSet1, dataSet2):

    return calcA(dataSet1, dataSet2) + calcB(dataSet1, dataSet2) * x


print(calcA(ageX, heightY))
print(calcB(ageX, heightY))
print(calcValue(1, ageX, heightY))
