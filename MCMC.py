import numpy as np
import matplotlib.pyplot as plt
import math
import random

def CharacterList():
    symbol_indices = {}
    symbol_count = {}
    symbolsList = []

    with open('symbols.txt', 'r', encoding='utf-8') as symbols:
        index = 0

        for line in symbols:
            line = line.strip('\n')
            symbolsList.append(line)
            symbol_count[line] = 0
            symbol_indices[line] = index
            index += 1
    
    symbols.close()

    encryptedMessage = open('message.txt', 'r', encoding='utf-8')
    encryptedString = []

    while True:
        char = encryptedMessage.read(1)
        char = char.lower()

        encryptedString.append(char)

        if not char:
            break

    encryptedString = encryptedString[:-1]

    encryptedMessage.close()

    training_text = open('war_and_peace.txt', 'r', encoding='utf-8')
    training_string = []

    while True:
        char = training_text.read(1)
        char = char.lower()

        training_string.append(char)

        if not char:
            break

    training_text.close()

    for n, i in enumerate(training_string):
        if i == '\n':
            training_string[n] = " "
        if i == '’':
            training_string[n] = "'"
        if i == '—':
            training_string[n] = "-"
        if i == '‘':
            training_string[n] = "'"
        if i == '“':
            training_string[n] = "'"
        if i == '”':
            training_string[n] = "'"
        if i == 'á':
            training_string[n] = "a"
        if i == 'à':
            training_string[n] = "a"
        if i == 'â':
            training_string[n] = "a"
        if i == 'ä':
            training_string[n] = "a"
        if i == 'é':
            training_string[n] = "e"
        if i == 'ë':
            training_string[n] = "e"
        if i == 'è':
            training_string[n] = "e"
        if i == 'ê':
            training_string[n] = "e"
        if i == 'í':
            training_string[n] = "i"
        if i == 'ï':
            training_string[n] = "i"
        if i == 'î':
            training_string[n] = "i"
        if i == 'ó':
            training_string[n] = "o"
        if i == 'ô':
            training_string[n] = "o"
        if i == 'ö':
            training_string[n] = "o"
        if i == 'ù':
            training_string[n] = "u"
        if i == 'û':
            training_string[n] = "u"
        if i == 'ü':
            training_string[n] = "u"
        if i == 'ú':
            training_string[n] = "u"
        if i == 'ç':
            training_string[n] = "c"
        if i == 'ý':
            training_string[n] = "y"
        if i == 'œ':
            training_string[n] = "o"
            training_string.insert(n, "e")
        if i == 'æ':
            training_string[n] = "a"
            training_string.insert(n, "e")
        else:
            continue

    return training_string, symbolsList, symbol_indices, encryptedString


def CharacterCount():
    trainingString, symbolList, symbol_indices, encryptedString = CharacterList()
    symbolCount = {}
    totalCount = 0

    for symbol in symbolList:
        symbolCount[symbol] = 0

    for character in trainingString:
        totalCount += 1
        if character in symbolList:
            symbolCount[character] += 1

        else:
            continue

    return symbolCount, totalCount
        


def ProduceTransitionMatrix():
    training_string, symbolList, symbol_indices, encryptedString = CharacterList()
    symbol_count = {}
    keys_list = list(symbol_indices)
    values_list = list(symbol_indices.values())
    symbol_total = 0
    transition_matrix = np.zeros((53, 53))
        
    for i in range(len(training_string) - 1):
        print(i)

        char1 = training_string[i]
        char2 = training_string[i+1]

        print(char1, char2)

        index1 = symbol_indices.get(char1)
        index2 = symbol_indices.get(char2)

        print(index1, index2)
        
        transition_matrix[index1][index2] += 1

    np.savetxt('trainsition_matrix.txt', transition_matrix)


def TransitionMatrix():
    trainingString, symbolList, symbol_indices, encryptedString = CharacterList()
    symbolCount, totalCount = CharacterCount()

    tm = np.loadtxt('trainsition_matrix.txt')
    tm = tm.astype(int)
    TM = np.zeros((53, 53))

    for i in range(len(tm)):
        for j in range(len(tm)):
            firstLetter = symbolList[i]
            letterCount = symbolCount[firstLetter]
            TM[i,j] = (tm[i, j]+1)/(letterCount+53)

    np.savetxt('transition_matrix_test.txt', TM, fmt='%2f')

    return TM


def TrainsitionMatrix():
    trainingString, symbolList, symbol_indices, encryptedString = CharacterList()
    symbolCount, totalCount = CharacterCount()

    tm = np.loadtxt('trainsition_matrix.txt')
    tm = tm.astype(int)

    for i in range(len(tm)):
        for j in range(len(tm)):
            tm[i,j] += 1

    np.savetxt('trainsition_matrix_test.txt', tm, fmt='%2f')

    return tm


def GenerateKey(symbolList):
    decryptionKey = symbolList.copy()
    random.shuffle(decryptionKey)

    return decryptionKey


def SampleAndSwap(decryptionKey):
    decryptionKey1 = decryptionKey.copy()
    decryptionIndices = np.arange(len(decryptionKey1))
    sampledIndices = random.sample(list(decryptionIndices), 2)
    
    decryptionKey1[sampledIndices[0]], decryptionKey1[sampledIndices[1]] = decryptionKey1[sampledIndices[1]], decryptionKey1[sampledIndices[0]]

    return decryptionKey1


def DecryptedString(encryptedMessage, symbolList, symbolIndices, decryptionKey):
    decryptedString = []

    for char in encryptedMessage:
        if char in symbolList:
            symbolIndex = symbolIndices[str(char)]
            decryptedChar = decryptionKey[symbolIndex]
            decryptedString.append(decryptedChar)
        else:
            break

    return decryptedString


def Score(decryptedString, symbolIndices, transitionMatrix, characterCount, totalCount):
    score = 0

    for i in range(len(decryptedString)-1):
        if i == 0:
            char = decryptedString[i]
            charCount = characterCount[char]
            statProb = charCount/totalCount
            score += math.log(statProb)

        else:
            char1 = decryptedString[i-1]
            char2 = decryptedString[i]

            symbolIndex1 = symbolIndices[char1]
            symbolIndex2 = symbolIndices[char2]

            prob = math.log(transitionMatrix[symbolIndex1, symbolIndex2])
            score += prob
        
    return score


def MetropolisHastings():
    trainingString, symbolList, symbolIndices, encryptedString = CharacterList()
    characterCount, totalCount = CharacterCount()
    transitionMatrix = TransitionMatrix()
    decryptionKey = GenerateKey(symbolList)
    decryptedString = DecryptedString(encryptedString, symbolList, symbolIndices, decryptionKey)
    score = Score(decryptedString, symbolIndices, transitionMatrix, characterCount, totalCount)
    print('Initial score is: ', score)

    for n in range(50000):
        newDecryptionKey = SampleAndSwap(decryptionKey)
        newDecryptedString = DecryptedString(encryptedString, symbolList, symbolIndices, newDecryptionKey)
        newScore = Score(newDecryptedString, symbolIndices, transitionMatrix, characterCount, totalCount)
        acceptanceProbability = np.exp(newScore - score)
        
        coin = np.random.uniform()
        if acceptanceProbability > coin:
            decryptionKey = newDecryptionKey
            score = newScore

        if n % 100 == 0:
            # print('Iteration is {}, score is {}.'.format(n, score))
            print(''.join(newDecryptedString[:60]))
            print('\n')

    finalDecryption = DecryptedString(encryptedString, symbolList, symbolIndices, newDecryptionKey)
    finalString = ''.join(finalDecryption)

    with open("decryptResult.txt", "w") as text_file:
        text_file.write(finalString)

    return finalString
    

def ScoreTest():
    trainingString, symbolList, symbolIndices, encryptedString = CharacterList()
    characterCount, totalCount = CharacterCount()
    transitionMatrix = TransitionMatrix()
    decryptedString1 = ['m', 'y', ' ', 'n', 'a', 'm', 'e', ' ', 'i', 's', ' ', 'l', 'u', 'k', 'e', ' ', 'a', 'n', 'd', ' ', 'i', ' ', 'r', 'e', 'a', 'l', 'l', 'y', ' ', 'h', 'o', 'p', 'e', ' ', 't', 'h', 'i', 's', ' ', 'w', 'o', 'r', 'k', 's']
    decryptedString2 = decryptedString1.copy()
    random.shuffle(decryptedString2)

    score1 = Score(decryptedString1, symbolIndices, transitionMatrix, characterCount, totalCount)
    print(score1)
    score2 = Score(decryptedString2, symbolIndices, transitionMatrix, characterCount, totalCount)
    print(score2)

trainingString, symbolList, symbolIndices, encryptedString = CharacterList()
transitionMatrix = TransitionMatrix()
# plt.imshow(transitionMatrix, cmap='Blues', interpolation='nearest')
# plt.xticks(range(len(symbolList)), symbolList, size='small')
# plt.yticks(range(len(symbolList)), symbolList, size='small')
# plt.show()

# score = Score(decryptedString, symbolIndices, transitionMatrix)
# print(score)
decryptedString = MetropolisHastings()
#ScoreTest()