import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.porter import *
from collections import Counter
import xml.etree.ElementTree as ET

######LAB1########
def Preprocessing(fileToRead, STFile):
    file = open(fileToRead).read()
    stop_words = open(STFile).read().split('\n')
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(i) for i in re.split('[^\w^\d]',file.lower()) if i not in stop_words and i !='']
    return tokens

def TextLawsZipf(fileToRead, STFile):
    tokens = Preprocessing(fileToRead, STFile)
    counts = Counter(tokens)
    print(counts)
    data = counts.most_common(300)
    plt.figure()
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]
    plt.loglog(x_val, y_val)
    plt.xticks([])
    plt.ylabel("Frequency")
    plt.show()


def TextLawsBeford(fileToRead, STFile):
    tokens = Preprocessing(fileToRead, STFile)
    counts = Counter(tokens)
    digits = list(counts.values())

    #Remove one digit frequencies
    for each in digits:
        if len(str(each)) <= 1:
            digits.remove(each)
    ###############################

    for x in range(len(digits)):
            digits[x] = int(str(digits[x])[:1])
    y_val = []
    for i in range(1,10):
        y_val.append(digits.count(i))
    plt.figure()
    plt.bar(range(1,10), y_val)
    plt.show()

def TextLawsHeap(fileToRead, STFile):
    tokens = Preprocessing(fileToRead, STFile)
    n = 0 #Number of terms
    v = 0 #Number of vocabulray
    nvPairs = []
    dict = {}
    for each in tokens:
        n = n + 1
        if each in dict:
            dict[each] = dict[each] + 1
        else:
            dict[each] = 1
            v = v + 1
        if n % 1000 == 0:
            nvPairs.append((n,v))
    x, y = zip(*nvPairs)
    plt.figure()
    plt.scatter(x,y)

    k = 6.5
    b = 0.585
    n = np.linspace(0, x[-1], int(x[-1]/100))
    v = k*n**b
    plt.plot(n,v)

    plt.show()


######LAB2########
def PositionalInvertedIndex(fileName, STFile):
    tree = ET.parse(fileName)
    root = tree.getroot()
    positionalIdx = {}
    fileToWrite = open("results.txt","w")

    for i in range (len(root)): #For each document
        stop_words = open(STFile).read().split('\n')
        #file = (root[i].find("TEXT").text)
        file = (root[i].find("HEADLINE").text + root[i].find("TEXT").text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(i) for i in re.split('[^\w^\d]', file.lower()) if i not in stop_words and i != '']
        #tokens = [i for i in re.split('[^\w^\d]', file.lower()) if i != '']

        fileNo = int(root[i][0].text)
        for pos, term in enumerate(tokens):
            # If this word occured before
            if term in positionalIdx:
                # if this word occured in this document before
                if fileNo in positionalIdx[term][1]:
                    positionalIdx[term][1][fileNo].append(pos+1)
                else:
                    positionalIdx[term][1][fileNo] = [pos+1]
                    # Increase document count
                    positionalIdx[term][0] = positionalIdx[term][0] + 1
            # If this word appears first time
            else:
                positionalIdx[term] = []
                positionalIdx[term].append(1)
                positionalIdx[term].append({})
                positionalIdx[term][1][fileNo] = [pos+1]
    sortedPosIdx = {}
    for i in sorted(positionalIdx):
        sortedPosIdx[i] = positionalIdx[i]

    for eachTerm in sortedPosIdx.keys():
        fileToWrite.write(eachTerm + ":" + str(sortedPosIdx[eachTerm][0]) + "\n")
        for eachDoc in sortedPosIdx[eachTerm][1]:
            fileToWrite.write("\t" + str(eachDoc) + ": " + str(sortedPosIdx[eachTerm][1][eachDoc])[1:-1] + "\n")
    return sortedPosIdx,int(root[-1][0].text)


def BuildCollectionMatrix(posIdx, numOfDocs):
    collectionMatrix = np.zeros((numOfDocs, len(posIdx)))
    for counter, eachTerm in enumerate(posIdx):
        for i in range(numOfDocs):
            if i in posIdx[eachTerm][1]:
                collectionMatrix[i][counter] = 1
    return collectionMatrix.transpose()

def BooleanSearch(positionalIdx, collectionMatrix, searchTerm):
    terms = searchTerm.split()
    for i in range(len(terms)):
        if terms[i] not in ["AND", "OR", "NOT"]:
            word = PorterStemmer().stem(terms[i].lower())
            # Find row index for that term in collection matrix (Include stemming and lower)
            idx = [idx for idx, key in enumerate(list(positionalIdx.items())) if key[0] == word]
            # Find the corresponding row vector in string
            string = "".join(str(int(i)) for i in collectionMatrix[idx].ravel())
            # Convert to base 2 string ints to be evaluated by eval
            terms[i] = str(int(string,base=2))
        elif terms[i] == "AND":
            terms[i] = '&'
        elif terms[i] == "OR":
            terms[i] = '|'
        elif terms[i] == "NOT":
            terms[i] = '~'
    result = eval("".join(terms))
    # A bit vector representing valid(1)/invalid(0) documents
    bitVector = "{:b}".format(result)
    searchResult = []
    # Fill in offsets
    while len(bitVector) < len(collectionMatrix[0]):
        bitVector = '0' + bitVector
    for i in range(len(bitVector)):
        if bitVector[i] == '1':
            searchResult.append(i)
    return searchResult


def PhraseSearch(positionalIdx, collectionMatrix, searchTerm):
    phraseToSearch = []
    phraseWords = []
    terms = searchTerm.split()
    # Combine phrases
    i = 0
    while i < len(terms):
        if terms[i][0] == "\"":
            phraseToSearch.append(terms[i][1:] + " AND " + terms[i+1][:-1])
            phraseWords.append(PorterStemmer().stem(terms[i][1:].lower()))
            phraseWords.append(PorterStemmer().stem(terms[i+1][:-1].lower()))
            i = i + 2
        else:
            phraseToSearch.append(terms[i])
            i = i + 1
    searchResult = BooleanSearch(positionalIdx,collectionMatrix, " ".join(phraseToSearch))
    # List of document ID and indexes for the corresponding word
    firstWord = positionalIdx[phraseWords[0]][1]
    secondWord = positionalIdx[phraseWords[1]][1]
    correctPhrase = []
    # Only keep documents where the two word phrases are correct
    for eachDoc in searchResult:
        for eachIndex in firstWord[eachDoc]:
            if eachIndex + 1 in secondWord[eachDoc]:
                correctPhrase.append(eachDoc)
                break
    return correctPhrase

def ProximitySearch(positionalIdx, collectionMatrix, searchTerm):
    terms = searchTerm.split()
    phraseToSearch = []
    phraseWords = []
    dist = ""
    i = 0
    while i < len(terms):
        if terms[i][0] == "#":
            j = 1
            while terms[i][j] != "(":
                dist = dist + terms[i][j]
                j = j + 1
            phraseToSearch.append(terms[i][j+1:-1] + " AND " + terms[i+1][:-1])
            phraseWords.append(PorterStemmer().stem(terms[i][j+1:-1].lower()))
            phraseWords.append(PorterStemmer().stem(terms[i+1][:-1].lower()))
            i = i + 2
        else:
            phraseToSearch.append(terms[i])
            i = i + 1
    searchResult = BooleanSearch(positionalIdx,collectionMatrix, " ".join(phraseToSearch))
    # List of document ID and indexes for the corresponding word
    firstWord = positionalIdx[phraseWords[0]][1]
    secondWord = positionalIdx[phraseWords[1]][1]
    correctPhrase = []
    # Only keep documents where the two word phrases are correct
    for eachDoc in searchResult:
        for eachIndex in firstWord[eachDoc]:
            if any(x in secondWord[eachDoc] for x in range(eachIndex-int(dist), eachIndex+int(dist))):
                correctPhrase.append(eachDoc)
                break
    return correctPhrase


#######Lab3##################
def tf(term, document, positionalIdx):
    if document not in positionalIdx[PorterStemmer().stem(term.lower())][1]:
        return 0
    return len(positionalIdx[PorterStemmer().stem(term.lower())][1][document])

def df(term, positionalIdx):
    return positionalIdx[PorterStemmer().stem(term.lower())][0]

def weight(term, document, positionalIdx, docCount):
    return (1 + np.log10(tf(term,document,positionalIdx))) * np.log10(docCount/df(term,positionalIdx))

def score(searchTerm, document, positionalIdx, docCount):
    result = 0
    for eachTerm in searchTerm:
        if tf(eachTerm, document, positionalIdx) != 0:
            result = result + weight(eachTerm, document, positionalIdx, docCount)
    return result

def RankedRetrieval(positionalIdx, collectionMatrix, searchTerm, docCount):
    terms = searchTerm.split()
    query = PorterStemmer().stem(terms[1].lower())
    for i in range(2, len(terms)):
        query = query + " OR " + PorterStemmer().stem(terms[i].lower())
    documents = BooleanSearch(positionalIdx, collectionMatrix, query)
    scores = []
    for eachDoc in documents:
        scores.append((eachDoc, score(terms[1:],eachDoc,positionalIdx,docCount)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main():
    '''
    #Lab1
    print(Preprocessing("pg10.txt", "englishST.txt"))
    print(Preprocessing("quran.txt", "englishST.txt"))
    print(Preprocessing("abstracts.wiki.txt", "englishST.txt"))
    TextLawsHeap("quran.txt", "englishST.txt")
    '''

    '''
    #Lab2
    '''
    positionalIdx, numOfDocs = PositionalInvertedIndex("trec.sample.xml", "englishST.txt")
    collectionMatrix = BuildCollectionMatrix(positionalIdx, numOfDocs)
    #print(PhraseSearch(positionalIdx, collectionMatrix, "\"middle east\" AND peace"))
    #print(BooleanSearch(positionalIdx, collectionMatrix, ""))
    #print(ProximitySearch(positionalIdx, collectionMatrix, "#10(income, taxes)"))

    #Lab3
    print(RankedRetrieval(positionalIdx, collectionMatrix, "1 unemployment rate UK", 1000))




if __name__ == "__main__":
    main()

