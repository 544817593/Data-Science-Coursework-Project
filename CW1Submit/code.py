import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.porter import *
from collections import Counter
import xml.etree.ElementTree as ET

fileName = "trec.5000.xml" # XML collection
STFile = "englishST.txt" # Stop words file
queryFile = "queries.boolean.txt" # Query file
rankedFile = "queries.ranked.txt" # Ranked IR file
stemmer = PorterStemmer()

tree = ET.parse(fileName)
root = tree.getroot()
positionalIdx = {}
fileToWrite = open("index.txt","w")
for i in range (len(root)): #For each document
    stop_words = open(STFile).read().split('\n')
    file = (root[i].find("HEADLINE").text + root[i].find("TEXT").text)
    tokens = [stemmer.stem(i) for i in re.split('[^\w^\d]', file.lower()) if i not in stop_words and i != '']

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

# Sort positional index
sortedPosIdx = {}
for i in sorted(positionalIdx):
    sortedPosIdx[i] = positionalIdx[i]

# Write to index.txt
for eachTerm in sortedPosIdx.keys():
    fileToWrite.write(eachTerm + ":" + str(sortedPosIdx[eachTerm][0]) + "\n")
    for eachDoc in sortedPosIdx[eachTerm][1]:
        fileToWrite.write("\t" + str(eachDoc) + ": " + str(sortedPosIdx[eachTerm][1][eachDoc])[1:-1] + "\n")

numOfDocs = int(root[-1][0].text) # Last document ID in the xml file
positionalIdx = sortedPosIdx # Just a rename here

# Build collection matrix
collectionMatrix = np.zeros((numOfDocs, len(positionalIdx)))
for counter, eachTerm in enumerate(positionalIdx):
    for i in range(numOfDocs):
        if i in positionalIdx[eachTerm][1]:
            collectionMatrix[i][counter] = 1
collectionMatrix = collectionMatrix.transpose()

'''SEARCH FUNCTIONS *****************************************************************'''
def booleanSearch(positionalIdx, collectionMatrix, searchTerm):
    terms = searchTerm.split()
    for i in range(len(terms)):
        if terms[i] not in ["AND", "OR", "NOT"]:
            word = stemmer.stem(terms[i].lower())
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


def phraseSearch(positionalIdx, collectionMatrix, searchTerm):
    phraseToSearch = []
    phraseWords = []
    terms = searchTerm.split()
    # Combine phrases
    i = 0
    while i < len(terms):
        if terms[i][0] == "\"":
            phraseToSearch.append(terms[i][1:] + " AND " + terms[i+1][:-1])
            phraseWords.append(stemmer.stem(terms[i][1:].lower()))
            phraseWords.append(stemmer.stem(terms[i+1][:-1].lower()))
            i = i + 2
        else:
            phraseToSearch.append(terms[i])
            i = i + 1
    searchResult = booleanSearch(positionalIdx,collectionMatrix, " ".join(phraseToSearch))
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

def proximitySearch(positionalIdx, collectionMatrix, searchTerm):
    terms = searchTerm.split(",")
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
            phraseToSearch.append(terms[i][j+1:].strip() + " AND " + terms[i+1][:-1].strip())
            phraseWords.append(stemmer.stem(terms[i][j+1:].lower().strip()))
            phraseWords.append(stemmer.stem(terms[i+1][:-1].lower().strip()))
            i = i + 2
        else:
            phraseToSearch.append(terms[i])
            i = i + 1
    searchResult = booleanSearch(positionalIdx,collectionMatrix, " ".join(phraseToSearch))
    # List of document ID and indexes for the corresponding word
    firstWord = positionalIdx[phraseWords[0].strip()][1]
    secondWord = positionalIdx[phraseWords[1].strip()][1]
    correctPhrase = []
    # Only keep documents where the two word phrases are correct
    for eachDoc in searchResult:
        for eachIndex in firstWord[eachDoc]:
            if any(x in secondWord[eachDoc] for x in range(eachIndex-int(dist), eachIndex+int(dist)+1)):
                correctPhrase.append(eachDoc)
                break
    return correctPhrase
'''SEARCH FUNCTIONS ENDS HERE *****************************************************************'''


searchQueryList = open(queryFile).readlines()
booleanResults = "" # Result to be exported to a file
for eachQuery in searchQueryList:
    query = eachQuery.split(' ', 1)[1].strip()
    queryNumber = eachQuery.split(' ', 1)[0].strip()
    # Check what search to perform
    if "#" in query:
        result = proximitySearch(positionalIdx, collectionMatrix, query)
    elif "\"" in query:
        result = phraseSearch(positionalIdx, collectionMatrix, query)
    else:
        result = booleanSearch(positionalIdx, collectionMatrix, query)
    # Output
    for index in result:
        booleanResults = booleanResults + str(queryNumber) + "," + str(index) + "\n"
    fileToWrite = open("results.boolean.txt", "w")
    fileToWrite.write(booleanResults)


'''RANKED IR FUNCTIONS *****************************************************************'''
def tf(term, document, positionalIdx):
    if term not in positionalIdx:
        return 0
    if document not in positionalIdx[stemmer.stem(term.lower())][1]:
        return 0
    return len(positionalIdx[stemmer.stem(term.lower())][1][document])

def df(term, positionalIdx):
    return positionalIdx[stemmer.stem(term.lower())][0]

def weight(term, document, positionalIdx, docCount):
    return (1 + np.log10(tf(term,document,positionalIdx))) * np.log10(docCount/df(term,positionalIdx))

def score(searchTerm, document, positionalIdx, docCount):
    result = 0
    for eachTerm in searchTerm:
        if tf(eachTerm, document, positionalIdx) != 0:
            result = result + weight(eachTerm, document, positionalIdx, docCount)
    return result

def rankedRetrieval(positionalIdx, searchTerm, docCount):
    # Preprocess
    terms = [stemmer.stem(i) for i in re.split('[^\w^\d]', searchTerm.lower()) if i not in stop_words and i != '']
    processedTerms = []
    for term in terms[1:]:
        if term not in stop_words:
            processedTerms.append(stemmer.stem(term.lower()))
    documents = []
    for eachTerm in processedTerms:
        if eachTerm in positionalIdx:
            idxes = positionalIdx[eachTerm][1].keys()
            for eachIdx in idxes:
                documents.append(eachIdx)
    documents = sorted(set(documents))
    scores = []
    for eachDoc in documents:
        scores.append((eachDoc, score(processedTerms,eachDoc,positionalIdx,docCount)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
'''RANKED IR FUNCTIONS ENDS HERE *****************************************************************'''

docCount = len(root) # Number of documents in the collection
searchQueryList = open(rankedFile).readlines()
rankedResults = "" # Result to be exported to a file
for eachQuery in searchQueryList:
    queryNumber = eachQuery.split(' ', 1)[0].strip()
    result = rankedRetrieval(positionalIdx, eachQuery, docCount)
    top150 = min(len(result),150)
    # Output
    for i in range(top150):
        rankedResults = rankedResults + str(queryNumber) + "," + str(result[i][0]) + "," + "{:.4f}".format(round(result[i][1],4)) + "\n"
    fileToWrite = open("results.ranked.txt", "w")
    fileToWrite.write(rankedResults)
