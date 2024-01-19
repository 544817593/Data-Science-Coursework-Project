import csv
import math
import random
import string
import sklearn
import scipy
import numpy as np

from scipy.stats import ttest_ind
from nltk.stem.porter import *
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from scipy import sparse
from sklearn import svm
from sklearn.linear_model import SGDClassifier



# read csv files, putting them into dictionaries where keys are system number and query number respectively
sysRel = {}
qRel = {}
with open('system_results.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row['system_number']) in sysRel.keys():
            sysRel[int(row['system_number'])].append((int(row['query_number']),int(row['doc_number']),int(row['rank_of_doc']),float(row['score'])))
        else:
            sysRel[int(row['system_number'])] = [(int(row['query_number']),int(row['doc_number']),int(row['rank_of_doc']),float(row['score']))]

with open('qrels.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row['query_id']) in qRel.keys():
            qRel[int(row['query_id'])].append((int(row['doc_id']),int(row['relevance'])))
        else:
            qRel[int(row['query_id'])] = [(int(row['doc_id']),int(row['relevance']))]


def EVAL(sysRel, qRel):
    irEval = [["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"]]  # final output list
    for system in sysRel.keys():
        meanP10 = 0.0
        meanR50 = 0.0
        meanRP = 0.0
        meanAP = 0.0
        meanNDCG10 = 0.0
        meanNDCG20 = 0.0
        for query in range(1,len(qRel)+1):
            currentSys = sysRel[system]
            currentQ = qRel[query]

            relevantRes = set([x[0] for x in currentQ])
            # precision at 10
            top10Res = set([x[1] for x in currentSys if x[0] == query][:10])
            precision10 = (len((relevantRes).intersection(top10Res)) / 10)
            meanP10 += float(precision10)
            precision10 = "%.3f" % round(precision10, 3)

            # recall at 50
            top50Res = set([x[1] for x in currentSys if x[0] == query][:50])
            recall50 = len(relevantRes.intersection(top50Res))/len(relevantRes)
            meanR50 += float(recall50)
            recall50 = "%.3f" % round(recall50, 3)

            # r-precision
            topRRes = set([x[1] for x in currentSys if x[0] == query][:len(relevantRes)])
            precisionR = (len((relevantRes).intersection(topRRes)) / len(relevantRes))
            meanRP += float(precisionR)
            precisionR = "%.3f" % round(precisionR, 3)

            # average precision
            allRes = [x[1] for x in currentSys if x[0] == query]
            averagePrecision = 0
            relevanceRank = 1
            for i in range(len(allRes)):
                if allRes[i] in relevantRes:
                    averagePrecision += relevanceRank / (i+1)
                    relevanceRank += 1
            averagePrecision = averagePrecision / len(relevantRes)
            meanAP += float(averagePrecision)
            averagePrecision = "%.3f" % round(averagePrecision, 3)

            # nDCG at 10
            top10Res = [x[1] for x in currentSys if x[0] == query][:10]
            relevanceVal = [x[1] for x in currentQ]
            dcg = 0 # discounted cumulative gain
            ig = [x[1] for x in currentQ] # ideal gains
            for i in range(len(top10Res)):
                if top10Res[i] in relevantRes:
                    if i <= 1:
                        dcg += int("".join([str(x[1]) for x in currentQ if x[0] == top10Res[i]]))
                    else:
                        dcg += int("".join([str(x[1]) for x in currentQ if x[0] == top10Res[i]])) * (1 / math.log2(i + 1))
            ig.sort(reverse=True)
            idcg = 0 # ideal discounted cumulative gain
            for i in range(len(ig)):
                if (i == 10):
                    break
                if i <= 1:
                    idcg += ig[i]
                else:
                    idcg += ig[i] * (1/math.log2(i+1))
            ndcg10 = 0
            if (idcg != 0):
                ndcg10 = (dcg / idcg)
                meanNDCG10 += float(ndcg10)
                ndcg10 = "%.3f" % round(ndcg10, 3)

            # nDCG at 20
            top20Res = [x[1] for x in currentSys if x[0] == query][:20]
            relevanceVal = [x[1] for x in currentQ]
            dcg = 0 # discounted cumulative gain
            ig = [x[1] for x in currentQ] # ideal gains
            for i in range(len(top20Res)):
                if top20Res[i] in relevantRes:
                    if i <= 1:
                        dcg += int("".join([str(x[1]) for x in currentQ if x[0] == top20Res[i]]))
                    else:
                        dcg += int("".join([str(x[1]) for x in currentQ if x[0] == top20Res[i]])) * (1 / math.log2(i + 1))
            ig.sort(reverse=True)
            idcg = 0 # ideal discounted cumulative gain
            for i in range(len(ig)):
                if (i == 20):
                    break
                if i <= 1:
                    idcg += ig[i]
                else:
                    idcg += ig[i] * (1/math.log2(i+1))
            ndcg20 = 0
            if (idcg != 0):
                ndcg20 = (dcg / idcg)
                meanNDCG20 += float(ndcg20)
                ndcg20 = "%.3f" % round(ndcg20, 3)

            irEval.append([system, query, precision10, recall50, precisionR, averagePrecision, ndcg10, ndcg20])
        irEval.append([system, "mean", "%.3f" % round((meanP10 / 10), 3), "%.3f" % round((meanR50 / 10), 3), "%.3f" % round((meanRP / 10), 3), "%.3f" % round((meanAP / 10), 3), "%.3f" % round((meanNDCG10 / 10), 3), "%.3f" % round((meanNDCG20 / 10), 3)])

    with open("ir_eval.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(irEval)

'''
    # IR Evaluation, 2 tailed t-test
    sys1 = [float(x[2]) for x in irEval if x[0] == 1 and x[1] != "mean"]
    print(sys1)
    sys2 = [float(x[2]) for x in irEval if x[0] == 2 and x[1] != "mean"]
    print(ttest_ind(sys1, sys2))
'''

EVAL(sysRel, qRel)

'''
Task 2 begins here
'''

stemmer = PorterStemmer()
ot = []
nt = []
quran = []

# English stop words
stop_words = open("EnglishST.txt").read().split('\n')

# read files
with open("train_and_dev.tsv", newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter="\t")
    for row in reader:
        if row[0] == "OT":
            ot.append(row[1])
        if row[0] == "NT":
            nt.append(row[1])
        if row[0] == "Quran":
            quran.append(row[1])

# preprocessing
otTokens = []
ntTokens = []
quranTokens = []

for sentence in ot:
    otTokens.append([stemmer.stem(i) for i in re.split('[^\w^\d]', sentence.lower()) if i not in stop_words and i != ''])
for sentence in nt:
    ntTokens.append([stemmer.stem(i) for i in re.split('[^\w^\d]', sentence.lower()) if i not in stop_words and i != ''])
for sentence in quran:
    quranTokens.append([stemmer.stem(i) for i in re.split('[^\w^\d]', sentence.lower()) if i not in stop_words and i != ''])

# calculate term frequency
otTokensFreq = {}
ntTokensFreq = {}
quranTokensFreq = {}

for sentence in otTokens:
    doc = [] # count each word once in document
    for word in sentence:
        if word not in otTokensFreq.keys():
            otTokensFreq[word] = 1
            doc.append(word)
            continue
        if word not in doc:
            otTokensFreq[word] += 1
            doc.append(word)

for sentence in ntTokens:
    doc = [] # count each word once in document
    for word in sentence:
        if word not in ntTokensFreq.keys():
            ntTokensFreq[word] = 1
            doc.append(word)
            continue
        if word not in doc:
            ntTokensFreq[word] += 1
            doc.append(word)

for sentence in quranTokens:
    doc = [] # count each word once in document
    for word in sentence:
        if word not in quranTokensFreq.keys():
            quranTokensFreq[word] = 1
            doc.append(word)
            continue
        if word not in doc:
            quranTokensFreq[word] += 1
            doc.append(word)

# calculate MI for a corpus
# tfCor1 is the current corpus, and nCor1 is the number of words in corpus one.
def calculateMI(tfCor1, tfCor2, tfCor3, nCor1, nCor2, nCor3):
    MI = {}
    uniqTerms = set(tfCor1.keys()).union(set(tfCor2.keys()), set(tfCor3.keys()))
    # calculate MI for each term
    for term in uniqTerms:
        N11 = 0
        N10 = 0
        N01 = 0
        N00 = 0
        if term in tfCor1.keys():
            N11 = tfCor1[term]
        if term in tfCor2.keys():
            N10 += tfCor2[term]
        if term in tfCor3.keys():
            N10 += tfCor3[term]
        N01 = nCor1 - N11
        N00 = nCor1 + nCor2 + nCor3 - N11 - N10 - N01
        N = N11+N10+N01+N00
        # four parts of the equation from lecture slides
        if (N*N11/((N10+N11)*(N11+N01))) == 0:
            part1 = 0
        else:
            part1 = N11/N * math.log2(N*N11/((N10+N11)*(N11+N01)))

        if N*N01/((N00+N01)*(N01+N11)) == 0:
            part2 = 0
        else:
            part2 = N01/N * math.log2(N*N01/((N00+N01)*(N01+N11)))

        if N*N10/((N10+N11)*(N10+N00)) == 0:
            part3 = 0
        else:
            part3 = N10/N * math.log2(N*N10/((N10+N11)*(N10+N00)))

        if N*N00/((N00+N01)*(N10+N00)) == 0:
            part4 = 0
        else:
            part4 = N00/N * math.log2(N*N00/((N00+N01)*(N10+N00)))
        tokenMI = part1+part2+part3+part4
        MI[term] = tokenMI
    return MI

# calculate MI's
otMI = sorted(calculateMI(otTokensFreq, ntTokensFreq, quranTokensFreq, len(otTokens), len(ntTokens), len(quranTokens)).items(), key=lambda item:item[1], reverse=True)
ntMI = sorted(calculateMI(ntTokensFreq, otTokensFreq, quranTokensFreq, len(ntTokens), len(otTokens), len(quranTokens)).items(), key=lambda item:item[1], reverse=True)
quranMI = sorted(calculateMI(quranTokensFreq, ntTokensFreq, otTokensFreq, len(quranTokens), len(ntTokens), len(otTokens)).items(), key=lambda item:item[1], reverse=True)

# calculate Chi2 for a corpus
# tfCor1 is the current corpus, and nCor1 is the number of words in corpus one.
def calculateChi2(tfCor1, tfCor2, tfCor3, nCor1, nCor2, nCor3):
    Chi2 = {}
    uniqTerms = set(tfCor1.keys()).union(set(tfCor2.keys()), set(tfCor3.keys()))
    # calculate Chi2 for each term
    for term in uniqTerms:
        N11 = 0
        N10 = 0
        N01 = 0
        N00 = 0
        if term in tfCor1.keys():
            N11 = tfCor1[term]
        if term in tfCor2.keys():
            N10 += tfCor2[term]
        if term in tfCor3.keys():
            N10 += tfCor3[term]
        N01 = nCor1 - N11
        N00 = nCor1 + nCor2 + nCor3 - N11 - N10 - N01
        N = N11 + N10 + N01 + N00
        # two parts of the equation from lecture slides
        top = N*(N11*N00-N10*N01)**2
        bot = (N11+N01)*(N11+N10)*(N10+N00)*(N01+N00)
        termChi2 = top/bot
        Chi2[term] = termChi2
    return Chi2

# calculate Chi2's
otChi2 = sorted(calculateChi2(otTokensFreq, ntTokensFreq, quranTokensFreq, len(otTokens), len(ntTokens), len(quranTokens)).items(), key=lambda item:item[1], reverse=True)
ntChi2 = sorted(calculateChi2(ntTokensFreq, otTokensFreq, quranTokensFreq, len(ntTokens), len(otTokens), len(quranTokens)).items(), key=lambda item:item[1], reverse=True)
quranChi2 = sorted(calculateChi2(quranTokensFreq, ntTokensFreq, otTokensFreq, len(quranTokens), len(ntTokens), len(otTokens)).items(), key=lambda item:item[1], reverse=True)


# Generate a ranked list of the results
print("Generating ranked list for OT:\n")
print("token\tMI score\ttoken\tChi2 score\n")
for i in range(10):
    print(otMI[i][0] + "\t" + str(round(otMI[i][1],5)) + "\t\t" + otChi2[i][0] + "\t" + str(round(otChi2[i][1],5)))
    print("\n")

print("Generating ranked list for NT:\n")
print("token\tMI score\ttoken\tChi2 score\n")
for i in range(10):
    print(ntMI[i][0] + "\t" + str(round(ntMI[i][1],5)) + "\t\t" + ntChi2[i][0] + "\t" + str(round(ntChi2[i][1],5)))
    print("\n")

print("Generating ranked list for Quran:\n")
print("token\tMI score\ttoken\tChi2 score\n")
for i in range(10):
    print(quranMI[i][0] + "\t" + str(round(quranMI[i][1],5)) + "\t\t" + quranChi2[i][0] + "\t" + str(round(quranChi2[i][1],5)))
    print("\n")

# LDA
allTokens = otTokens + ntTokens + quranTokens
common_dictionary = Dictionary(allTokens)
common_corpus = [common_dictionary.doc2bow(text) for text in allTokens]
lda = LdaModel(common_corpus, num_topics=20,id2word=common_dictionary)
docScores = lda.get_document_topics(bow=common_corpus)

# computes the sorted average score for each topic
def computeAverageScore(docScores):
    averageScore = {}
    for doc in docScores:
        for topic in doc: # topics are tuples of (topic, score)
            if topic[0] in averageScore:
                averageScore[topic[0]] += topic[1]
            else:
                averageScore[topic[0]] = topic[1]

    for topic in averageScore.keys():
        averageScore[topic] /= len(docScores)
    return sorted(averageScore.items(), key=lambda x: x[1], reverse=True)


docScores_ot = docScores[0:len(otTokens)]
docScores_nt = docScores[len(otTokens):len(otTokens) + len(ntTokens)]
docScores_quran = docScores[len(otTokens) + len(ntTokens):]

# find the top topic for each corpus
topOT = computeAverageScore(docScores_ot)[0]
topNT = computeAverageScore(docScores_nt)[0]
topQURAN = computeAverageScore(docScores_quran)[0]

# find the top 10 tokens and their probability scores
def computeTopTokens(topTopicTuple, lda):
    topic = str(topTopicTuple[0])
    probability = lda.print_topic(topTopicTuple[0], 10)
    return "topic " + topic + ":\n" + probability


print("Generating top 10 tokens and their probability scores:")
print("For OT:")
print(computeTopTokens(topOT,lda) + "\n")
print("For NT:")
print(computeTopTokens(topNT,lda) + "\n")
print("For QURAN:")
print(computeTopTokens(topQURAN,lda) + "\n")

'''
Task 3 begins here
'''

trainingData = open("train_and_dev.tsv").read()

# preprocess data
def preProcess(data):
    removeChar = re.compile(f'[{string.punctuation}]')
    documents_train = [] # list of sentences in list of strings
    categories_train = [] # list of classes
    vocab_train = set([]) # unique words

    documents_dev = [] # list of sentences in list of strings
    categories_dev = [] # list of classes
    vocab_dev = set([]) # unique words

    lines = data.split('\n')
    random.shuffle(lines)
    N_train = int(np.ceil(len(lines) * 0.9))

    # training set
    for line in lines[0:N_train]:
        line = line.strip()
        if line:
            category, text = line.split('\t')
            words = removeChar.sub('',text).lower().split()
            for word in words:
                vocab_train.add(word)
            documents_train.append(words)
            categories_train.append(category)

    # develop set
    for line in lines[N_train:]:
        line = line.strip()
        if line:
            category, text = line.split('\t')
            words = removeChar.sub('',text).lower().split()
            for word in words:
                vocab_dev.add(word)
            documents_dev.append(words)
            categories_dev.append(category)
    return documents_train, categories_train, vocab_train, documents_dev, categories_dev, vocab_dev

# preprocess test data
def preProcessTest(data):
    removeChar = re.compile(f'[{string.punctuation}]')
    documents_test = [] # list of sentences in list of strings
    categories_test = [] # list of classes
    vocab_test = set([]) # unique words

    lines = data.split('\n')
    # training set
    for line in lines:
        line = line.strip()
        if line:
            category, text = line.split('\t')
            words = removeChar.sub('',text).lower().split()
            for word in words:
                vocab_test.add(word)
            documents_test.append(words)
            categories_test.append(category)
    return documents_test, categories_test, vocab_test

# map unique terms to an ID
def uniqueTerms(vocab):
    word2id = {}
    for wordId, word in enumerate(vocab):
        word2id[word] = wordId
    return word2id

# map unique classes to an ID
def uniqueClasses(categories):
    cat2Id = {}
    for catId, cat in enumerate(set(categories)):
        cat2Id[cat] = catId
    return cat2Id

# convert data to BOW format
def convertToBOWMatrix(preprocessedData, word2id):
    matrixSize = (len(preprocessedData),len(word2id)+1)
    oovIndex = len(word2id) # out of vocabulary index
    # [docId, tokenId] matrix
    X = scipy.sparse.dok_matrix(matrixSize)
    # iterate through each document(line) in the dataset
    for docId, doc in enumerate(preprocessedData):
        # count word, or if word not found in word2id, increment oov count
        for word in doc:
            X[docId, word2id.get(word,oovIndex)] += 1
    return X


documents_train, categories_train, vocab_train, documents_dev, categories_dev, vocab_dev = preProcess(trainingData)
word2Id = uniqueTerms(vocab_train)
cat2Id = uniqueClasses(categories_train)
# data dict in the format of (docId, wordId) -> word count
X_train = convertToBOWMatrix(documents_train, word2Id)
X_dev = convertToBOWMatrix(documents_dev, word2Id)
# data labels
y_train = [cat2Id[cat] for cat in categories_train]
y_dev = [cat2Id[cat] for cat in categories_dev]

# train an SVM model
baselineModel = sklearn.svm.SVC(C=1000)
baselineModel.fit(X_train, y_train)

y_train_pred = baselineModel.predict(X_train)
y_dev_pred = baselineModel.predict(X_dev)

# compute precision, recall and f1 scores
def calculateScores(category, y_pred, y_actual):
    matrix = np.zeros((2,2))
    for i in range(len(y_actual)):
        if y_actual[i] == category and y_pred[i] == category:
            matrix[0][0] += 1
        if y_actual[i] != category and y_pred[i] != category:
            matrix[1][1] += 1
        if y_actual[i] == category and y_pred[i] != category:
            matrix[1][0] += 1
        if y_actual[i] != category and y_pred[i] == category:
            matrix[0][1] += 1

    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

p_quran_train, r_quran_train, f1_quran_train = calculateScores(0, y_train_pred, y_train)
p_ot_train, r_ot_train, f1_ot_train = calculateScores(1, y_train_pred, y_train)
p_nt_train, r_nt_train, f1_nt_train = calculateScores(2, y_train_pred, y_train)
p_macro_train = (p_quran_train + p_ot_train + p_nt_train)/3
r_macro_train = (r_quran_train + r_ot_train + r_nt_train)/3
f1_macro_train = 2 * p_macro_train * r_macro_train / (p_macro_train + r_macro_train)

p_quran_dev, r_quran_dev, f1_quran_dev = calculateScores(0, y_dev_pred, y_dev)
p_ot_dev, r_ot_dev, f1_ot_dev = calculateScores(1, y_dev_pred, y_dev)
p_nt_dev, r_nt_dev, f1_nt_dev = calculateScores(2, y_dev_pred, y_dev)
p_macro_dev = (p_quran_dev + p_ot_dev + p_nt_dev)/3
r_macro_dev = (r_quran_dev + r_ot_dev + r_nt_dev)/3
f1_macro_dev = 2 * p_macro_dev * r_macro_dev / (p_macro_dev + r_macro_dev)

# test set
testData = open("test.tsv").read()
documents_test, categories_test, vocab_test = preProcessTest(testData)
# data dict in the format of (docId, wordId) -> word count
X_test = convertToBOWMatrix(documents_test, word2Id)
# data labels
y_test = [cat2Id[cat] for cat in categories_test]

y_test_pred = baselineModel.predict(X_test)
p_quran_test, r_quran_test, f1_quran_test = calculateScores(0, y_test_pred, y_test)
p_ot_test, r_ot_test, f1_ot_test = calculateScores(1, y_test_pred, y_test)
p_nt_test, r_nt_test, f1_nt_test = calculateScores(2, y_test_pred, y_test)
p_macro_test = (p_quran_test + p_ot_test + p_nt_test)/3
r_macro_test = (r_quran_test + r_ot_test + r_nt_test)/3
f1_macro_test = 2 * p_macro_test * r_macro_test / (p_macro_test + r_macro_test)

baselineResults = [['system','split','p-quran','r-quran','f-quran','p-ot','r-ot','f-ot','p-nt','r-nt','f-nt','p-macro','r-macro','f-macro']]
baselineResults.append(['baseline','train',str(round(p_quran_train,3)), str(round(r_quran_train,3)), str(round(f1_quran_train,3)), str(round(p_ot_train,3)), str(round(r_ot_train,3)), str(round(f1_ot_train,3)), str(round(p_nt_train,3)), str(round(r_nt_train,3)), str(round(f1_nt_train,3)), str(round(p_macro_train,3)), str(round(r_macro_train,3)), str(round(f1_macro_train,3))])
baselineResults.append(['baseline','dev',str(round(p_quran_dev,3)), str(round(r_quran_dev,3)), str(round(f1_quran_dev,3)), str(round(p_ot_dev,3)), str(round(r_ot_dev,3)), str(round(f1_ot_dev,3)), str(round(p_nt_dev,3)), str(round(r_nt_dev,3)), str(round(f1_nt_dev,3)), str(round(p_macro_dev,3)), str(round(r_macro_dev,3)), str(round(f1_macro_dev,3))])
baselineResults.append(['baseline','test',str(round(p_quran_test,3)), str(round(r_quran_test,3)), str(round(f1_quran_test,3)), str(round(p_ot_test,3)), str(round(r_ot_test,3)), str(round(f1_ot_test,3)), str(round(p_nt_test,3)), str(round(r_nt_test,3)), str(round(f1_nt_test,3)), str(round(p_macro_test,3)), str(round(r_macro_test,3)), str(round(f1_macro_test,3))])

with open("classification.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(baselineResults)

# improve the system
# changing the SVM parameters
improveModel = SGDClassifier(loss='modified_huber',alpha=0.0002,tol=1e-7,max_iter=5000)
improveModel.fit(X_train, y_train)

y_train_pred = improveModel.predict(X_train)
y_dev_pred = improveModel.predict(X_dev)

p_quran_train, r_quran_train, f1_quran_train = calculateScores(0, y_train_pred, y_train)
p_ot_train, r_ot_train, f1_ot_train = calculateScores(1, y_train_pred, y_train)
p_nt_train, r_nt_train, f1_nt_train = calculateScores(2, y_train_pred, y_train)
p_macro_train = (p_quran_train + p_ot_train + p_nt_train)/3
r_macro_train = (r_quran_train + r_ot_train + r_nt_train)/3
f1_macro_train = 2 * p_macro_train * r_macro_train / (p_macro_train + r_macro_train)

p_quran_dev, r_quran_dev, f1_quran_dev = calculateScores(0, y_dev_pred, y_dev)
p_ot_dev, r_ot_dev, f1_ot_dev = calculateScores(1, y_dev_pred, y_dev)
p_nt_dev, r_nt_dev, f1_nt_dev = calculateScores(2, y_dev_pred, y_dev)
p_macro_dev = (p_quran_dev + p_ot_dev + p_nt_dev)/3
r_macro_dev = (r_quran_dev + r_ot_dev + r_nt_dev)/3
f1_macro_dev = 2 * p_macro_dev * r_macro_dev / (p_macro_dev + r_macro_dev)

y_test_pred = improveModel.predict(X_test)
p_quran_test, r_quran_test, f1_quran_test = calculateScores(0, y_test_pred, y_test)
p_ot_test, r_ot_test, f1_ot_test = calculateScores(1, y_test_pred, y_test)
p_nt_test, r_nt_test, f1_nt_test = calculateScores(2, y_test_pred, y_test)
p_macro_test = (p_quran_test + p_ot_test + p_nt_test)/3
r_macro_test = (r_quran_test + r_ot_test + r_nt_test)/3
f1_macro_test = 2 * p_macro_test * r_macro_test / (p_macro_test + r_macro_test)

baselineResults.append(['improved','train',str(round(p_quran_train,3)), str(round(r_quran_train,3)), str(round(f1_quran_train,3)), str(round(p_ot_train,3)), str(round(r_ot_train,3)), str(round(f1_ot_train,3)), str(round(p_nt_train,3)), str(round(r_nt_train,3)), str(round(f1_nt_train,3)), str(round(p_macro_train,3)), str(round(r_macro_train,3)), str(round(f1_macro_train,3))])
baselineResults.append(['improved','dev',str(round(p_quran_dev,3)), str(round(r_quran_dev,3)), str(round(f1_quran_dev,3)), str(round(p_ot_dev,3)), str(round(r_ot_dev,3)), str(round(f1_ot_dev,3)), str(round(p_nt_dev,3)), str(round(r_nt_dev,3)), str(round(f1_nt_dev,3)), str(round(p_macro_dev,3)), str(round(r_macro_dev,3)), str(round(f1_macro_dev,3))])
baselineResults.append(['improved','test',str(round(p_quran_test,3)), str(round(r_quran_test,3)), str(round(f1_quran_test,3)), str(round(p_ot_test,3)), str(round(r_ot_test,3)), str(round(f1_ot_test,3)), str(round(p_nt_test,3)), str(round(r_nt_test,3)), str(round(f1_nt_test,3)), str(round(p_macro_test,3)), str(round(r_macro_test,3)), str(round(f1_macro_test,3))])

with open("classification.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(baselineResults)