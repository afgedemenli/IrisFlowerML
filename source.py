import pandas
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

irisData = pandas.read_csv("./iris.data")

# Get the feature data
features = irisData[["feature1", "feature2", "feature3", "feature4"]]

# Get the target data
targetVariables = irisData.Class

#  Shuffling and dividing the data
featureTrain, featureTestAndValidation, targetTrain, targetTestAndValidation = train_test_split(features,
                                                                                                targetVariables,
                                                                                                test_size=.8)
featureTest = featureTestAndValidation[60:]
featureValidation = featureTestAndValidation[:60]
targetTest = targetTestAndValidation[60:]
targetValidation = targetTestAndValidation[:60]
# print(len(featureTestAndValidation))
# for i in range(len(featureTestAndValidation)):
#     if i < len(featureTestAndValidation) / 2:
#         featureTest.append(featureTestAndValidation[i])
#         targetTest.append(targetTestAndValidation[i])
#     else:
#         featureValidation.append(featureTestAndValidation[i])
#         targetValidation.append(targetTestAndValidation[i])


#  Training and finding the optimal depth for both methods
optimalDepths = {}
for method in ["gini", "entropy"]:
    accuracy = 0
    previousAccuracy = 0
    depth = 0
    plotx = []
    ploty = []
    while accuracy > previousAccuracy or depth == 0:
        depth += 1
        decisionTree = DecisionTreeClassifier(max_depth=depth, criterion=method).fit(featureTrain, targetTrain)
        predictions = decisionTree.predict(featureValidation)
        previousAccuracy = accuracy
        accuracy = accuracy_score(targetValidation, predictions)
        plotx.append(depth)
        ploty.append(1 - accuracy)  # plot the loss

    optimalDepths[method] = depth - 1
    plotx = plotx[:-1]
    ploty = ploty[:-1]
    # pyplot.plot(plotx, ploty)
    ax = pyplot.figure().gca()
    ax.plot(plotx, ploty, marker='s')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pyplot.show()

#  Gini Test
giniDT = DecisionTreeClassifier(max_depth=optimalDepths["gini"], criterion="gini").fit(featureTrain, targetTrain)
giniPredictions = giniDT.predict(featureTest)
giniAccuracy = accuracy_score(targetTest, giniPredictions)
giniLoss = 1 - giniAccuracy

#  Information Gain Test
informationGainDT = DecisionTreeClassifier(max_depth=optimalDepths["entropy"], criterion="entropy").fit(featureTrain,
                                                                                                        targetTrain)
informationGainPredictions = informationGainDT.predict(featureTest)
informationGainAccuracy = accuracy_score(targetTest, informationGainPredictions)
informationGainLoss = 1 - informationGainAccuracy

print("Loss in Information Gain: ")
print("\tAmount: " + str(int(round(informationGainLoss * 60))) + "/60\n\tRatio: " + str(round(informationGainLoss, 6)))
print("------------------------\nLoss in Gini Impurity: ")
print("\tAmount: " + str(int(round(giniLoss * 60))) + "/60\n\tRatio: " + str(round(giniLoss, 6)))
