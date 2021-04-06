from MyModules import *


def CalculateDistance(x, y):
    return distance.euclidean(x, y)

class MyKNN:
    def fit(self, TrainingData, TrainingTarget):
        #print("---------------Inside fit--------------")
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget
        #print("Data Training Done.")

    def predict(self, TestData):
        #print("---------------inside predict----------")
        predictions = []
        for row in TestData:
            label = self.shortest(row)
            predictions.append(label)
        #print("Data Testing done.")
        return predictions

    def shortest(self, row):   #Here k=1
        MinIndex = 0
        MinDistance = CalculateDistance(row, self.TrainingData[0])

        for i in range(1, len(self.TrainingData)):
            Distance = CalculateDistance(row, self.TrainingData[i])
            if Distance < MinDistance:
                MinDistance = Distance
                MinIndex = i

        return self.TrainingTarget[MinIndex]


def My_KNN():
    Line = "="*55
    #print("Inside user defined KNN implementation.")
    iris = load_iris()      # 150/5 (4 Features, 1 Label)
    data = iris.data        # 150/4
    target = iris.target    # 150/1

    print(Line)
    print("Actual Dataset : ")
    print(Line)
    for i in range(len(iris.target)):
        print(f"ID : {i}, Data : {iris.data[i]}, Features : {iris.target[i]}")

    # Data split randomly
    # 75/4       75/4        75/4        75/4
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)
    #print("Dataset loaded successfully.")
    print(Line)
    print("Training Dataset : ")
    print(Line)
    for i in range(len(data_train)):
        print(f"ID : {i}, Data : {data_train[i]}, Features : {target_train[i]}")

    print(Line)
    print("Testing Dataset : ")
    print(Line)
    for i in range(len(data_train)):
        print(f"ID : {i}, Data : {data_test[i]}, Features : {target_test[i]}")

    print(Line)
    obj = MyKNN()
    obj.fit(data_train, target_train)
    ret = obj.predict(data_test)

    for i in range(len(data_train)):
        print(f"ID : {i}, Expectation : {target_test[i]}, Prediction : {ret[i]}")
    icnt = 0
    for i in range(len(data_train)):
        if target_test[i]!=ret[i]:
            icnt +=1
    my_accuracy = (len(data_train)-icnt)/len(data_train)*100
    print(Line)
    print("Number of wrong answers by ML model : ",icnt)
    print("My calculated Accuracy is : ",my_accuracy)
    print(Line)
    Accuracy = accuracy_score(target_test, ret)
    #print("Result is : ")
    return Accuracy


def main():
    result = My_KNN()
    print("Accuracy of KNN is : ", result*100)


if __name__ == '__main__':
    main()