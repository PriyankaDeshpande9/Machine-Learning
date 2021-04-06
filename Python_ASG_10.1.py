import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



def load_data():
    df = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")
    print(df.head())
    df['Wether'] = df['Wether'].replace('Sunny', 1)
    df['Wether'] = df['Wether'].replace('Overcast', 2)
    df['Wether'] = df['Wether'].replace('Rainy', 3)

    df['Temperature'] = df['Temperature'].replace('Hot', 1)
    df['Temperature'] = df['Temperature'].replace('Mild', 2)
    df['Temperature'] = df['Temperature'].replace('Cool', 3)


    df['Play'] = df['Play'].replace('Yes', 1)
    df['Play'] = df['Play'].replace('No', 0)

    #print(df)
    X = df.iloc[:, [1, 2]].values
    Y = df.iloc[:, -1].values

    return X, Y

def calculate_accuracy(predicted_val, actual_test_val):
    accuracy = accuracy_score(predicted_val, actual_test_val)
    return accuracy

def main():
    Line = "=" * 55
    data, target= load_data()
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)
    #print(len(data_train), len(target_train), len(data_test), len(target_test))

    obj = KNeighborsClassifier(n_neighbors=3)
    obj.fit(data, target)
    output = obj.predict(data_test)

    Accuracy_KNN = calculate_accuracy(target_test, output)

    print(Line)
    print("Accuracy with KNN algo is : ",Accuracy_KNN*100)
    print(Line)

if __name__=='__main__':
    main()
