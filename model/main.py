import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle as pk


def clean_data():
    df = pd.read_csv('Data/data.csv')
    df = df.drop(['id'], axis=1)
    df = df.drop(['Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def create_model(data):
    scaler = StandardScaler()
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    X = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy of model : ", accuracy_score(y_test, y_pred))
    print("Classification Report :\n", classification_report(y_test, y_pred))

    return model, scaler


def main():
    data = clean_data()
    model, scaler = create_model(data)

    with open('model,pkl', 'wb') as f:
        pk.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pk.dump(scaler, f)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
