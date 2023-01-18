from regression import linearRegression
import pandas as pd
import matplotlib.pyplot as plt

def main():

    data = pd.read_csv("bp.csv")
    print(data.head())

    # first column corresponds to X Values (year)
    # second column corresponds to Y values (value)
    X = data['age'].values
    Y = data['bloodPressure'].values

    print(X)
    print(Y)

    # creating object model
    # through trial and error, optimal learning rate
    # is around 0.0001
    model = linearRegression(iterations=1000, learningRate=0.0001)
    model.fit(X, Y)


    # printing parameter values of line of best fit
    # after training
    print(f"weight value: {model.weight}")
    print(f"bias value: {model.bias}")

    plt.scatter(X,Y, color = 'red')
    #plt.plot(list(range(1, 300)), [model.weight * x + model.bias for x in range(1, 300)], color="black")
    plt.plot(list(range(16, 80)), [model.weight * x + model.bias for x in range(16, 80)], color="black")
    plt.show()

    print(f"-----Model Prediction-----")
    print(f"Predicted blood pressure of person age 56: {model.predict(56)}")


if __name__ == "__main__":
    main()
