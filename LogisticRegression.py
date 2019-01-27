import numpy as np;

#sigmoid used as activation function
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def get_prediction(w, b, X):
	return sigmoid(np.dot(w.T,X) + b)

def initialize(dim):
    #initialize hyperparameters
    w = np.zeros((dim,1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    # m is number of training sets
    m = X.shape[1]
    A = get_prediction(w, b, X)
    cost = np.sum(-(Y*np.log(A) + (1 - Y)*np.log(1 - A))) / m
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    cost = np.squeeze(cost)
    gradients = {"dw": dw,
                 "db": db}
    
    return gradients, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    
    for iteration in range(num_iterations):
        gradients, cost = propagate(w,b,X,Y)
        dw = gradients["dw"]
        db = gradients["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        #only store every 100 costs
        if iteration % 100 == 0:
            costs.append(cost)
    
    parameters = {"w": w,
                  "b": b}
    
    gradients = {"dw": dw,
                 "db": db}
    
    return parameters, gradients, costs

def predict(w, b, X):
    # m is number of training sets
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    A = get_prediction(w, b, X)
    
    Y_prediction = np.array([np.where(A[0] <= 0.5, 0, 1)])
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    #initialize
    w, b = initialize(X_train.shape[0])

    # gradient descent
    parameters, gradients, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    # retrieve hyperparameters for model
    w = parameters["w"]
    b = parameters["b"]
    
    # predict test/train set
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    model_data = {"costs": costs,
         "Y_train": Y_train,
         "Y_test": Y_test,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return model_data

def show_results(model_data):
    train_accuracy = 100 - np.mean(np.abs(model_data["Y_prediction_train"] - model_data["Y_train"])) * 100
    test_accuracy = 100 - np.mean(np.abs(model_data["Y_prediction_test"] - model_data["Y_test"])) * 100
    print("train accuracy: " + str(train_accuracy) + " %")
    print("test accuracy: " + str(test_accuracy) + " %")