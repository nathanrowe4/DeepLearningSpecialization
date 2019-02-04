import DeepNeuralNetwork as DNN

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    costs = []

    parameters = DNN.initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = DNN.L_model_forward(X, parameters)

        cost = DNN.compute_cost(AL, Y)

        grads = DNN.L_model_backward(AL, Y, caches)

        parameters = DNN.update_parameters(parameters, grads, learning_rate)
    
    return parameters