# imports
import sys
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix

question = sys.argv[1]

def faruk_ulutas_21803717_hw1(question):
    # Question 1 Functions
    def rgb_to_grayscale(images):
        return 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]

    def normalize_data(data):
        return (data - data.min()) / (data.max() - data.min()) * 0.8 + 0.1

    def normalize_to_zero_one_range(images):
        return (images - images.min()) / (images.max() - images.min())

    def display_images(images, title, is_grayscale=True, n=200):
        plt.figure(figsize=(20, 20))
        plt.suptitle(title)
        for i in range(n):
            plt.subplot(10, 20, i + 1)
            plt.axis('off')
            if is_grayscale:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.show()

    def initialize_weights_and_biases(Lin, Lhid):
        wo = np.sqrt(6 / (Lin + Lhid))
        W1 = np.random.uniform(-wo, wo, (Lhid, Lin))
        W2 = np.random.uniform(-wo, wo, (Lin, Lhid))
        b1 = np.zeros(Lhid)
        b2 = np.zeros(Lin)
        return W1, W2, b1, b2

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def aeCost(We, data, params):
        Lhid = params["Lhid"]
        Lin = params["Lin"]
        beta = params["beta"]
        rho = params["rho"]
        lambda_ = params["lambda"]

        W1 = We[:Lin*Lhid].reshape((Lhid, Lin))
        W2 = We[Lin*Lhid:2*Lin*Lhid].reshape((Lin, Lhid))
        b1 = We[2*Lin*Lhid:2*Lin*Lhid+Lhid]
        b2 = We[2*Lin*Lhid+Lhid:]

        a2 = sigmoid(np.dot(W1, data) + b1[:, None])
        a3 = sigmoid(np.dot(W2, a2) + b2[:, None])

        rho_hat = np.mean(a2, axis=1)
        kl_divergence = np.sum(rho * np.log(rho/rho_hat) + (1 - rho) * np.log((1 - rho)/(1 - rho_hat)))
        cost = 0.5 * np.sum((a3 - data)**2) / data.shape[1] + \
               0.5 * lambda_ * (np.sum(W1**2) + np.sum(W2**2)) + \
               beta * kl_divergence

        delta3 = (a3 - data) * a3 * (1 - a3)
        rho_hat = rho_hat.reshape(-1, 1)
        delta2 = (np.dot(W2.T, delta3) + beta * (- rho/rho_hat + (1 - rho)/(1 - rho_hat))) * a2 * (1 - a2)
        W1_grad = np.dot(delta2, data.T) / data.shape[1] + lambda_ * W1
        W2_grad = np.dot(delta3, a2.T) / data.shape[1] + lambda_ * W2
        b1_grad = np.sum(delta2, axis=1) / data.shape[1]
        b2_grad = np.sum(delta3, axis=1) / data.shape[1]

        grads = np.concatenate((W1_grad.flatten(), W2_grad.flatten(), b1_grad.flatten(), b2_grad.flatten()))
        return cost, grads

    def gradient_descent(data, params, max_iter=200, learning_rate=0.1):
        Lin = params["Lin"]
        Lhid = params["Lhid"]
        W1, W2, b1, b2 = initialize_weights_and_biases(Lin, Lhid)
        We = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
        
        for i in range(max_iter):
            cost, grads = aeCost(We, data, params)
            We -= learning_rate * grads
            if i % 10 == 0:
                print(f'Iteration {i}/{max_iter}, cost: {cost}')
        return We

    def display_features(W, title):
        n = W.shape[0]
        ncols = math.isqrt(n) if math.isqrt(n)**2 == n else math.isqrt(n) + 1
        nrows = n // ncols
        if n % ncols:
            nrows += 1
        plt.figure(figsize=(20, 20))
        plt.suptitle(title)
        for i, w in enumerate(W):
            plt.subplot(nrows, ncols, i + 1)
            plt.axis('off')
            plt.imshow(w.reshape(6, 8), cmap='gray')
        plt.show()

    def train_and_display(data, Lhid_values, lambda_values, beta, rho):
        Lin = data.shape[1]
        for Lhid in Lhid_values:
            for lambda_ in lambda_values:
                params = {"Lin": Lin, "Lhid": Lhid, "beta": beta, "rho": rho, "lambda": lambda_}
                We = gradient_descent(data.T, params)
                W1 = We[:Lin*Lhid].reshape((Lhid, Lin))
                display_features(W1, f'Hidden Layer Features (Lhid={Lhid}, lambda={lambda_})')


    # Question 2 Functions
    def initialize_weights(D, P, V=250):
        word_embedding_weights = np.random.normal(0, 0.01, (V, D))
        embed_to_hid_weights = np.random.normal(0, 0.01, (D * 3, P))
        hid_to_output_weights = np.random.normal(0, 0.01, (P, V))
        hid_bias = np.ones((P, 1))
        output_bias = np.ones((V, 1))
        return word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias

    def stable_softmax(x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        return numerator / denominator

    def forward_pass(trigram, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias):
        word_embeddings = word_embedding_weights[trigram].reshape(D*3, 1)
        hidden_layer = 1 / (1 + np.exp(-(embed_to_hid_weights.T @ word_embeddings + hid_bias)))
        pre_output = hid_to_output_weights.T @ hidden_layer + output_bias
        output = stable_softmax(pre_output)
        return word_embeddings, hidden_layer, output

    def backpropagation(word_embeddings, hidden_layer, output, target, hid_to_output_weights):
        output_error = output - target
        hidden_error = (hid_to_output_weights @ output_error) * hidden_layer * (1 - hidden_layer)
        d_embed_to_hid = word_embeddings @ hidden_error.T
        d_hid_to_output = hidden_layer @ output_error.T
        return d_embed_to_hid, d_hid_to_output, output_error, hidden_error

    def train_network(D, P, trainx, traind, valx, vald, learning_rate=0.15, momentum_rate=0.85, max_epochs=50, batch_size=200, V=250):
        weights = initialize_weights(D, P, V)
        word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias = weights

        velocity_embed_to_hid = np.zeros_like(embed_to_hid_weights)
        velocity_hid_to_output = np.zeros_like(hid_to_output_weights)

        epoch = 0
        best_val_error = float('inf')

        while epoch < max_epochs:
            epoch += 1
            # Shuffle the training data
            permutation = np.random.permutation(len(trainx))
            trainx, traind = trainx[permutation], traind[permutation]

            for i in range(0, len(trainx), batch_size):
                batch_trainx = trainx[i:i + batch_size]
                batch_traind = traind[i:i + batch_size]

                d_embed_to_hid_sum = np.zeros_like(embed_to_hid_weights)
                d_hid_to_output_sum = np.zeros_like(hid_to_output_weights)

                for trigram, target in zip(batch_trainx, batch_traind):
                    # Forward pass
                    word_embeddings, hidden_layer, output = forward_pass(trigram, *weights)

                    # Backpropagation
                    d_embed_to_hid, d_hid_to_output, _, _ = backpropagation(word_embeddings, hidden_layer, output, target, hid_to_output_weights)

                    d_embed_to_hid_sum += d_embed_to_hid
                    d_hid_to_output_sum += d_hid_to_output

                # Update weights
                velocity_embed_to_hid = momentum_rate * velocity_embed_to_hid - learning_rate * d_embed_to_hid_sum / batch_size
                velocity_hid_to_output = momentum_rate * velocity_hid_to_output - learning_rate * d_hid_to_output_sum / batch_size

                embed_to_hid_weights += velocity_embed_to_hid
                hid_to_output_weights += velocity_hid_to_output

            # Calculate cross-entropy error on validation data
            val_error = 0
            for trigram, target in zip(valx, vald):
                _, _, output = forward_pass(trigram, *weights)
                val_error += -np.sum(target * np.log(output))
            val_error /= len(valx)

            # Stop training if validation error increases
            if val_error > best_val_error:
                break
            else:
                best_val_error = val_error

        return weights

    def test_network(weights, testx, testd, top_n=10, V=250):
        word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias = weights

        for trigram in testx[:5]:
            _, _, output = forward_pass(trigram, *weights)
            top_n_indices = output.flatten().argsort()[-top_n:][::-1]
            print(f'Trigram: {trigram}, Top {top_n} predicted words: {top_n_indices}')

    # Question 3 Functions
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # Defining the forward propagation function for RNN
    def forward_prop(X_t, W, b, W2, b2, h_prev):
        h = np.tanh(np.dot(W, np.concatenate([X_t, h_prev])) + b)
        y = softmax(np.dot(W2, h) + b2)
        return h, y

    # Defining the backward propagation function
    def back_prop(X, Y, W, b, W2, b2, h, y):
        grad_W2 = np.outer((y - Y), h)
        grad_b2 = y - Y
        grad_h = np.dot((y - Y), W2)
        grad_W = np.outer(grad_h, np.concatenate([X, h]))
        grad_b = grad_h
        return grad_W, grad_b, grad_W2, grad_b2

    # Function to initialize weights and biases
    def initialize_weights_and_biases_qs3(input_dim, hidden_dim):
        W = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim + hidden_dim))
        b = np.zeros(hidden_dim)
        return W, b

    # Defining the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Defining the forward propagation function for LSTM
    def forward_prop_lstm(X_t, Wf, bf, Wi, bi, Wo, bo, Wc, bc, W2, b2, h_prev, c_prev):
        concat_input = np.concatenate([X_t, h_prev])
        ft = sigmoid(np.dot(Wf, concat_input) + bf)
        c_tilda = np.tanh(np.dot(Wc, concat_input) + bc)
        it = sigmoid(np.dot(Wi, concat_input) + bi)
        c = ft * c_prev + it * c_tilda
        ot = sigmoid(np.dot(Wo, concat_input) + bo)
        h = ot * np.tanh(c)
        y = softmax(np.dot(W2, h) + b2)
        return h, c, y

    # Defining the backward propagation function for LSTM
    def back_prop_lstm(X, Y, Wf, bf, Wi, bi, Wo, bo, Wc, bc, W2, b2, h, c, y, h_prev, c_prev):
        concat_input = np.concatenate([X, h_prev])
        ft = sigmoid(np.dot(Wf, concat_input) + bf)
        c_tilda = np.tanh(np.dot(Wc, concat_input) + bc)
        it = sigmoid(np.dot(Wi, concat_input) + bi)
        ot = sigmoid(np.dot(Wo, concat_input) + bo)

        grad_W2 = np.outer((y - Y), h)
        grad_b2 = y - Y
        grad_h = np.dot((y - Y), W2)
        grad_o = grad_h * np.tanh(c) * ot * (1 - ot)
        grad_c = grad_h * ot * (1 - np.tanh(c)**2) + grad_o * np.tanh(c) * (1 - ot)
        grad_i = grad_c * c_tilda * it * (1 - it)
        grad_f = grad_c * c_prev * ft * (1 - ft)
        grad_c_tilda = grad_c * it * (1 - c_tilda**2)
        
        grad_Wo = np.outer(grad_o, concat_input)
        grad_bo = grad_o
        grad_Wf = np.outer(grad_f, concat_input)
        grad_bf = grad_f
        grad_Wi = np.outer(grad_i, concat_input)
        grad_bi = grad_i
        grad_Wc = np.outer(grad_c_tilda, concat_input)
        grad_bc = grad_c_tilda

        return grad_Wf, grad_bf, grad_Wi, grad_bi, grad_Wo, grad_bo, grad_Wc, grad_bc, grad_W2, grad_b2

    # Initializing weights and biases
    def initialize_weights_and_biases_lstm(input_dim, hidden_dim):
        Wf = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim + hidden_dim))
        bf = np.zeros(hidden_dim)
        Wi = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim + hidden_dim))
        bi = np.zeros(hidden_dim)
        Wo = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim + hidden_dim))
        bo = np.zeros(hidden_dim)
        Wc = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim + hidden_dim))
        bc = np.zeros(hidden_dim)
        return Wf, bf, Wi, bi, Wo, bo, Wc, bc


    if question == '1':
        print(question)
        with h5py.File('data1.h5', 'r') as f:
            data = np.array(f['data'])

        grayscale_data = rgb_to_grayscale(data)
        mean_removed_data = grayscale_data - grayscale_data.mean(axis=(1, 2), keepdims=True)
        std_data = mean_removed_data.std()
        clipped_data = np.clip(mean_removed_data, -3 * std_data, 3 * std_data)
        normalized_data = normalize_data(clipped_data)
        normalized_rgb = normalize_to_zero_one_range(data)

        sample_indices = random.sample(range(data.shape[0]), 200)

        sample_rgb = normalized_rgb[sample_indices]
        sample_normalized = normalized_data[sample_indices]

        display_images(sample_rgb, '200 Random Sample Patches - RGB Format', is_grayscale=False)
        display_images(sample_normalized, '200 Random Sample Patches - Normalized Grayscale')

        flattened_data = normalized_data.reshape(normalized_data.shape[0], -1)

        Lin = flattened_data.shape[1]
        Lhid = 64
        lambda_ = 5e-4

        beta_values = [0.1, 0.5, 1]
        rho_values = [0.05, 0.1, 0.2]

        for beta in beta_values:
            for rho in rho_values:
                params = {"Lin": Lin, "Lhid": Lhid, "beta": beta, "rho": rho, "lambda": lambda_}
                We = gradient_descent(flattened_data.T, params)
        
        # Extract the weights for the first layer
        W1 = We[:Lin*Lhid].reshape((Lhid, Lin))
        display_features(W1, 'Hidden Layer Features')

        Lhid_values = [20, 50, 80]
        lambda_values = [1e-5, 1e-4, 1e-3]
        beta = 0.1
        rho = 0.05
        train_and_display(flattened_data, Lhid_values, lambda_values, beta, rho)
    elif question == '2':
        print(question)
        with h5py.File('data2.h5', 'r') as f:
            trainx = np.array(f['trainx'])
            traind = np.array(f['traind'])
            valx = np.array(f['valx'])
            vald = np.array(f['vald'])
            testx = np.array(f['testx'])
            testd = np.array(f['testd'])

        trainx[trainx == 250] -= 1
        valx[valx == 250] -= 1
        testx[testx == 250] -= 1

        D, P = 32, 256
        trained_weights = train_network(D, P, trainx, traind, valx, vald)
        test_network(trained_weights, testx, testd)

        D, P = 16, 128
        trained_weights = train_network(D, P, trainx, traind, valx, vald)
        test_network(trained_weights, testx, testd)

        D, P = 8, 64
        trained_weights = train_network(D, P, trainx, traind, valx, vald)
        test_network(trained_weights, testx, testd)
    elif question == '3':
        print(question)
        
        # RNN
        with h5py.File('data3.h5', 'r') as hf:
            trX = hf['trX'][:]
            trY = hf['trY'][:]
            tstX = hf['tstX'][:]
            tstY = hf['tstY'][:]

        # Splitting the training set into training and validation sets
        train_size = int(0.9 * len(trX))
        val_size = len(trX) - train_size
        train_X, val_X = np.split(trX, [train_size])
        train_Y, val_Y = np.split(trY, [train_size])

        # Setting learning rate (eta), momentum factor (alpha) and number of training epochs
        eta = 0.1
        alpha = 0.85
        epochs = 50

        # Setting input dimension and hidden dimension sizes
        input_dim = 3
        hidden_dim = 128

        # Initializing weights and biases
        W, b = initialize_weights_and_biases_qs3(input_dim, hidden_dim)
        W2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim))
        b2 = np.zeros(6)

        val_errors = []

        for epoch in range(epochs):
            np.random.shuffle(train_X)

            num_batches = len(train_X) // 32

            for i in range(num_batches):
                X_batch = train_X[i*32:(i+1)*32]
                Y_batch = train_Y[i*32:(i+1)*32]

                for X, Y in zip(X_batch, Y_batch):
                    h_prev = np.zeros((hidden_dim,))
                    for t in range(150):
                        h, y = forward_prop(X[t], W, b, W2, b2, h_prev)
                        h_prev = h

                    grad_W, grad_b, grad_W2, grad_b2 = back_prop(X[-1], Y, W, b, W2, b2, h, y)

                    # Updating the weights and biases using the calculated gradients
                    W -= eta * grad_W + alpha * W
                    b -= eta * grad_b + alpha * b
                    W2 -= eta * grad_W2 + alpha * W2
                    b2 -= eta * grad_b2 + alpha * b2

            # Evaluating the model on the validation set after each epoch
            val_preds = np.array([forward_prop(x[-1], W, b, W2, b2, h_prev)[1] for x in val_X])
            val_loss = np.mean(-np.sum(val_Y * np.log(val_preds), axis=1))
            val_errors.append(val_loss)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

        # Plotting the validation loss over training epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs+1), val_errors, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.show()

        # Evaluating the model on the test set
        test_preds = np.array([forward_prop(x[-1], W, b, W2, b2, h_prev)[1] for x in tstX])
        test_preds_class = np.argmax(test_preds, axis=1)

        tstY_class = np.argmax(tstY, axis=1)

        test_preds_class = test_preds_class.astype(int)
        tstY_class = tstY_class.astype(int)

        # Calculating the test accuracy
        test_acc = np.mean(test_preds_class == tstY_class)
        print(f'Test Accuracy: {test_acc}')

        # Getting the predicted classes for the training set
        train_preds = np.array([forward_prop(x[-1], W, b, W2, b2, h_prev)[1] for x in train_X])
        train_preds_class = np.argmax(train_preds, axis=1)
        trainY_class = np.argmax(train_Y, axis=1)
        train_preds_class = train_preds_class.astype(int)
        trainY_class = trainY_class.astype(int)

        # Computing the confusion matrices for training and test sets
        conf_mat_train = confusion_matrix(trainY_class, train_preds_class)
        conf_mat_test = confusion_matrix(tstY_class, test_preds_class)

        print("Training Confusion Matrix: \n", conf_mat_train)
        print("Test Confusion Matrix: \n", conf_mat_test)

        # LSTM
        with h5py.File('data3.h5', 'r') as hf:
            trX = hf['trX'][:]
            trY = hf['trY'][:]
            tstX = hf['tstX'][:]
            tstY = hf['tstY'][:]

        # Splitting the training set into training and validation sets
        train_size = int(0.9 * len(trX))
        val_size = len(trX) - train_size
        train_X, val_X = np.split(trX, [train_size])
        train_Y, val_Y = np.split(trY, [train_size])

        # Setting learning rate (eta), momentum factor (alpha) and number of training epochs
        eta = 0.1
        alpha = 0.85
        epochs = 50

        # Setting input dimension and hidden dimension sizes
        input_dim = 3
        hidden_dim = 128

        # Initializing weights and biases
        Wf, bf, Wi, bi, Wo, bo, Wc, bc = initialize_weights_and_biases_lstm(input_dim, hidden_dim)
        W2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim))
        b2 = np.zeros(6)

        val_errors = []

        for epoch in range(epochs):
            np.random.shuffle(train_X)

            num_batches = len(train_X) // 32

            for i in range(num_batches):
                X_batch = train_X[i*32:(i+1)*32]
                Y_batch = train_Y[i*32:(i+1)*32]  

                for X, Y in zip(X_batch, Y_batch):
                    h_prev = np.zeros((hidden_dim,))
                    c_prev = np.zeros((hidden_dim,))
                    for t in range(150):
                        h, c, y = forward_prop_lstm(X[t], Wf, bf, Wi, bi, Wo, bo, Wc, bc, W2, b2, h_prev, c_prev)
                        h_prev = h
                        c_prev = c

                    grad_Wf, grad_bf, grad_Wi, grad_bi, grad_Wo, grad_bo, grad_Wc, grad_bc, grad_W2, grad_b2 = back_prop_lstm(X[-1], Y, Wf, bf, Wi, bi, Wo, bo, Wc, bc, W2, b2, h, c, y, h_prev, c_prev)

                    # Updating the weights and biases using the calculated gradients
                    Wf -= eta * grad_Wf + alpha * Wf
                    bf -= eta * grad_bf + alpha * bf
                    Wi -= eta * grad_Wi + alpha * Wi
                    bi -= eta * grad_bi + alpha * bi
                    Wo -= eta * grad_Wo + alpha * Wo
                    bo -= eta * grad_bo + alpha * bo
                    Wc -= eta * grad_Wc + alpha * Wc
                    bc -= eta * grad_bc + alpha * bc
                    W2 -= eta * grad_W2 + alpha * W2
                    b2 -= eta * grad_b2 + alpha * b2

            # Evaluating the model on the validation set after each epoch
            val_preds = np.array([forward_prop_lstm(x[-1], Wf, bf, Wi, bi, Wo, bo, Wc, bc, W2, b2, h_prev, c_prev)[2] for x in val_X])
            val_loss = np.mean(-np.sum(val_Y * np.log(val_preds), axis=1))
            val_errors.append(val_loss)
            print(f'Epoch {epoch+1}/{epochs} - validation loss: {val_loss}')

        # Plotting the validation loss over epochs
        plt.plot(range(1, epochs+1), val_errors)
        plt.title('Validation loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

faruk_ulutas_21803717_hw1(question)
