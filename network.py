from readData import load
import numpy as np
import matplotlib.pyplot as plt

"""load the data created by readData.py file"""
i_train, l_train, i_test, l_test = load()

x = i_train
y = l_train


"""I tried to use different activation functions in this function
 to see if has any impact on the error rate"""
def sigmoid_activation(si):
    si_return = []
    sigmoid_slope = 10
    for ha in range(0, 10):
        temp_cal = 1/(1 + np.exp(-(sigmoid_slope * si[ha])))
        #temp_cal = ((np.exp((sigmoid_slope * si[ha])) - np.exp(-(sigmoid_slope * si[ha])))) / ((np.exp((sigmoid_slope * si[ha])) + np.exp(-(sigmoid_slope * si[ha]))))
        si_return.insert(ha, temp_cal)
    return si_return

"""This is the activation function thr training algorithm 
is using"""
def step_activation_func(s):
    s_return = []
    for f in range(0, 10):
        if s[f] >= 0:
            s_return.insert(f, 1)
        else:
            s_return.insert(f, 0)
    return s_return


def max_index_func(v):
    return_index = 0
    for g in range(0, len(v) - 1):
        if v[g] >= v[return_index]:
            return_index = g
    return return_index


"""initialise the required variables"""
learning_rate = 0.1
epoch = 0
error = 0
exit_threshold = 10
change_condition = 0

"""Tried to use different weight distributions to see if that
 has a impact on the error rate"""
#weights = np.random.uniform(low=-1, high=1, size=(10, 784))
weights = np.random.rand(10, 784)

bias = np.random.rand(10, 1)
induced_localfield = []
induced_localfield_temp = []
n = 1000
d = []
max_index_array = []
epoch_plot = []
error_plot = []
error_plot.append(0)
epoch_plot.append(0)


"""Algorithm starts here"""
while True:
    for i in range(0, n):
        temp_train = x[i]
        for j in range(0, 10):
            temp_weight = np.asarray(weights[j])
            induced_localfield_cal = np.dot(temp_weight, temp_train)
            induced_localfield_temp.insert(j, induced_localfield_cal)
        induced_localfield.append(induced_localfield_temp)
        induced_localfield_temp = []
        max_index = max_index_func(induced_localfield[i])
        max_index_array.append(max_index)
        if not (max_index == y[i]):
            error = error + 1
    if change_condition == 1:
        test_error = (float(error)/float(n)) * 100
        print "number of errors on test::", error, "error percentage", test_error

    epoch = epoch + 1

    if change_condition == 1:
        break
    else:

        for k in range(0, n):
            for out in range(0, 10):
                label_temp = y[k]
                if out == label_temp:
                    d.insert(out, 1)
                else:
                    d.insert(out, 0)
            #d_out = sigmoid_activation(induced_localfield[k])
            d_out = step_activation_func(induced_localfield[k])
            i_train_temp = i_train[k]
            for w in range(0, 10):
                d_diff = (d[w] - d_out[w])
                weights_delta = (learning_rate * d_diff) * np.array(i_train_temp)
                weights[w] = weights[w] + weights_delta
            d = []
            d_out = []

        error_percentage = (float(error) /float(n)) * 100
        print "epoch number::", epoch, "number of error::", error, "error %::", error_percentage
        epoch_plot.append(epoch)
        error_plot.append(error)
        error = 0
        induced_localfield = []
        max_index_array = []
        d = []
        if error_percentage <= float(exit_threshold):
            change_condition = 1
            error = 0
            x = i_test
            y = l_test
            n = 10000
        else:
            error_percentage = 0

plt.figure(1)
plt.plot(epoch_plot, error_plot, color="red")
plt.xlabel("epoch number")
plt.ylabel("number of errors")
plt.savefig('epochVSerror.png')