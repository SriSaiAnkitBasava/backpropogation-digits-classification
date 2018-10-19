import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

"""create your dataset here and save"""


def create_dataset():
    x_point = np.random.uniform(0, 1, 300)
    v_point = np.random.uniform(-0.1, 0.1, 300)
    desired_point = desired_out(x_point, v_point)
    plt.figure(1)
    #ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    """x-points vs desired-output points plot"""
    plt.scatter(x_point, desired_point)
    #plt.savefig('dataset_created.png')
    np.save("x_points", x_point)
    np.save("v_points", v_point)
    np.save("d_points", desired_point)
    return x_point, v_point, desired_point


def desired_out(x, v):
    d_i = []
    for di in range(0, 300):
        temp_di = float(np.sin(20 * x[di]) + (3 * x[di]) + v[di])
        d_i.append(temp_di)
        # d_i.insert(float(temp_di), di)
    return d_i


def desired_out_network(x, v, w, size):
    if size == 2:
        t_w = w

        temp_di = float((t_w[0] * (x)) + (t_w[1]))

        #temp_di = float((np.sin(20 * x)) + (t_w[0] * x) + (t_w[1]))

        #temp_di = float((t_w[0] * np.sin(20 * x)) + (t_w[0] * x) + (t_w[1]))
        """1st round assumed equation"""
        #temp_di = float(np.sin(t_w[0] * x) + (t_w[0] * x) + (v * t_w[1]))
    else:
        if size == 1:
            temp_di = float((w * (x)))

            #temp_di = float((np.sin(20 * x)) + (w * x))

            #temp_di = float((np.sin(w * x)) + (w * x))
            """1st round assumed equation"""
            #temp_di = float(np.sin(w * x) + (w * x))
    return temp_di


def desired_out_activation(l, size):
    lo_i = []
    for di in range(0, size):
        #temp_lo_i = float(1 / (1 + math.exp(l[di])))
        temp_lo_i = float(np.tanh(l[di]))
        lo_i.append(temp_lo_i)
        # d_i.insert(float(temp_di), di)
    return lo_i


def desired_out_activation_backpropogation(l, size):
    lo_i = []
    for di in range(0, size):
        temp_lo_i = float(1.0 - np.tanh(l)**2)
        lo_i.append(temp_lo_i)
        # d_i.insert(float(temp_di), di)
    return lo_i


"""output side weights backprop"""


def out_backpropogation(local, err, size):
    gradient_out = []
    derivative_output = 1
    #derivative_output = desired_out_activation_backpropogation(local, 1)

    for i in range(0, size):
        """
        print "-----------------"
        print derivative_output
        print local
        print err
        print "-----------------"
        """
        temp_gradient = -1.0 * local[i] * err * derivative_output
        gradient_out.append(temp_gradient)

    return gradient_out

def hidden_backpropogation(w, local, err, x):
    gradient_hidden = []
    derivative_output = desired_out_activation_backpropogation(local, 1)
    temp_hidden_weights = -1 * err * derivative_output * w * x
    gradient_hidden.append(temp_hidden_weights)
    temp_hidedn_bias = -1 * err * derivative_output * w
    gradient_hidden.append(temp_hidedn_bias)
    return gradient_hidden




"""initialize the data set and other stuff"""

x, v, d_out = create_dataset()
learning_rate = 0.05
epochs = 0


#weights_input = np.random.uniform(-1, 1, (24, 2))
#weights_output = np.random.uniform(-1, 1, (1, 24))

weights_input = np.random.rand(240, 2)
weights_output = np.random.rand(1, 240)


weights_test = weights_input[1].reshape(2, 1)
error_count = 0
threshold_error = 0.0001
# print x[1]
print weights_test
print np.shape(weights_test)
output_bias = np.random.rand(1)
previous_error = 300.0
final_d = []
exit_condition = -1
sigma_error = 0.0
check_var = 0


"""start backpropogation algorithm"""

while exit_condition < 1:

    epochs = epochs + 1
    calculated_output = []
    sigma_error = 0.0
    """show the input to the network (forward network)"""
    for k in range(0, 300):

        x_i = x[k]
        v_i = v[k]
        """hidden neuron"""
        localfield_input = []
        for i in range(0, 240):
            local_weights = weights_input[i].reshape(2, 1)
            localfield_temp = desired_out_network(x_i, v_i, local_weights, 2)
            localfield_input.append(localfield_temp)
        hidden_activation = desired_out_activation(localfield_input, 240)

        """output neuron"""
        localfield_output = 0.0
        local_output_all = []
        for j in range(0, 240):
            local_output_weights = weights_output[0][j]
            local_output = desired_out_network(hidden_activation[j], v_i, local_output_weights, 1)
            local_output_all.append(local_output)
            localfield_output = localfield_output + local_output
        #print localfield_output
        #print "-----------------"
        pre_output_activation = localfield_output + output_bias

        """cannot take values more than 1"""
        output_activation = pre_output_activation
        #output_activation = desired_out_activation(pre_output_activation, 1)

        calculated_output.append(output_activation)

        #print "network output in forward direction ", k, ":", output_activation

        """start feedback network(backward propogation"""

        """output side weight update"""
        hidden_gradient_out = []
        #print "test test-----"
        #print d_out[k]
        #print calculated_output[k]
        temp_error = (d_out[k]) - calculated_output[k]
        #temp_error = ((np.asarray(np.asarray(d_out[k]) - calculated_output[k])))
        error = temp_error * temp_error * 1/2
        sigma_error = sigma_error + error
        gradient_output = out_backpropogation(hidden_activation, temp_error, 240)
        #gradient_output = out_backpropogation(localfield_output, error, 1)

        """hidden neurons gradient"""
        for h in range(0, 240):
            temp_hidden_gradient = hidden_backpropogation(weights_output[0][h], localfield_input[h], temp_error, x[k])
            hidden_gradient_out.append(temp_hidden_gradient)

        """
        print "--------------*************"
        print weights_output[0][k]
        print learning_rate
        print gradient_output[0]

        print "--------------*************"
        """
        if check_var == 0:
            output_bias = output_bias - float(learning_rate * temp_error)
            #print "error percentage: ", error, "hidden neuron:", k
            for e in range(0, 24):
                weights_output[0][e] = (weights_output[0][e]) - (learning_rate * (gradient_output[e]))
                #print "didden gradient: ", hidden_gradient_out[e]
                #print "hidden gradent count: ", hidden_gradient_out.count()
                #print "weights: ", weights_input[e]
                temp_gradient_hidden = np.asarray(hidden_gradient_out[e])
                temp_gradient_hidden = np.reshape(temp_gradient_hidden, (2, 1))
                temp_gradient_hidden = np.asarray(temp_gradient_hidden)
                #print "temp gradent: ", temp_gradient_hidden
                weights_input[e] = weights_input[e] - np.reshape((learning_rate * temp_gradient_hidden), (1, 2))

        else:
            if check_var == 1:
                print "done-------"
                #print "total error(check): ", sigma_error, "epoch number: ", epochs

    sigma_error = float(sigma_error/300.0)
    print "total error: ", sigma_error, "epoch number: ", epochs




    if float(previous_error) < float(sigma_error):
        print "learning rate changed---"
        learning_rate = learning_rate * 0.9

    temp_cal_error = abs(float(sigma_error-previous_error))

    if (float(temp_cal_error) < float(threshold_error)) & (check_var == 0):
        check_var = 1
        exit_condition = exit_condition + 1
        print "reached exit condition"
    else:
        previous_error = sigma_error
        sigma_error = 0


    if check_var == 1:
        final_d = calculated_output
        exit_condition = exit_condition + 1
        break
    else:
        calculated_output = []


print "x-point shape:", x.shape
print "final output points", len(final_d)
print "-------final plot-------------"
"""
final_final_d = []
for ch in range(0, 300):
    tem_input_weights = np.asarray(weights_input[ch])
    temp_out_cal = float((tem_input_weights[0] * np.sin(x[ch])) + (v[ch] * tem_input_weights[1]))
    final_final_d.append(temp_out_cal)
    
"""
plt.figure(1)
plt.scatter(x, final_d, color='red')
plt.savefig('final-plot2.png')


print "after all inputs---------"
print x
print final_d
print d_out
