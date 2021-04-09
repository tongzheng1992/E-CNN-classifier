import tensorflow as tf

def average_utility(utility_matrix, inputs, labels, act_set):
  utility = 0
  for i in range(len(inputs)):
    x = inputs[i]
    y = labels[i]
    utility += utility_matrix[x,y]
  average_utility = utility/len(inputs)
  return average_utility