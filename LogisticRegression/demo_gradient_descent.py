if __name__ == '__main__':
    # Function to minimise
    fc = lambda x, y: (3 * x ** 2) + (x * y) + (5 * y ** 2)
    # set partial derivates
    partial_derivate_x = lambda x, y: (6 * x) + y
    partial_derivate_y = lambda x, y: (10 * y) + x
    # set variables
    x = 10
    y = -13
    # Learning rate
    learning_rate = 0.1
    print("FC = %s" % (fc(x, y)))
    # one epoch ins one period of minimisation
    for epoch in range(0, 20):
        # Compute gradients
        x_gradient = partial_derivate_x(x, y)
        y_gradient = partial_derivate_y(x, y)
        # Apply gradient descent
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        # Keep track of the function value
        print("FC = %s" % (fc(x, y)))

    # Print final variables values
    print("")
    print("x = %s" % x)
    print("y = %s" % y)
