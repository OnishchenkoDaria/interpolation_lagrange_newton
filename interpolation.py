import time
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return np.exp(x**2)

INTERVAL_START = -1
INTERVAL_FINISH = 4

def user_input(default_exponent=4):
    exponent_input = input("Enter the degree of the interpolation polynomial (default is 4): ")
    try:
        exponent = int(exponent_input) if exponent_input else default_exponent
    except ValueError:
        print("Invalid input! Using default degree.")
        exponent = default_exponent
    return exponent + 1

#initialise interpolation nodes
def create_nodes(nodes_count):
    #equable interpolation
    x_range_equable = np.linspace(INTERVAL_START, INTERVAL_FINISH, nodes_count)
    y_range_equable = function(x_range_equable)

    #chebishov interpolation
    x_range_chebyshov = np.array([
        (INTERVAL_START + INTERVAL_FINISH)/2 + (INTERVAL_FINISH - INTERVAL_START)/2 * np.cos((2*i + 1)*np.pi / (2*nodes_count))
        for i in range(nodes_count)
    ])
    y_range_chebyshov = function(x_range_chebyshov)
    
    return x_range_equable, y_range_equable, x_range_chebyshov, y_range_chebyshov

def lagrange_polinom(x, nodes_count, y_range, x_range):
    L = 0  
    for i in range(nodes_count):
        l = 1  
        for j in range(nodes_count):
            if i != j:
                l *= (x - x_range[j]) / (x_range[i] - x_range[j])
        L += y_range[i] * l
    return L

def newton_polinom(x, nodes_count, y_range, x_range):
    n = nodes_count
    #compute f[]
    #create matrix = table
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_range  #first column is y values
    
    for j in range(1, n): 
        for i in range(n - j): 
            #formula
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x_range[i + j] - x_range[i])
            #results is an upper triangle matrix 

    #add diagonal elements as coefficients
    coefficients = divided_diff[0, :]
    #initialise polinom
    result = coefficients[0]
    product_term = 1
    for i in range(1, n):
        product_term *= (x - x_range[i - 1])
        result += coefficients[i] * product_term
    return result

def calculate_inaccuracy(f, p):
    inaccuracy_range = np.array([
        np.abs(f[i] - p[i])
        for i in range(len(f))
    ])
    return inaccuracy_range, np.max(inaccuracy_range)

def monotonous_check(f):
    differences = np.diff(f)
    if np.all(differences > 0):
        print("The function is strictly increasing")
        return True
    elif np.all(differences < 0):
        print("The function is strictly decreasing")
        return True    
    elif np.all(differences == 0):
        print("The function is constant")
        return True
    else:
        print("The function is not monotonous")
        return False

def test_case(kind, nodes_count, argument_test_imterval, function_values, arg_range_equable, arg_range_chebyshov, f_range_equable, f_range_chebyshov):
    #testing using Lagrange
    start_time = time.time()
    y_lagrange_equable = [lagrange_polinom(x, nodes_count, f_range_equable, arg_range_equable) for x in argument_test_imterval]
    lagrange_equable_time = time.time() - start_time
    
    start_time = time.time()
    y_lagrange_chebyshov = [lagrange_polinom(x, nodes_count, f_range_chebyshov, arg_range_chebyshov) for x in argument_test_imterval]
    lagrange_chebyshov_time = time.time() - start_time
    
    #testing using Newton
    start_time = time.time()
    y_newton_equable = [newton_polinom(x, nodes_count, f_range_equable, arg_range_equable) for x in argument_test_imterval]
    newton_equable_time = time.time() - start_time

    start_time = time.time()
    y_newton_chebyshov = [newton_polinom(x, nodes_count, f_range_chebyshov, arg_range_chebyshov) for x in argument_test_imterval]
    newton_chebyshov_time = time.time() - start_time

    #calculating inaccuracies
    equable_lagrange_inaccuracy, max_eq_lagrange = calculate_inaccuracy(function_values, y_lagrange_equable)
    chebyshov_lagrange_inaccuracy, max_ch_lagrange = calculate_inaccuracy(function_values, y_lagrange_chebyshov)
    equable_newton_inaccuracy, max_eq_newton = calculate_inaccuracy(function_values, y_newton_equable)
    chebyshov_newton_inaccuracy, max_ch_newton = calculate_inaccuracy(function_values, y_newton_chebyshov)

    print(f"\n {kind} Interpolation Results:\n")
    print(f"Max inaccuracy (Equable Lagrange): {max_eq_lagrange:.6f}, Time = {lagrange_equable_time:.6f} seconds")
    print(f"Max inaccuracy (Chebyshev Lagrange): {max_ch_lagrange:.6f}, Time = {lagrange_chebyshov_time:.6f} seconds")
    print(f"Max inaccuracy (Equable Newton): {max_eq_newton:.6f}, Time = {newton_equable_time:.6f} seconds")
    print(f"Max inaccuracy (Chebyshev Newton): {max_ch_newton:.6f}, Time = {newton_chebyshov_time:.6f} seconds\n")

    #plot Lagrange and Newton polynomials
    plt.plot(argument_test_imterval, y_lagrange_equable, label="Lagrange (Equable)", color="red", linestyle="--", linewidth=2)
    plt.plot(argument_test_imterval, y_lagrange_chebyshov, label="Lagrange (Chebyshev)", color="blue", linestyle="--", linewidth=2)
    plt.plot(argument_test_imterval, y_newton_equable, label="Newton (Equable)", color="orange", linestyle=":", linewidth=2)
    plt.plot(argument_test_imterval, y_newton_chebyshov, label="Newton (Chebyshev)", color="green", linestyle=":", linewidth=2)
    
    #plot interpolation nodes
    plt.scatter(arg_range_equable, f_range_equable, color="red", label="Equable Nodes", zorder=nodes_count)
    plt.scatter(arg_range_chebyshov, f_range_chebyshov, color="blue", label="Chebyshev Nodes", zorder=nodes_count)

    plt.title(f"{kind} Interpolation Visualization")
    plt.xlabel("Argument")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    #visualization of inaccuracies
    plt.figure(figsize=(12, 8))
    plt.plot(argument_test_imterval, equable_lagrange_inaccuracy, label="Lagrange (Equable)", color="red", linewidth=4)
    plt.plot(argument_test_imterval, chebyshov_lagrange_inaccuracy, label="Lagrange (Chebyshev)", color="blue", linewidth=4)
    plt.plot(argument_test_imterval, equable_newton_inaccuracy, label="Newton (Equable)", color="orange", linewidth=2)
    plt.plot(argument_test_imterval, chebyshov_newton_inaccuracy, label="Newton (Chebyshev)", color="green", linewidth=2)
    plt.title(f"{kind} Interpolation Inaccuracies")
    plt.xlabel("argument")
    plt.ylabel("Inaccuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    nodes_count = user_input()
    #initialise interval of interpokation
    interval = np.linspace(INTERVAL_START, INTERVAL_FINISH, nodes_count)
    x_range_equable, y_range_equable, x_range_chebyshov, y_range_chebyshov = create_nodes(nodes_count)
    
    #setting actual values
    x_test_interval = np.linspace(INTERVAL_START, INTERVAL_FINISH, 100)
    y_values = [function(x) for x in x_test_interval]

    #true interpolation
    test_case("True", nodes_count, x_test_interval, y_values, x_range_equable, x_range_chebyshov, y_range_equable, y_range_chebyshov)

    #inverse interpolation
    if(monotonous_check(y_values)):
        test_case("Inverse", nodes_count, y_values, x_test_interval, y_range_equable, y_range_chebyshov, x_range_equable, x_range_chebyshov)