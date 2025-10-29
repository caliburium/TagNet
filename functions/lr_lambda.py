"""
def lr_lambda(progress):
    alpha = 0.1
    beta = 0.1
    return (1 + alpha * progress) ** (-beta)

"""
def lr_lambda(progress):
    # mu_0 = 0.01
    alpha = 10
    beta = 0.75
    # return mu_0 * (1 + alpha * progress) ** (-beta)
    return (1 + alpha * progress) ** (-beta)

