def update(params, grads, eta):
    W_upd = []
    b_upd = []
    
    for i in range(len(params[1])):
        W_upd.append(params[0][i] - eta * grads[0][i])
        b_upd.append(params[1][i] - eta * grads[1][i])

    return W_upd, b_upd