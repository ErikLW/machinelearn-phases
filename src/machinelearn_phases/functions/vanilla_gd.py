def update(params, grads, eta, momentum = False, velocity = None):
    W_upd = []
    b_upd = []
    W_velo = []
    b_velo = []

    if momentum != False:
        if velocity == None:
            velocity = grads
        else:
            for i in range(len(params[1])):
                W_velo.append(grads[0][i] + momentum * velocity[0][i])
                b_velo.append(grads[1][i] + momentum * velocity[1][i])
            velocity = (W_velo, b_velo)


        for i in range(len(params[1])):
            W_upd.append(params[0][i] - eta * velocity[0][i])
            b_upd.append(params[1][i] - eta * velocity[1][i])
        params = (W_upd, b_upd)
        return params, velocity


    
    else:
        for i in range(len(params[1])):
            W_upd.append(params[0][i] - eta * grads[0][i])
            b_upd.append(params[1][i] - eta * grads[1][i])
        params = (W_upd, b_upd)
        return params
    