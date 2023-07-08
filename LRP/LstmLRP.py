


def lstm_lrp(model, input_data):
    
    Rj, _, _ = model.predict(input_data) # return the output/prediction to test LRP for LSTMs (1, timesteps, 1)
    Rj = Rj[:,-1,:] # only the last prediction matters
    # get all hidden states and cell states for all time steps
    hidden_states, cell_states, gate_activations, signal_activations = get_lstm_states(model, input_data, True)


    # Initialise the relevance scores 

    #Ri = 0 # input gate does not contribute itself (signal takes it all)
    #Ro = 0 # output gate does not contribute itself (signal takes it all)
    #Rf = 0 # forget gat does not contribute itself (signal takes it all)
    Rc = 0 # cell state relevance 
    #Rh = 0 # hidden state relevance

    relevance = []

    for t in reversed(range(timesteps)): 
        ap = gate_activations[t, 0, :,:] * signal_activations[t, 0, :, :]
        Rp = np.multiply.reduce(gate_activations[t:, 1, :,:]) * ap * Rj#cell_states[-1]
        Rc = Rp
        

        aj = input_data[0, t, :] # !! REPLACE THIS LATER WITH more generic input    
        Rj = np.dot(aj, w_c) #### !!!!!!!!!!!!!
        relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], signal_activations[0, 0, :, :].reshape(50), Rc.reshape(50), 3))
        
    return relevance


def lstm_lrp_rudder(model, input_data):
    # source: https://arxiv.org/pdf/1806.07857.pdf
    
    # return the output/prediction to test LRP for LSTMs (1, timesteps, 1)
    RyT, _, _ = model.predict(input_data) 

    # prediction y_T
    RyT = RyT[0,-1,:] 
    
    # get all hidden states and cell states for all time steps
    hidden_states, cell_states, gate_activations, signal_activations = get_lstm_states(model, input_data, True)

    # last cell state
    cT = cell_states[-1, 0, :]

    relevance = []

    for t in reversed(range(timesteps)): 
        
        # rules according to Rudder
        zt = signal_activations[t, 0, 0, :]
        it = gate_activations[t, 0, 0, :]
        Rzt =  (zt * it) * RyT / cT
        
        # using linear rule
        relevance.append(lrp_linear(np.array(w_c), np.array(b_c), input_data[0, t, :], zt, Rzt, 3))
        
    return relevance