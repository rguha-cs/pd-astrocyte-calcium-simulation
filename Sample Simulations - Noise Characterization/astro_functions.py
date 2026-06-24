import cupy as cp

# Define inhibition function for IP3 receptor 
def ip3r_inhibition(Ca, K_inh, m):
    return 1 / (1 + (Ca / (K_inh * 2)) ** m)  

# Compute IP₃R-mediated Ca²⁺ release with inhibition
def compute_J_IP3R(ip3_val, Ca, Ca_ER, k_IP3R, K_IP3R, n, K_inh, m):
    ip3_activation = (ip3_val ** n) / (K_IP3R ** n + ip3_val ** n)
    inhibition = 1 / (1 + (Ca / (K_inh * 4.5)) ** m)  
    return k_IP3R * ip3_activation * inhibition * (Ca_ER - Ca) * 0.18 

# Compute SERCA uptake with ER-dependent boost
def compute_J_SERCA(Ca, Ca_ER, V_SERCA, K_SERCA, alpha):
    serca_base = V_SERCA * (Ca ** 4 / (K_SERCA ** 4 + Ca ** 4)) 
    serca_boost = (1 + alpha * Ca_ER) 
    return serca_base * serca_boost * 1.5 

# Compute CICR-mediated calcium release
def compute_J_CICR(Ca, Ca_ER, k_CICR, K_CICR):
    activation = (Ca ** 2 / (K_CICR ** 2 + Ca ** 2))
    inhibition = 1 / (1 + (Ca / 4.0) ** 2)
    return k_CICR * activation * (Ca_ER - Ca) * inhibition * 0.05

# Sigmoid function for IP₃ receptor activation
def sigmoid_ip3(ip3_val, K_IP3R, n):
    return ip3_val ** n / (K_IP3R ** n + ip3_val ** n)

