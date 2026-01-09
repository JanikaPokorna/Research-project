import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
beta = 0.3
gamma = 0.1
N = 1000

def SIR(t,y):
    S,I,R = y
    Sdot = -beta * S * I /N
    Idot = beta * S * I/N - gamma * I
    Rdot = gamma * I
    return[Sdot,Idot,Rdot]

# initial conditions
y0 = [990, 10, 0]
t_span = (0,100)
t_eval = np.linspace(*t_span, 30)

sol = solve_ivp(SIR, t_span, y0, 'RK45', t_eval)

S, I, R = sol.y 


# plot

plt.figure(figsize=(8,4))
plt.plot(sol.t, S, label='susceptible')
plt.plot(sol.t, I, label='infected')
plt.plot(sol.t, R, label='recovered')
plt.legend()
plt.xlabel("Days")
plt.ylabel("People")
plt.title("SIR Epidemic Model")
plt.grid(True)
plt.show()