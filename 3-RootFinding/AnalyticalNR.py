#!/usr/bin/env python
# coding: utf-8

# In[1]:


#N_bar, the total change in mols.
def N_bar(dnvec):
    import jax.numpy as jnp
    return jnp.sum((dnvec))


# In[2]:


#nu, a constant dependant on EOS, that affects a.
def nu(EOS):
    if EOS == 'SRK':
        return 0.42748
    if EOS == 'PR':
        return 0.45724


# In[3]:


#c, the accentricity polynomial.
def c(i, EOS, w):
    if EOS == 'SRK':
        ci = 0.48
        ci += 1.574*w[i]
        ci += -0.176*w[i]**2
        return ci
    if EOS == 'PR':
        ci = 0.37464
        ci += 1.54226*w[i]
        ci += -0.26992*w[i]**2
        return ci


# In[4]:


#ai, the attraction paramater of component i.
def ai(i, Tc_mx, Tc, R, EOS, Pc, w):
    ai = (R*Tc[i])**2*nu(EOS)/Pc[i]
    ai = ai*(1 + c(i, EOS, w)*(1-(Tc_mx/Tc[i])**0.5))**2
    return ai


# In[5]:


#aij, the binary attraction paramater of component system i-j.
def aijf(Tc_mx, Tc, R, EOS, Pc, w, k, C):
    import jax.numpy as jnp
    aij = jnp.zeros([C, C])
    for i in range(C):
        for j in range(C):
            aij = aij.at[i,j].set((ai(i, Tc_mx, Tc, R, EOS, Pc, w)*ai(j, Tc_mx, Tc, R, EOS, Pc, w))**0.5*(1-k[i][j]))
    return aij


# In[6]:


#bi, the repulsion paramater of component i.
def b(i, EOS, R, Tc, Pc):
    if EOS == 'SRK':
        bi = 0.08664*R*Tc[i]/Pc[i]
        return bi
    if EOS == 'PR':
        bi = 0.07780*R*Tc[i]/Pc[i]
        return bi


# In[7]:


#D1, a parameter that defines the EOS.
def D1(EOS):
    import jax.numpy as jnp
    if EOS == 'SRK':
        u0 = 1
        w0 = 0
    if EOS == 'PR':
        u0 = 2
        w0 = -1      
    D1 = (u0 + jnp.sqrt(u0**2-4*w0))/2
    return D1


# In[8]:


#D2, a parameter that defines the EOS.
def D2(EOS):
    import jax.numpy as jnp
    if EOS == 'SRK':
        u0 = 1
        w0 = 0
    if EOS == 'PR':
        u0 = 2
        w0 = -1      
    D2 = (u0 - jnp.sqrt(u0**2-4*w0))/2
    return D2


# In[9]:


#a_tot, the weighted sum of binary attraction interactions.
def a_tot(n, n_tot, aij, C):
    a_t = 0
    for i in range(C):
        for j in range(C):
            a_t += n[i]*n[j]/n_tot**2*aij[i, j]
    return a_t


# In[10]:


#b_tot, the weighted sum of repulsion interactions.
def b_tot(y, EOS, R, Tc, Pc, C):
    b_t = 0
    for i in range(C):
        b_t += y[i]*b(i, EOS, R, Tc, Pc)
    return b_t


# In[11]:


#alphak, the relative attraction of all interactions with component k.
def alpha(j, y, aij, n, n_tot, C):
    alpk = 0
    for i in range(C):
        alpk += y[i]*aij[i,j]
    return alpk/a_tot(n, n_tot, aij, C)


# In[12]:


#alpha_bar, total change in the alphaks from delta_n.
def alpha_bar(dnvec, y, aij, n, n_tot, C):
    alp_bar = 0
    for i in range(C):
        alp_bar += dnvec[i]*alpha(i, y, aij, n, n_tot, C)
    return alp_bar


# In[13]:


#a_bar, the realtive change in a_tot due from delta_n.
def a_bar(dnvec, aij, n, n_tot, C):
    a_b = 0
    for i in range(C):
        for j in range(C):
            a_b += dnvec[i]*dnvec[j]*aij[i,j]
    a_b = a_b/a_tot(n, n_tot, aij, C)
    return a_b


# In[14]:


#betai, the relative repulsion of component k.
def beta(i, EOS, R, Tc, Pc, y, C):
    return b(i, EOS, R, Tc, Pc)/b_tot(y, EOS, R, Tc, Pc, C)


# In[15]:


#beta_bar, the total change in betais from delta_n.
def beta_bar(dnvec, EOS, R, Tc, Pc, y, C):
    bet_bar = 0
    for i in range(C):
        bet_bar += dnvec[i]*beta(i, EOS, R, Tc, Pc, y, C)
    return bet_bar


# In[16]:


#K, the ratio of critcal volume of mixture to repulsion parameter.
def K(Vc_mx, y, EOS, R, Tc, Pc, C):
    return Vc_mx/b_tot(y, EOS, R, Tc, Pc, C)


# In[17]:


#F1-F6 are EOS based factors which are f(D1, D2, K)
def F1(Kv, EOS):
    return 1/(Kv-1)

def F2(Kv, EOS):
    F2 = 2/(D1(EOS)-D2(EOS))
    F2 *= (D1(EOS)/(Kv+D1(EOS))-D2(EOS)/(Kv+D2(EOS)))
    return F2

def F3(Kv, EOS):
    F3 = 1/(D1(EOS)-D2(EOS))
    F3 *= ((D1(EOS)/(Kv+D1(EOS)))**2-(D2(EOS)/(Kv+D2(EOS)))**2)
    return F3

def F4(Kv, EOS):
    F4 = 1/(D1(EOS)-D2(EOS))
    F4 *= ((D1(EOS)/(Kv+D1(EOS)))**3-(D2(EOS)/(Kv+D2(EOS)))**3)
    return F4

def F5(Kv, EOS):
    import jax.numpy as jnp
    F5 = 2/(D1(EOS)-D2(EOS))
    F5 *= jnp.log((Kv+D1(EOS))/(Kv+D2(EOS)))
    return F5

def F6(Kv, EOS):
    import jax.numpy as jnp
    F6 = 2/(D1(EOS)-D2(EOS))
    F6 *= (D1(EOS)/(Kv+D1(EOS))-D2(EOS)/(Kv+D2(EOS)))-jnp.log((Kv+D1(EOS))/(Kv+D2(EOS)))
    return F6


# In[18]:


#Generalized vector of functions that descirbes a cubic equation of state.

def GeneralizedCubicFunction(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k):
    import jax.numpy as jnp
    #Unpack input xvec.
    Tc_mx = xvec[0]
    Vc_mx = xvec[1]
    dnvec = xvec[2:]

    fvec = jnp.zeros([1,len(xvec)])
    
    #Calculate variable values
    aij = aijf(Tc_mx, Tc, R, EOS, Pc, w, k, C)
    a_totv = a_tot(n, n_tot, aij, C)
    b_totv = b_tot(y, EOS, R, Tc, Pc, C)
    N_barv = N_bar(dnvec)
    a_barv = a_bar(dnvec, aij, n, n_tot, C)
    alpha_barv = alpha_bar(dnvec, y, aij, n, n_tot, C)
    beta_barv = beta_bar(dnvec, EOS, R, Tc, Pc, y, C)
    
    Kv = K(Vc_mx, y, EOS, R, Tc, Pc, C)
    
    F1v = F1(Kv, EOS)
    F2v = F2(Kv, EOS)
    F3v = F3(Kv, EOS)
    F4v = F4(Kv, EOS)
    F5v = F5(Kv, EOS)
    F6v = F6(Kv, EOS)
    
    
    #First derivative of each component's Helmholtz Energy should be 0. 
    for i in range(C):
        betav = beta(i, EOS, R, Tc, Pc, y, C)
        pA = R*Tc_mx/n_tot
        pB = a_totv/(b_totv*n_tot)
        A1 = dnvec[i]/y[i]
        A2 = F1v*(betav*N_barv + beta_barv)
        A3 = betav*F1v**2*beta_barv
        B1 = betav*beta_barv*F3v
        B2 = 0
        for j in range(C):
            B2 += dnvec[j]*aij[i,j]
        B2 = -F5v/a_totv*B2
        B3 = F6v*(betav*beta_barv-alpha(i, y, aij, n, n_tot, C)*beta_barv-alpha_barv*betav)
        A = pA*(A1+A2+A3)
        B = pB*(B1+B2+B3)
        fvec = fvec.at[0,i].set(A+B)
    
    #Sum of all second derivatives of Helmholtz Energy should be 0.
    pA = R*Tc_mx/n_tot**2
    pB = a_totv/(b_totv*n_tot**2)
    A1 = 0
    for i in range(C):
        A1 += dnvec[i]**3/y[i]**2
    A1 = -A1
    A2 = 3*(N_barv*(beta_barv*F1v)**2)
    A3 = 2*((F1v*beta_barv)**3)
    B1 = 3*(beta_barv**2)*((2*alpha_barv-beta_barv)*(F3v+F6v))
    B2 = -2*(beta_barv**3)*F4v
    B3 = -3*beta_barv*a_barv*F6v
    A = pA*(A1+A2+A3)
    B = pB*(B1+B2+B3)
    fvec = fvec.at[0,C].set(A+B)
    
    #Euclidean distance of delta_n should be 1.
    norm_mols = 0
    for i in range(C):
        norm_mols += dnvec[i]**2
    norm_mols += -1
    fvec = fvec.at[0,C+1].set(norm_mols)
    return fvec


# In[19]:


def CalcParams(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k):
    import jax.numpy as jnp
    #Unpack input xvec.
    Tc_mx = xvec[0]
    Vc_mx = xvec[1]
    dnvec = xvec[2:]

    fvec = jnp.zeros([1,len(xvec)])
    
    #Calculate variable values
    aij = aijf(Tc_mx, Tc, R, EOS, Pc, w, k, C)
    a_totv = a_tot(n, n_tot, aij, C)
    b_totv = b_tot(y, EOS, R, Tc, Pc, C)
    N_barv = N_bar(dnvec)
    a_barv = a_bar(dnvec, aij, n, n_tot, C)
    alpha_barv = alpha_bar(dnvec, y, aij, n, n_tot, C)
    beta_barv = beta_bar(dnvec, EOS, R, Tc, Pc, y, C)
    
    ALPHA = jnp.zeros(C)
    BETA = jnp.zeros(C)
    for i in range(C):
        ALPHA = ALPHA.at[i].set(alpha(i, y, aij, n, n_tot, C))
        BETA = BETA.at[i].set(beta(i, EOS, R, Tc, Pc, y, C))
    
    Kv = K(Vc_mx, y, EOS, R, Tc, Pc, C)
    
    F1v = F1(Kv, EOS)
    F2v = F2(Kv, EOS)
    F3v = F3(Kv, EOS)
    F4v = F4(Kv, EOS)
    F5v = F5(Kv, EOS)
    F6v = F6(Kv, EOS)
    return aij, a_totv, b_totv, N_barv, a_barv, ALPHA, alpha_barv, BETA, beta_barv, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, Vc_mx, dnvec


# In[20]:


def df1dn_(y, aij, a, b, ALPHA, BETA, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, R, C, n_tot):
    import jax.numpy as jnp
    M1 = R*Tc_mx*n_tot
    M2 = a/(b*n_tot)
    df1dn = jnp.zeros([C, C])
    kd = jnp.eye(C)
    for i in range(C):
        for j in range(C):
            T1 = 1/y[i]*kd[i, j]
            T2 = F1v*(BETA[i]+BETA[j])
            T3 = BETA[i]*F1v**2*BETA[j]
            E1 = BETA[i]*BETA[j]*F3v
            E2 = 0
            for k in range(C):
                E2 += aij[i, k]*kd[k, j]
            E2 *= -F5v/a
            E3 = F6v*BETA[i]*BETA[j]
            E4 = -F6v*ALPHA[i]*BETA[j]
            E5 = -F6v*BETA[i]*ALPHA[j]
            
            df1dn = df1dn.at[i,j].set(M1*(T1 + T2 + T3) + M2*(E1 + E2 + E3 + E4 + E5))
    return df1dn


# In[21]:


def df2dn_(y, dnvec, aij, a, a_barv, b, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, R, C, n_tot):
    import jax.numpy as jnp
    df2dn = jnp.zeros(C)
    M1 = R*Tc_mx/n_tot**2
    M2 = a*BETA_bar/(n_tot**2*b)*(F3v + F6v)
    M3 = -3*a/(n_tot**2*b)
    kd = jnp.eye(C)
    for i in range(C):
        T1 = 0
        for j in range(C):
            T1 += 3*kd[j, i]*dnvec[j]**2/y[j]**2
        T1 *= -1
        T2 = 3*(BETA_bar*F1v)**2
        T3 = 6*N_barv*BETA_bar*F1v**2*BETA[i]
        T4 = 6*(F1v*BETA_bar)**2*F1v*BETA[i]
        E1 = 6*BETA[i]*(2*ALPHA_bar-BETA_bar)
        E2 = 6*BETA_bar*ALPHA[i]
        E3 = -3*BETA_bar*BETA[i]
        S1 = BETA[i]*(a_barv*F6v+2*BETA_bar**2*F4v)
        S2 = 0
        for j in range(C):
            S2 += dnvec[j]*(aij[j, i] + aij[i, j])
        S2 *= BETA_bar*F6v/a
        df2dn = df2dn.at[i].set(M1*(T1 + T2 + T3 + T4) + M2*(E1 + E2 + E3) + M3*(S1 + S2))
    return df2dn


# In[22]:


#F1-F6 are EOS based factors which are f(D1, D2, K)
def dF1_(Kv, EOS, b_tot):
    return -1/(b_tot*(Kv-1)**2)

def dF3_(Kv, EOS, b_tot):
    F3 = 2/(b_tot*(D1(EOS)-D2(EOS)))
    F3 *= (D2(EOS)**2/(Kv+D2(EOS))**3-D1(EOS)**2/(Kv+D1(EOS))**3)
    return F3

def dF4_(Kv, EOS, b_tot):
    F4 = 3/(b_tot*(D1(EOS)-D2(EOS)))
    F4 *= (D2(EOS)**3/(Kv+D2(EOS))**4-D1(EOS)**3/(Kv+D1(EOS))**4)
    return F4

def dF5_(Kv, EOS, b_tot):
    import jax.numpy as jnp
    F5 = -2/(b_tot*(Kv+D1(EOS))*(Kv+D2(EOS)))
    return F5

def dF6_(Kv, EOS, b_tot):
    import jax.numpy as jnp
    F6 = 2/(D1(EOS)-D2(EOS))
    F6 *= D2(EOS)/(b_tot*(Kv+D2(EOS))**2) - D1(EOS)/(b_tot*(Kv+D1(EOS))**2) - (D2(EOS)-D1(EOS))/(b_tot*(Kv+D1(EOS))*(Kv+D2(EOS)))
    return F6


# In[23]:


def df1dv_(dnvec, aij, a, b, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, dF1, dF3, dF5, dF6, Tc_mx, R, C, n_tot):
    import jax.numpy as jnp
    df1dv = jnp.zeros(C)
    M1 = R*Tc_mx/n_tot*dF1
    M2 = a/(b*n_tot)
    for i in range(C):
        T1 = BETA[i]*N_barv
        T2 = BETA_bar*(1+2*BETA[i]*F1v)
        E1 = BETA[i]*BETA_bar*dF3
        E2 = 0
        for j in range(C):
            E2 += aij[i, j]*dnvec[j]
        E2 *= -dF5/a
        E3 = dF6*(BETA[i]*BETA_bar-ALPHA[i]*BETA_bar-ALPHA_bar*BETA[i])
        df1dv = df1dv.at[i].set(M1*(T1 + T2) + M2*(E1 + E2 + E3))
    return df1dv


# In[24]:


def df2dv_(a, a_bar, b, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, dF1, dF3, dF4, dF5, dF6, Tc_mx, R, C, n_tot):
    import jax.numpy as jnp
    df2dv = 0
    df2dv = 6*R*Tc_mx*F1v*BETA_bar**2/n_tot**3*dF1*(N_barv + BETA_bar*F1v)
    df2dv += a*BETA_bar/(b*n_tot**2)*(3*BETA_bar*(2*ALPHA_bar-BETA_bar)*(dF3+dF6)-2*BETA_bar**2*dF4-3*a_bar*dF6)
    return df2dv


# In[25]:


def daijdT_(Tc, Pc, Tc_mx, w, k, R, C, EOS):
    import jax.numpy as jnp
    daijdT = jnp.zeros([C, C])
    cv = jnp.zeros(C)
    for i in range(C):
        cv = cv.at[i].set(c(i, EOS, w))
    NU = nu(EOS)
    
    for i in range(C):
        for j in range(C):
            M = -(1-k[i, j])*R**2*Tc[i]*Tc[j]*NU/(jnp.sqrt(Pc[i]*Pc[j]))
            T1 = cv[i]/(2*jnp.sqrt(Tc_mx*Tc[i]))
            T2 = cv[j]/(2*jnp.sqrt(Tc_mx*Tc[j]))
            S1 = 1+cv[j]*(1-jnp.sqrt(Tc_mx/Tc[j]))
            S2 = 1+cv[i]*(1-jnp.sqrt(Tc_mx/Tc[i]))
            daijdT = daijdT.at[i, j].set(M*(T1*S1 + T2*S2))
    return daijdT


# In[26]:


def dadT_(y, daijdT, C):
    import jax.numpy as jnp
    dadT = 0
    for i in range(C):
        for j in range(C):
            dadT += y[i]*y[j]*daijdT[i, j]
    return dadT


# In[27]:


def dabdT_(dnvec, a_tot, dadT, aij, daijdT, C):
    import jax.numpy as jnp
    dabdT = 0
    T1 = 0
    T2 = 0
    for i in range(C):
        for j in range(C):
            T1 += dnvec[i]*dnvec[j]*aij[i, j]
            T2 += dnvec[i]*dnvec[j]*daijdT[i, j]
    dabdT = -dadT/a_tot**2*T1+T2/a_tot
    return dabdT


# In[28]:


def dALPHAdT_(y, a_totv, dadT, aij, daijdT, C):
    import jax.numpy as jnp
    dALPHAdT = jnp.zeros(C)
    T1 = jnp.dot(daijdT, y*a_totv)
    T2 = jnp.dot(aij, y*dadT)
    dALPHAdT = dALPHAdT.at[:].set((T1-T2)/a_totv**2)
    return dALPHAdT


# In[29]:


def dALPHAbdT_(dnvec, dALPHAdT, C):
    import jax.numpy as jnp
    dALPHAbdT = 0
    for i in range(C):
        dALPHAbdT += dnvec[i]*dALPHAdT[i]
    return dALPHAbdT


# In[30]:


def df1dT_(dnvec, daijdT, a_tot, dadT, b_tot, ALPHA, dALPHAdT, ALPHA_bar, dALPHAbdT, BETA, BETA_bar, N_barv, F1v, F3v, F4v, F5v, F6v, y, R, n_tot, C):
    import jax.numpy as jnp
    df1dT = jnp.zeros(C)
    M1 = R/n_tot
    M2 = 1/(b_tot*n_tot)*dadT
    M3 = -a_tot*F6v/(b_tot*n_tot)
    M4 = -F5v/(b_tot*n_tot)
    for i in range(C):
        T1 = dnvec[i]/y[i]
        T2 = F1v*(BETA[i]*N_barv + BETA_bar)
        T3 = BETA[i]*F1v**2*BETA_bar
        E1 = BETA[i]*BETA_bar*F3v
        E2 = F6v*(BETA[i]*BETA_bar-ALPHA[i]*BETA_bar-ALPHA_bar*BETA[i])
        S1 = BETA_bar*dALPHAdT[i]
        S2 = BETA[i]*dALPHAbdT
        Q = 0
        for j in range(C):
            Q += dnvec[j]*daijdT[i, j]
        df1dT =  df1dT.at[i].set(M1*(T1 + T2 + T3) + M2*(E1 + E2) + M3*(S1 + S2) + M4*Q)
    return df1dT    


# In[31]:


def df2dT_(dnvec, a_tot, dadT, dabdT, a_barv, b_tot, ALPHA, dALPHAdT, ALPHA_bar, dALPHAbdT, BETA, BETA_bar, N_barv, F1v, F3v, F4v, F5v, F6v, y, R, n_tot, C):
    import jax.numpy as jnp
    df2dT = 0
    M1 = R/n_tot**2
    M2 = BETA_bar/(b_tot*n_tot**2)*dadT
    M3 = 3*a_tot*BETA_bar/(b_tot*n_tot**2)
    T1 = 0
    for i in range(C):
        T1 -= dnvec[i]**3/y[i]**2
    T2 = 3*N_barv*(BETA_bar*F1v)**2
    T3 = 2*(BETA_bar*F1v)**3
    df2dT += M1*(T1 + T2 + T3)
    df2dT += M2*(3*BETA_bar*(2*ALPHA_bar-BETA_bar)*(F3v+F6v)-2*BETA_bar**2*F4v-3*a_barv*F6v)
    df2dT += M3*(2*BETA_bar*(F3v+F6v)*dALPHAbdT - F6v*dabdT)
    return df2dT 


# In[32]:


def JacMat(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k):
    import jax.numpy as jnp
    aij, a_totv, b_totv, N_barv, a_barv, ALPHA, ALPHA_bar,    BETA, BETA_bar, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, Vc_mx, dnvec    = CalcParams(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)
    df1dn = df1dn_(y, aij, a_totv, b_totv, ALPHA, BETA, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, R, C, n_tot)
    df2dn = df2dn_(y, dnvec, aij, a_totv, a_barv, b_totv, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, F2v, F3v, F4v, F5v, F6v, Tc_mx, R, C, n_tot)
    df3dn = 2*dnvec
    
    Kv = K(Vc_mx, y, EOS, R, Tc, Pc, C)
    dF1 = dF1_(Kv, EOS, b_totv)
    dF3 = dF3_(Kv, EOS, b_totv)
    dF4 = dF4_(Kv, EOS, b_totv)
    dF5 = dF5_(Kv, EOS, b_totv)
    dF6 = dF6_(Kv, EOS, b_totv)

    df1dv = df1dv_(dnvec, aij, a_totv, b_totv, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, dF1, dF3, dF5, dF6, Tc_mx, R, C, n_tot)
    df2dv = df2dv_(a_totv, a_barv, b_totv, ALPHA, ALPHA_bar, BETA, BETA_bar, N_barv, F1v, dF1, dF3, dF4, dF5, dF6, Tc_mx, R, C, n_tot)
    
    
    daijdT = daijdT_(Tc, Pc, Tc_mx, w, k, R, C, EOS)
    dadT = dadT_(y, daijdT, C)
    dabdT = dabdT_(dnvec, a_totv, dadT, aij, daijdT, C)
    dALPHAdT = dALPHAdT_(y, a_totv, dadT, aij, daijdT, C)
    dALPHAbdT = dALPHAbdT_(dnvec, dALPHAdT, C)
    df1dT = df1dT_(dnvec, daijdT, a_totv, dadT, b_totv, ALPHA, dALPHAdT, ALPHA_bar, dALPHAbdT, BETA, BETA_bar, N_barv, F1v, F3v, F4v, F5v, F6v, y, R, n_tot, C)
    df2dT = df2dT_(dnvec, a_totv, dadT, dabdT, a_barv, b_totv, ALPHA, dALPHAdT, ALPHA_bar, dALPHAbdT, BETA, BETA_bar, N_barv, F1v, F3v, F4v, F5v, F6v, y, R, n_tot, C)
    
    J = jnp.zeros([C+2, C+2])
    J = J.at[:-2, 2:].set(df1dn)
    J = J.at[-2, 2:].set(df2dn)
    J = J.at[-1, 2:].set(df3dn)
    J = J.at[:-2, 1].set(df1dv)
    J = J.at[-2, 1].set(df2dv)
    J = J.at[:-2, 0].set(df1dT)
    J = J.at[-2, 0].set(df2dT)
    return J


# In[33]:


def InitializeNR(y, Vc, Tc, C):
    import jax.numpy as jnp
    
    #Initialize first guess based on composition and component critical values.
    dn0 = jnp.zeros(jnp.shape(y))
    Tc0 = 0
    Vc0 = 0
    for i in range(C):
        Vc0 += y[i]*Vc[i]
        Tc0 += y[i]*Tc[i]
        dn0 = dn0.at[i].set(y[i]**(2/3))
    Tc0 = 3*Tc0
    return Tc0, Vc0, dn0


# In[34]:


def MixNewtonRaphson(MxN, n_tot, DataSet, EOS, printflag):
    from functools import partial
    import jax
    import jax.numpy as jnp
    import time
    import KeyFunctions as me
    from IPython.display import clear_output
    
    #LookUpMixture to create all nesseccary globals.
    [y, Tc, Pc, w, C, R, Vc, k, mxNames] = me.LookUpMix(MxN, DataSet, EOS, printflag)
    n = y*n_tot
    
    #Initialize first xvec.
    temptc, tempvc, tempdn = InitializeNR(y, Vc, Tc, C)
    x0vec = jnp.array([temptc, tempvc])
    x0vec = jnp.append(x0vec, tempdn)
    
    if printflag:
        print("Initial X-Vector Guess:")
        print(x0vec)
        print("---------------------------------------------")
    
    start_time = time.time()


    #Create Jacobian matrix function and initialize F matrix.
    Fmat = jnp.zeros([1, len(x0vec)])
    jitf = partial(jax.jit, static_argnames=['EOS', 'C', 'n_tot', 'R'])
    Jmat = jitf(JacMat)
    
    #Iterate with x(k+1) = x(k) + D*dx.
    xvec = x0vec
    dxvec = jnp.ones(jnp.shape(xvec))
    itr = 0
    itr_time = 0
    
    if printflag:
        print("dxVector Magnitudes:")
    while (abs(dxvec[0])>1e-4 or abs(dxvec[1])>10e-8 or any(abs(dxvec[2:])>1e-4)):
        #Display to console current run status.
        clear_output(wait = True)
        display("Current Status:")
        display("Mixture "+str(MxN)+" Iteration "+str(itr))
        display("Last Iteration Time: "+str(itr_time))
        
        #Count iterations for dampening factor.
        itr_start = time.time()
        itr += 1
        
        #Calcualte function values and Jacobian at xvec(k).
        F = GeneralizedCubicFunction(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)
        cubtime = time.time()
        Fmat = jnp.append(Fmat, F, axis = 0)
        F = -1*jnp.transpose(F)
        J = Jmat(xvec,y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)

        #Solve dxvec, uses LU decomposition with partial pivoting.
        dxvec = jnp.linalg.solve(J, F)
        dxvec = jnp.transpose(dxvec)
        dxvec = jnp.reshape(dxvec, [jnp.shape(J)[0]])
        
        #Define Q, the dampening factor. 0 for binary mixtures, and 518 for ~13 iterations for non-binary.
        if C ==2:
            Q = 0
        else:
            Q =700
        #Apply dampening.
        D = 1/(1+Q*jnp.exp(-0.5*itr))
        xvec = xvec + D*dxvec

        #Max iterations is set as 30.
        if itr>30:
            print("Convergence not achieved in 30 iterations.")
            Pc = None
            return Fmat, xvec, Pc
        
        #Calculate iteration time for full report.
        itr_end = time.time()
        itr_time = itr_end-itr_start
        print(str(itr)+"    "+str(round(jnp.linalg.norm(dxvec), 4))+"    "+str(round((itr_time), 4)))
        
    #Calculate Mixture runtime for summary report.    
    end_time = time.time()
    rn_time = end_time-start_time
    
    #Calculate final function values for evaltuation of convergence.
    F = GeneralizedCubicFunction(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)      
    Fmat = jnp.append(Fmat[1:, :], F, axis = 0)
    
    #Check convergence and calculate Pc.
    conv_flag = jnp.linalg.norm(F-jnp.zeros(jnp.shape(F)))
    if conv_flag >= 1:
        print("Convergence achieved, function values high. Critical point may be false or nonexistant.")
    else:
        if printflag:
            print("Magnitude of final function vector: " + str(round(conv_flag, 4)))
    if printflag:
        print("---------------------------------------------")
    aij = aijf(xvec[0], Tc, R, EOS, Pc, w, k, C)
    Pc = R*xvec[0]/(xvec[1]-b_tot(y, EOS, R, Tc, Pc, C)) - a_tot(n, n_tot, aij, C)/((xvec[1]+D1(EOS)*b_tot(y, EOS, R, Tc, Pc, C))*(xvec[1]+D2(EOS)*b_tot(y, EOS, R, Tc, Pc, C)))
    
    return Fmat, xvec, Pc, itr, rn_time


# In[35]:


def CompNewtonRaphson(y, Tc, Pc, w, C, R, Vc, k, n_tot, EOS, ini = None, printflag = 0):
    from functools import partial
    import jax
    import jax.numpy as jnp
    import time
    import KeyFunctions as me
    from IPython.display import clear_output
    
    n = y*n_tot
    
    #Initialize first xvec.
    temptc, tempvc, tempdn = InitializeNR(y, Vc, Tc, C)
    if ini is not None:
        if len(ini) == 2:
            x0vec = jnp.array(ini)
            x0vec = jnp.append(x0vec, tempdn)
        else:
            x0vec = jnp.array(ini)
    else:
        x0vec = jnp.array([temptc, tempvc])
        x0vec = jnp.append(x0vec, tempdn)
    
    if printflag:
        print("Initial X-Vector Guess:")
        print(x0vec)
        print("---------------------------------------------")
    
    start_time = time.time()


    #Create Jacobian matrix function and initialize F matrix.
    Fmat = jnp.zeros([1, len(x0vec)])
    jitf = partial(jax.jit, static_argnames=['EOS', 'C', 'n_tot', 'R'])
    Jmat = jitf(JacMat)
    
    #Iterate with x(k+1) = x(k) + D*dx.
    xvec = x0vec
    dxvec = jnp.ones(jnp.shape(xvec))
    itr = 0
    itr_time = 0
    
    if printflag:
        print("dxVector Magnitudes:")
    while (abs(dxvec[0])>1e-4 or abs(dxvec[1])>10e-8 or any(abs(dxvec[2:])>1e-4)):   
        #Count iterations for dampening factor.
        itr_start = time.time()
        itr += 1
        
        #Calcualte function values and Jacobian at xvec(k).
        F = GeneralizedCubicFunction(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)
        cubtime = time.time()
        Fmat = jnp.append(Fmat, F, axis = 0)
        F = -1*jnp.transpose(F)
        J = Jmat(xvec,y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)

        #Solve dxvec, uses LU decomposition with partial pivoting.
        dxvec = jnp.linalg.solve(J, F)
        dxvec = jnp.transpose(dxvec)
        dxvec = jnp.reshape(dxvec, [jnp.shape(J)[0]])
        
        #Define Q, the dampening factor. 0 for binary mixtures, and 518 for ~13 iterations for non-binary.
        if ini is not None:
            Q = 100
        else:
            Q = 518
        #Apply dampening.
        D = 1/(1+Q*jnp.exp(-0.5*itr))
        xvec = xvec + D*dxvec

        #Max iterations is set as 30.
        if itr>50:
            print("Convergence not achieved in 30 iterations.")
            Pc = None
            return Fmat, xvec, Pc
        
        #Calculate iteration time for full report.
        itr_end = time.time()
        itr_time = itr_end-itr_start
        if printflag:
            print(str(itr)+"    "+str(round(jnp.linalg.norm(dxvec), 4))+"    "+str(round((itr_time), 4)))
        
    #Calculate Mixture runtime for summary report.    
    end_time = time.time()
    rn_time = end_time-start_time
    
    #Calculate final function values for evaltuation of convergence.
    F = GeneralizedCubicFunction(xvec, y, Tc, Pc, w, EOS, C, n, n_tot, R, Vc, k)      
    Fmat = jnp.append(Fmat[1:, :], F, axis = 0)
    
    #Check convergence and calculate Pc.
    conv_flag = jnp.linalg.norm(F-jnp.zeros(jnp.shape(F)))
    if conv_flag >= 1:
        print("Convergence achieved, function values high. Critical point may be false or nonexistant.")
    else:
        if printflag:
            print("Magnitude of final function vector: " + str(round(conv_flag, 4)))
    if printflag:
        print("---------------------------------------------")
    aij = aijf(xvec[0], Tc, R, EOS, Pc, w, k, C)
    Pc = R*xvec[0]/(xvec[1]-b_tot(y, EOS, R, Tc, Pc, C)) - a_tot(n, n_tot, aij, C)/((xvec[1]+D1(EOS)*b_tot(y, EOS, R, Tc, Pc, C))*(xvec[1]+D2(EOS)*b_tot(y, EOS, R, Tc, Pc, C)))
    
    return Fmat, xvec, Pc, itr, rn_time




