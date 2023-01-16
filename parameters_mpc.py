def ParametersMPC(Ad,Bd,Cd,Dd,n,p,q,N,M,py,pu):

    ## add integral action
    Aksi = np.block([[Ad,np.zeros((n,q))],[Cd@Ad,np.identity(q)]])
    Bksi = np.block([[Bd],[Cd@Bd]])
    Cksi = np.block([np.zeros((q,n)),np.identity(q)]) 

    ## G (matriz de dinamica)
    Gaux = Cksi@Bksi
    for i in range(1,N):
        Gaux = np.block([[Gaux],[Cksi@matrix_power(Aksi,i)@Bksi]])
    # end for
    G = Gaux
    if M > 1:
        for i in range(1,M):
            G = np.block([G,np.block([[np.zeros((q*i,p))],[Gaux[:((N*q)-(q*i)),:]]])])
        # end for
    # end if

    # phi
    phi = Cksi@Aksi
    for i in range(2,N+1):
        phi = np.block([[phi],[Cksi@matrix_power(Aksi,i)]])
    # end for

    ## Py (output weight)
    Py = py*np.identity(q*N)

    ## Pu (control weight, != 0)
    Pu = pu*np.identity(p*M)

    ## Kmpc
    kaux = np.linalg.inv((G.transpose()@Py@G) + Pu)@G.transpose()@Py
    kmpc = kaux[:p,:]

    return G,phi,Py,Pu,kmpc
  
  def QuadProg(n,p,q,N,M,G,Py,Pu):
    
    ## Gn
    Gn = Py@G

    ## Hqp
    Hqp = 2*((G.transpose()@Py@G) + Pu)

    ## I
    IpM = np.identity(p*M)

    ## T
    Ip1 = np.matlib.repmat(np.identity(p),M,1)
    TM = Ip1
    if M > 1: # TMp
        for i in range(1,M):
            TM = np.block([TM,np.block([[np.zeros((p*i,p))],[Ip1[:((M*p)-(p*i)),:]]])])
        # end for
    # end if

    ## Aqp = S
    Aqp = np.vstack((IpM,-IpM,TM,-TM,G,-G))

    return Gn,Hqp,Aqp
