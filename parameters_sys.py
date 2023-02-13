def SysContinuous(Ac,Bc,Cc,Dc,plot):

    ## continuous-time linear state-space representation
    sysc = ss(Ac,Bc,Cc,Dc) 

    ## dimensions
    n = len(Ac) # number of states (=A rows)
    p = len(Bc[0]) # number of control actions (=B columns)
    q = len(Cc) # number of outputs (=C rows)

    ## step response
    if plot == 1:
        # vetor de tempo 
        tvec = np.arange(0,50,1)

        # data
        tout0, yout0 = step_response(sysc,T=tvec,input=0) # data 1
        tout1, yout1 = step_response(sysc,T=tvec,input=1) # data 2

        # plot 
        figure = plt.figure()
        # plt.suptitle('Step response')

        # subplot 1
        plt.subplot(221)
        plt.plot(tout0, yout0[0])
        plt.title('From: Main rotor voltage')
        plt.ylabel('To: Pitch angle')
        plt.ylim([0.0,0.1])
        plt.grid(True)

        # subplot 2
        plt.subplot(223)
        plt.plot(tout0, yout0[1])
        #plt.title('From: Main rotor voltage')
        plt.ylabel('To: Yaw angle')
        plt.ylim([-7,7])
        plt.grid(True)

        # subplot 3
        plt.subplot(222)
        plt.plot(tout1, yout1[0], 'tab:orange')
        plt.title('From: Tail rotor voltage')
        #plt.ylabel('To: Pitch angle')
        plt.ylim([0.0,0.1])
        plt.grid(True)

        # subplot 4
        plt.subplot(224)
        plt.plot(tout1, yout1[1], 'tab:orange')
        #plt.title('From: Tail rotor voltage')  
        #plt.ylabel('To: Yaw angle')
        plt.ylim([-7,7])
        plt.grid(True)

        # extra label
        figure.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad) \n \n")

        # show
        plt.show() 

    # end if

    return sysc,n,p,q
    
   def SysDiscrete(sysc,Ts,fsim,tsim):

    ## emulation time
    Te = Ts/fsim

    ## discrete-time state-space model 
    sysd = sample_system(sysc,Ts,method='zoh')
    Ad = sysd.A
    Bd = sysd.B
    Cd = sysd.C
    Dd = sysd.D

    ## discrete-time state-space model for emulation
    syse = sample_system(sysc,Te,method='zoh')
    Ae = syse.A
    Be = syse.B
    Ce = syse.C
    De = syse.D

    return Ad,Bd,Cd,Dd,Ae,Be,Ce,De,Te 
