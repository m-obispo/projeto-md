import numpy as np
import matplotlib.pyplot as plt
import time
import numexpr as ne

nDim = 2
N = 750
tFinal = 750
dt = 0.01
g = 0.
gama = 0.01
alfa = 0

np.seterr(invalid='ignore')

#Dimensoes do espaco
L = np.array((50,50))

#Coordenadas das particulas
r = np.zeros((N,nDim))
v = np.random.normal(0,2,(N,nDim))

#Arrays com as posicoes relativas das particulas
R  = np.zeros((N,N,nDim))
Rm2 = np.zeros((N,N))
Rm6 = np.zeros((N,N))
Rm12 = np.zeros((N,N))

def posicionaParticulas(disposicao='aleatoria',
        param=(L/5,L/5+1.12*np.sqrt(N))):
    '''
        Posiciona as particulas em uma disposição aleatória ou cubica.

        Parametros:
        -----------
        param: coordenadas da diagonal da caixa.
    '''
    if disposicao=='aleatoria' :
        for dim in range(nDim) :
            r[:,dim] = param[0][dim] + np.random.rand(N)*(param[1][dim]-param[0][dim])
    else :
        area=np.prod(param[1]-param[0])
        distancia=np.sqrt(area/N)
        Nx = int((param[1]-param[0])[0]/distancia)
        print(area,distancia,Nx)
        n=np.arange(N)
        r[:,0] = distancia*(n%Nx) + param[0][0]
        r[:,1] = distancia*(n//Nx) + param[0][1]

def calcPotencias() :
    global R, Rm2, Rm6, Rm12
    for dim in range(nDim) :
        rdim  = r[:,[dim]]
        rdimT = rdim.T
        #R[:,:,dim] = rdim-rdimT
        ne.evaluate('rdim-rdimT',out=R[:,:,dim])
    #R2 = (R*R).sum(axis=2)
    ne.evaluate('sum(R*R,axis=2)',out=Rm2)
    Rm2 = 1/Rm2
    Rm2[np.isinf(Rm2)] = 0
    #with np.errstate(divide='ignore') :
    #    Rm6 = R2**(-3)
    ne.evaluate('Rm2**3',out=Rm6)
    Rm12=Rm6*Rm6

def energiaPotencial() :
    energia = 4*(Rm12-Rm6).sum()/2
    energia = ne.evaluate('sum(Rm12-Rm6)')
    energia *= 2
    return energia

f0=np.zeros((N,N))
f1 =np.zeros((N,nDim))

def forca() :
    global f1
    f0 = 48*((Rm12-Rm6/2)*Rm2)
    for dim in range(nDim) :
        f1[:,dim] = (R[:,:,dim]*f0).sum(axis=1)
        f1[:,dim] += 48/r[:,dim]**13 + 48/(r[:,dim]-L[dim])**13
    f1[:,1] += -g
    f1 += -gama*v #força dissipativa, responsável pelo resfriamento controlado do sistema
    return f1

posicionaParticulas('cubica')

fig, [graf1, graf2, graf3] = plt.subplots(1,3)
graf1.plot([0,L[0],L[0],0,0],[0,0,L[1],L[1],0],'k')
graf4 = graf3.twinx()
#plotR = plt.plot(r[:,0],r[:,1],'o')
plotR = graf1.scatter(r[:,0],r[:,1])
#plotR = graf1.scatter(r[:,0],r[:,1], c=N, cmap=plt.get_cmap('nipy_spectral'),s=(100/L)**2)
plt.gca().set_aspect('equal', 'datalim')

t0 = time.time()
ePot = energiaPotencial()

#Primeiro passo leapfrog
calcPotencias()
f=forca()
#print(f)
v += f*dt/2

vec_t = np.arange(0,tFinal,dt)

vec_KE = np.zeros(len(vec_t))
vec_virial = np.zeros(len(vec_t))
vec_Eint = np.zeros(len(vec_t))
vec_Temp = np.zeros(len(vec_t))
vec_P = np.zeros(len(vec_t))

#alfadt = alfa*np.sqrt(dt)

i=0
for t in vec_t:
    r += v*dt
    calcPotencias()
    f=forca()
    v += f*dt
    if i%10==0:
        if i%100:
            #Cálculos de Temperatura e Pressão
            vec_KE[i] = (v**2).mean(axis=0).sum()/2 #Calcula a energia cinética média das partículas
            vec_virial[i] = (r*f).sum()/N           #Calcula a média das forças
            #eta = np.random.normal(0,0.5,(N,nDim))
            #v += alfadt*eta
            energiaTotal = energiaPotencial()+(v**2).sum()/2
            vec_Eint[i] = energiaTotal
            media_KE = vec_KE[i-5:i].sum()          #Calcula a energia cinética média no tempo
            media_w = vec_virial[i-5:i].sum()       #Calcula o virial
            vec_Temp[i] = 2*media_KE/3
            V = np.product(L)
            vec_P[i] = (2*N*media_KE + media_w)/3/V
            print(t,energiaTotal,'\nT =', vec_Temp[i],'\nP = ',vec_P[i],'\n')#[-5:].sum()

        # #Gráfico PxT
        # graf1.set_title('t=%.1f'%t)
        # graf2.clear()
        # graf2.set_ylim(0,vec_P.max())
        # graf2.set_xlim(0,vec_Temp.max())
        # graf2.set_xlabel('Temperatura')
        # graf2.set_ylabel('Pressão')
        # graf2.plot(vec_Temp[vec_Temp!=0], vec_P[vec_Temp!=0],'r',label='Pressão')
        # graf2.legend(loc='lower right')
        # #Gráfico ExT
        # graf3.clear()
        # graf3.set_xlim(0,vec_Temp.max())
        # graf3.set_ylim(vec_Eint.min(),vec_Eint.max())
        # graf3.set_xlabel('Temperatura')
        # graf3.set_ylabel('Energia Interna')
        # graf3.plot(vec_Temp[vec_Temp!=0], vec_Eint[vec_Temp!=0],'b',label='Energia Interna')
        # graf3.legend(loc='lower right')

        plotR.set_offsets(r)
        plt.pause(0.0000001)
    else:
        vec_KE[i] = (v**2).mean(axis=0).sum()/2 #Calcula a energia cinética média das partículas
        vec_virial[i] = (r*f).sum()/N           #Calcula a média das forças
        energiaTotal = energiaPotencial()+(v**2).sum()/2
        vec_Eint[i] = energiaTotal
    i+=1
print("Tempo de execucao:",time.time()-t0)

plt.show()
