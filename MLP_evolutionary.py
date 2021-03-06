import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import time
from copy import  deepcopy
import pickle
# learning rate epsilon set globally:
epsilon = .3
ICHECK=False


# activation function:
sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid=lambda x: np.exp(x)/( (np.exp(x) + 1)**2)

def load_MLP_from_file(filename):
    f = open(filename,"r")
    nnmpl = pickle.load(f)       
    
    f.close()
    return nnmpl
class MLP(object):
    """Neural Network with 1 hidden layer,
    Trained with backpropagation"""
    def __init__(self, nx, nh, ny=1, Wmin=-15, Wmax=15, noinit = False):
        super(MLP, self).__init__()
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.Wmin = Wmin
        self.Wmax = Wmax
        
        # do random weight initialization
        if not noinit:
            self.init_weights()

        return
    
    def save_to_file(self, filename):
        g = open(filename,"w")
        pickle.dump(self,g)
        
        g.close
        return
    
    def copy(self,weightless = True):
        '''If weightless==True, the Weigth matrices Whx And Wyh are not assigned
            They can be initialized to random by calling MLP.init_weights()
            if weightless==False, these are copied as well
        '''
        if weightless:
            return MLP(self.nx,self.nh,self.ny,self.Wmin,self.Wmax,noinit=True)
        else:
            mlp = MLP(self.nx,self.nh,self.ny,self.Wmin,self.Wmax,noinit=True)
            #mlp.Whx = self.Whx[:]
            #mlp.Wyh = self.Wyh[:]
            mlp.Whx = deepcopy(self.Whx)
            mlp.Wyh = deepcopy(self.Wyh)
            return mlp
    
    def init_weights(self):
        """
        Init weight matrices
        """
        nx, ny, nh = self.nx, self.ny, self.nh
        #### weight matrices; unirand init::
        Whx = np.random.uniform(self.Wmin,self.Wmax, (nh, nx+1) ) # +1 for the bias input
        Wyh = np.random.uniform(self.Wmin,self.Wmax, (ny, nh+1) ) 
        
        self.Whx = Whx
        self.Wyh = Wyh 
        return


    def ffwd(self, x):
        """Sweep through the network of sigmoidal units; 
        store all activations and outputs and return y (output)
        """

        Whx, Wyh = self.Whx, self.Wyh

        # process x+bias input -> h
        x = np.r_[x, 1]
        ha = Whx.dot(x)
        h = sigmoid(ha)

        # process h+bias input -> y
        h = np.r_[h, 1]
        #ya = np.dot(Wyh, h)
        ya = Wyh.dot(h)
        y = sigmoid(ya)

        # this next line is short for: self.x=x; self.h=h, ....
        #self.__dict__.update( dict(x=x, ha=ha,h=h,ya=ya,y=y ) )
        # return output:
        return y

    def backpropagation(self, t):
        '''print self.Wyh
        print self.Wyh.T
        print self.Whx
        print self.y
        print self.x
        input('waiting')'''
        # store the feedback weights:
        Why = self.Wyh.T[:-1, :] # ignoring the weight attached to the bias..

        # compute error 'vector':
        e = self.y - t
        # the weight update for the output weights is as normal:
        # inp * error * dsigmoid * learning rate
        # here, I'm using some tricks to skip a few of these steps:
        h = self.h[:, np.newaxis]
        e = e[np.newaxis, :]
        '''print'h=',h
        print'e=',e
        print 'sigmeps', dsigmoid(self.ya) * epsilon
        print 'edot',e.dot(h.T)
        time.sleep(5)
        '''
        # the actual weight update
        delta_y = (dsigmoid(self.ya) * e).T
        self.Wyh -= epsilon * delta_y * h.T

        # now, propagate the error back, using Why:
        ha = self.ha[:,np.newaxis]
        x =   self.x[:, np.newaxis]
        delta_h = dsigmoid(ha) * Why.dot(delta_y) 
        self.Whx  -= epsilon * delta_h * x.T
        return
    '''
    original
        # the actual weight update
        self.Wyh -= e.dot(h.T) * dsigmoid(self.ya) * epsilon

        # now, propagate the error back, using Why:
        ha = self.ha[:, np.newaxis]
        delta_bp = Why.dot(e) * dsigmoid(ha)

        # use this new delta to update hidden weights:

        x = self.x[:, np.newaxis]
        self.Whx -= ( delta_bp.dot(x.T) ) * epsilon
        return'''
    
    '''
    from demo
         
        # the actual weight update
        delta_y = (dsigmoid(self.ya) * e).T
        self.Wyh -= epsilon * delta_y * h.T

        # now, propagate the error back, using Why:
        ha = self.ha[:,np.newaxis]
        x =   self.x[:, np.newaxis]
        delta_h = dsigmoid(ha) * Why.dot(delta_y) 
        self.Whx  -= epsilon * delta_h * x.T
        return
'''

#obviously only works for arrays of the (key,value) format


class Abathur:
    def __init__(self,(nx,nh,ny),dataset,Wmin=-1, Wmax=1, poolsize=100, survivors=80, \
                 leftover_parents=5,individual_mutation_chance=.2, \
                 gene_mutation_chance=.1, mutation_swing = 1, \
                 crossover_chance = .5, discrete=False ):
        #the pool should always be kept in an ordered state
        
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.dataset = dataset
        self.generation = 0
        self.poolsize = poolsize
        self.survivors = survivors
        self.leftover_parents = leftover_parents
        self.indmut = individual_mutation_chance
        self.genmut = gene_mutation_chance
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.Wdiff = Wmax-Wmin
        self.mutswing = mutation_swing
        self.discrete = discrete
        self.crosschance = crossover_chance
        
        self.pool = []
        for i in range(0,poolsize):
            self.insort(self.pool,MLP(nx,nh,ny,Wmin=self.Wmin,Wmax=self.Wmax))
            
        print "poolsize=",len(self.pool)
        return
    
    def insort(self,array, specimen):
        i = 0;
        key = self.fitness(specimen,self.discrete)
        value = specimen

        if len(array) == 0:
            array.insert(0,(key,value))
            return
    
        if key<=array[-1][0]:
            array.insert(len(array),(key,value))
            return
        
        for element in array:
            if key>element[0]:
                array.insert(i,(key,value))
                break
            i = i+1
                
        return 
    
    def fitness(self,specimen,discrete):
        E=0
        for x,t in self.dataset:
            y = specimen.ffwd(x)
            if discrete==1:
                E += np.abs(t-np.floor(y+.5))
            elif discrete==2:
                E += np.abs(t-np.floor(y+.5)) + (t-y) ** 2
            else:
                E += (t-y) ** 2
            
            #print t,y,np.abs(t-int(y+.5))
        #to make it 'the higher the better'
        E = np.sum(E)
        return -E#/len(self.dataset)
    
    '''def mate(self,parent1,parent2):
        child1 = MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax,noinit=True)
        child2 = MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax,noinit=True)
        avgWhx = (parent1.Whx + parent2.Whx)/2
        avgWyh = (parent1.Wyh + parent2.Wyh)/2
        deltaWhx = parent1.Whx - parent2.Whx
        deltaWyh = parent1.Wyh - parent2.Wyh
        
        child1.Whx = avgWhx - deltaWhx*.16
        child2.Whx = avgWhx + deltaWhx*.16
        
        child1.Wyh = avgWyh - deltaWyh*.16
        child2.Wyh = avgWyh + deltaWyh*.16
        
        return child1,child2'''
    
    def mate(self, *parents):
        '''
        Currently only works for 2 parents
        accepts any number of parents, but the basic structure of the first
        will be copied (nx, nh, etc.) and is expected to be identical to Abathur specs
        '''
        # goes through all the parends and does a 'rotation' with the genes

        #afterwards, it is assumed that there are at least 2 parents
        if len(parents) == 1:
            return parents[0].copy()
        
        parentcount = len(parents)
        
        avgWhx = np.zeros((self.nh, self.nx+1))
        avgWyh = np.zeros((self.ny, self.nh+1))
        for parent in parents:
            avgWhx += parent.Whx
            avgWyh += parent.Wyh
        avgWhx /= parentcount
        avgWyh /= parentcount
    
        
        children = [parents[0].copy(weightless=True)] * (parentcount+1)
       
        children[0].Whx = deepcopy(avgWhx)
        children[0].Wyh = deepcopy(avgWyh)
       
        #parent1 gene and parent 2 gene
        for i,(p1g,p2g) in enumerate(zip(parents[0].Whx.flat,parents[1].Whx.flat)):
            if np.random.random()<=self.crosschance:
                children[1].Whx.flat[i] = p1g
                children[2].Whx.flat[i] = p2g
            else:
                children[2].Whx.flat[i] = p1g
                children[1].Whx.flat[i] = p2g
        
        for i,(p1g,p2g) in enumerate(zip(parents[0].Wyh.flat,parents[1].Wyh.flat)):
            if np.random.random()<=self.crosschance:
                children[1].Wyh.flat[i] = p1g
                children[2].Wyh.flat[i] = p2g
            else:
                children[2].Wyh.flat[i] = p1g
                children[1].Wyh.flat[i] = p2g
                
        '''for i,pgs in enumerate([parent.Wyh.flat for parent in parents]):
            if np.random.random()<=self.crosschance:
                for j in range(1,parentcount+1):
                    children[j].Wyh.flat[i] = pgs[j] 
            else:
                children[2].Wyh.flat[i] = p1g
                children[1].Wyh.flat[i] = p2g
        '''
        return children 
        
    #TODO - verify if mutations are done correctly
    #TODO - increase efficiency based on mutation chance
    def mutate(self,pool):
        #i =0 
        
        
        if ICHECK: print "Total Errors - Starting mutation:",self.check_integrity(pool) 
        newpool = []
        for indituple in pool:
            #TODO it is probably not necessary to use copy
            individual = indituple[1].copy(weightless = False)
            '''for i in range(0,len(individual.Whx)):
                    for j in range(0,len(individual.Whx[0])):
                            individual.Whx.flat[i] += 1'''
            if np.random.random()<=self.indmut:
                #need to modify the individual weights, which are floats,
                #therfore iteration over the space will not work    
                
                for i in range(0,len(individual.Whx.flat)):
                    if np.random.random()<=self.genmut:
                         individual.Whx.flat[i] += self.Wdiff * self.mutswing * \
                                                 (np.random.random() - .5)
                for i in range(0,len(individual.Wyh.flat)):
                    if np.random.random()<=self.genmut:
                         individual.Wyh.flat[i] += self.Wdiff * self.mutswing * \
                                                 (np.random.random() - .5)
                
            self.insort(newpool,individual)
            
        if ICHECK: print "Total Errors - Returning:",self.check_integrity(newpool) 
            #i+=1
        return newpool
    
    def evolve(self):
        if ICHECK: print "Total Errors - evolve start:",self.check_integrity(self.pool) 
        self.generation += 1
        
        #the chance of getting a good mutation on the best specimen is lower than
        # getting one in one of the lower quality specimens, therefore the first
        # one is skipped TODO - maybe implement a forced mutation for  clone of 
        # the first?
        # Just adding mutated clones of the first as random samples might lead
        # to evoutionary dead-ends
        prime_individual = self.pool[0][1]
        self.pool = self.mutate(self.pool[1:])
        self.insort(self.pool,prime_individual)
        
        #this line should always be first -- the pool is assured to be sorted
        newpool = self.pool[:self.leftover_parents]
        #TODO children produced is n+1 not 3/2*n
        #for i in range(0,self.survivors-self.leftover_parents-1,2):
        #    children = self.mate(self.pool[i][1],self.pool[i+1][1])
        #produce enough children to meet the quota
        
        '''does not work for negatives!!!
        #totalscore = 0
        #for individual in self.pool:
        #    totalscore += self.pool[0]'''
        
        # the chance of surviving is defined by survivors/total/2/nr of children per mating
        # for the first individual. Then tapers off by geometrical order (/2)
        #this ensures the end result will tend to the requested survivor count
        # the rest is filled with random individuals
        # for efficiency considerations, if the chance drops below .03% it should stop
        
        #children/mateing
        cpm = 3
        nr_matings_req= 1.0*self.survivors/3.0/cpm
        repr_chance = .5
        
        
        if ICHECK: print "Total Errors - after mutation:",self.check_integrity(self.pool) 
        #Select mating pairs
        for i in range (0,self.poolsize):
            nr_matings = 0
            for j in range (i,self.poolsize):
                if (np.random.random()<repr_chance):
                    nr_matings += 1
                    children = self.mate(self.pool[i][1], self.pool[j][1])
                    #print "mating ",i,"with",j
                    for child in children:
                        self.insort(newpool,child)
                    if nr_matings>=nr_matings_req:
                        #print "children produced:",nr_matings * cpm
                        #print "too many children"
                        break;
                    
                    if len(newpool)>self.survivors:
                        #print "too many surivors"
                        break
            nr_matings_req = nr_matings_req/2 - 1 
            repr_chance /= 2
            
            if len(newpool)>self.survivors:
                break
            
        if ICHECK: print "Total Errors - mating halfway:",self.check_integrity(self.pool) 

        self.pool = newpool
        if len(self.pool) > self.poolsize:
            self.pool = newpool[:self.poolsize]
        else:
            for i in range(len(self.pool),self.poolsize):
                self.insort(self.pool,MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax))
   
        if ICHECK: print "Total Errors - evolve end:",self.check_integrity(self.pool) 
        return
    #returns the best performing individual
    def prime_specimen(self):
        return self.pool[0][1]
    
    def disp_best_specimens(self,n=5):
        print "Generation:",self.generation
        print     "Place -   Score"
        for i in range(0,min(n,len(self.pool))):
            if self.fitness(self.pool[i][1],self.discrete)!=self.pool[i][0]:
                print "--------\nCORRUPTION DETECTED\n--------"
            print "%d.   -   %f"%(i,-self.pool[i][0])
            
        if ICHECK :print "Total Errors - printing prime specimens:",self.check_integrity(self.pool) 

    def check_integrity(self,pool):
        s = 0
        for specimen in pool:
            if self.fitness(specimen[1],self.discrete) != specimen[0]:
                s+=1
        return s
    
def train(net, dataset, plot_learning=False):
    dset = deepcopy(dataset)
    epoch = 0

    error_list = []

    while True:

        np.random.shuffle(dset)
        E = 0
        # now, loop over all exampes:
        for x, t in dset:
            # compute the output
            y = net.ffwd(x)
            # tell it what the target was:
            net.backpropagation(t)
            E += 0.5 * (y - t)**2

        epoch += 1
        # keep track of the 'average error'
        E = E / len(dset)
        error_list.append(  E  )

        # check whether we should stop learning:
        if epoch%1000 == 0:
            print epoch,"epoch completed, current error:", E
            print E<1.5e-4, (E < 1.5e-4).all()
        if (E < 1e-3).all():
            break
        if epoch > 5000:
            print E, E.all()
            net.init_weights()
            plt.plot(error_list)
            plt.show()
            print 'Stuck in local minimum. Restarting...'
            epoch = 0
            #error_list = []

    if plot_learning:
        plt.plot(error_list)
        plt.show()
    return



def learn_xor():
    # np.random.seed(22)
    # make a network:
    net = MLP(nx=2, nh=2, ny=1)

    dataset = [ 
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 1], 0),]

    train(net, dataset, plot_learning = True)

    for x,t in dataset:
        print x, net.ffwd(x) 
    return

def get_dataset(targets_string, is_multidim=False):
    # load the alphabet:
    braille_alphabet = pd.read_csv('braille_binary.csv')

    # extract all datacolumns:
    symbols = braille_alphabet[['ASCII Glyph']].values.flatten()
    braille = braille_alphabet[['bitstring']].values.flatten()
    braille = [ ast.literal_eval(x) for x in braille]

    #this is easier than checking for dimensions, also less prone to user error
    if (not is_multidim):
        targets = list(targets_string)
        target_values = [(1 if s in targets else 0) for s in symbols  ]
    else:
        #wraps the 
        target_values = [([(1 if s in list(ts) else 0) for ts in targets_string]) for s in symbols]
    # after the next line: 'dataset' will be a list of 
    # x,t tuples, which can be used by the function 'train()'
    return symbols, zip(braille, target_values)

def learn_vowels():
    #get dataset
    symbols, dataset = get_dataset('AEIOU')
    
    # now, let's make/train a perceptron:
    net = MLP(nx=6, nh=15, ny=1)
    train(net, dataset, plot_learning=False)

    # what is the output for the entire dataset?
    for s, (x, t) in zip(symbols, dataset):
        y = net.ffwd(x)
        #enhanced the print statement a bit -- shows if  a value is incorrectly predicted
        print "output for {}: {} => {}  {}".format(s, np.round(y, 3), int(y > 0.5), \
                          ' -- FAILED' if not t==int(y>0.5) else ' ')

    return

def learn_all():
    symbols, dataset = get_dataset(['AEIOU','1234567890','!"#$%&()*+=,-./ '], True)
    net = MLP(nx=6, nh=20, ny=3)
    
    train(net, dataset, plot_learning=False)        
                                
    for s, (x, t) in zip(symbols, dataset):
        y = net.ffwd(x)
        incorrect = [not (int(y[i]>0.5)==t[i]) for i in range(0,3)]
        incorrect = np.any(incorrect)
        print "output for {}: {} => {} {} {} {}".format(s, np.round(y, 3), \
                          'Vowel' if  y[0]>0.5 else ' ', \
                          'Number' if y[1]>0.5 else ' ',\
                          'Punctuation' if y[2]>0.5 else ' ', \
                          ' -- FAILED' if incorrect else ' ')
    return
    

def learn_vowels_by_evolution():
    #get dataset
    symbols, dataset = get_dataset('AEIOU')
    
    # now, let's make/train a perceptron:
    abathur = Abathur((6,10,1),dataset, poolsize=1500, survivors = 1100,\
                      leftover_parents=40,Wmin=-15,Wmax=15,\
                      individual_mutation_chance=.3,gene_mutation_chance=.2,\
                      discrete = 2)
    
    best_specimens=[]
    for i in range(0,100):
        abathur.disp_best_specimens()
        best_specimens.append(abathur.prime_specimen())
        print abathur.fitness(best_specimens[i],discrete=True)
        print abathur.prime_specimen().Whx
        print abathur.prime_specimen().Wyh
        abathur.evolve()
    
    abathur.disp_best_specimens()
    net = abathur.prime_specimen()
    best_specimens.append(net)
    print net.Whx
    print net.Wyh
    print abathur.fitness(abathur.prime_specimen(),discrete=True)


    # what is the output for the entire dataset?
    for s, (x, t) in zip(symbols, dataset):
        y = net.ffwd(x)
        #enhanced the print statement a bit -- shows if  a value is incorrectly predicted
        print "output for {}: {} => {}  {}".format(s, np.round(y, 3), int(y > 0.5), \
                          ' -- FAILED' if not t==int(y>0.5) else ' ')
    
    return best_specimens

def learn_all_by_evolution():
    #get dataset
    symbols, dataset =  get_dataset(['AEIOU','1234567890','!"#$%&()*+=,-./ '], True)
    
    # now, let's make/train a perceptron:
    abathur = Abathur((6,10,3),dataset, poolsize=1500, survivors = 1100,\
                      leftover_parents=30,Wmin=-15,Wmax=15,\
                      individual_mutation_chance=.5,gene_mutation_chance=.4,\
                      discrete= 0 )
    
    best_specimens = []
    for i in range(0,100):
        abathur.disp_best_specimens()
        best_specimens.append(abathur.prime_specimen())
        print abathur.fitness(best_specimens[i],discrete=True)
        print abathur.prime_specimen().Whx
        print abathur.prime_specimen().Wyh
        abathur.evolve()
    
    abathur.disp_best_specimens()
    print abathur.fitness(abathur.prime_specimen(),discrete=True)
    net = abathur.prime_specimen()
    best_specimens.append(net)
    print net.Whx
    print net.Wyh
    net.save_to_file("all100gencont.txt" )
    
    # what is the output for the entire dataset?
    for s, (x, t) in zip(symbols, dataset):
        y = net.ffwd(x)
        incorrect = [not (int(y[i]>0.5)==t[i]) for i in range(0,3)]
        incorrect = np.any(incorrect)
        print "output for {}: {} => {} {} {} {}".format(s, np.round(y, 3), \
                          'Vowel' if  y[0]>0.5 else ' ', \
                          'Number' if y[1]>0.5 else ' ',\
                          'Punctuation' if y[2]>0.5 else ' ', \
                          ' -- FAILED' if incorrect else ' ')
    return best_specimens

if __name__ == '__main__':
    #learn_xor()
    #learn_vowels()
    #learn_all()
    
    #TODO - there is a bug in the metric
    #bs = learn_vowels_by_evolution()
    bs = learn_all_by_evolution()
    
    
# pairing mating:
    # vowels: 100gen discrete: 1
    # vowels: 30gen continuous: 2
    # voiwels: 100gen cont: 1
    # voiwels: 100gen combi: 1
    # all: 100gen cont: 15
    # all: 100gen disc: 19
    # qll: 100gen combi: 19

# no mating
    # all: 100gen cont: 22
    # all: 100gen dosc: 22
    # vowels: 100 gen cont: 2
    
#advanced mating:
    #vowel 100gen cont: 0 (0.12913)
    #vowel 100gen combi: 0, 0.00000 (after about 90 gens)
    #all 100gen combi: 14 (27.4314)
    #all 100gen cont: 16 (14.64) 
    #all 100gen disc: 