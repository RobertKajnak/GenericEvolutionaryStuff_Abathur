import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import time
# learning rate epsilon set globally:
epsilon = .3

# activation function:
sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid=lambda x: np.exp(x)/( (np.exp(x) + 1)**2)

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
                 gene_mutation_chance=.1, mutation_swing = 1, discrete=False ):
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
            if discrete:
                E += np.abs(t-np.floor(y+.5))
            else:
                E += (t-y) ** 2
            
            #print t,y,np.abs(t-int(y+.5))
        #to make it 'the higher the better'
        E = np.sum(E)
        return -E#/len(self.dataset)
    
    def mate(self,parent1,parent2):
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
        
        return child1,child2
    
    def mutate(self,pool):
        #i =0 
        newpool = []
        for indituple in pool:
            individual = indituple[1]
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
            #i+=1
        return newpool
    
    def evolve(self):
        self.generation += 1
        
        prime_individual = self.pool[0][1]
        self.pool = self.mutate(self.pool[1:])
        self.insort(self.pool,prime_individual)
        
        newpool = self.pool[:self.leftover_parents]
        for i in range(0,self.survivors-self.leftover_parents-1,2):
            child1,child2 = self.mate(self.pool[i][1],self.pool[i+1][1])
            self.insort(newpool,child1)
            self.insort(newpool,child2)
    
        self.pool = newpool
        for i in range(self.survivors,self.poolsize):
            self.insort(self.pool,MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax))
    
        return
    #returns the best performing individual
    def prime_specimen(self,):
        return self.pool[0][1]
    
    def disp_best_specimens(self,n=5):
        print "Generation:",self.generation
        print     "Place -   Score"
        for i in range(0,min(n,len(self.pool))):
            print "%d.   -   %f"%(i,-self.pool[i][0])

    
    
def train(net, dataset, plot_learning=False):
    dset = dataset[:]
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
                      individual_mutation_chance=.3,gene_mutation_chance=.2)
    
    for i in range(0,100):
        abathur.disp_best_specimens()
        print abathur.prime_specimen().Whx
        print abathur.prime_specimen().Wyh
        abathur.evolve()
    
    abathur.disp_best_specimens()
    net = abathur.prime_specimen()
    print net.Whx
    print net.Wyh
    print abathur.fitness(abathur.prime_specimen())


    # what is the output for the entire dataset?
    for s, (x, t) in zip(symbols, dataset):
        y = net.ffwd(x)
        #enhanced the print statement a bit -- shows if  a value is incorrectly predicted
        print "output for {}: {} => {}  {}".format(s, np.round(y, 3), int(y > 0.5), \
                          ' -- FAILED' if not t==int(y>0.5) else ' ')
    
    return

def learn_all_by_evolution():
    #get dataset
    symbols, dataset =  get_dataset(['AEIOU','1234567890','!"#$%&()*+=,-./ '], True)
    
    # now, let's make/train a perceptron:
    abathur = Abathur((6,10,3),dataset, poolsize=1500, survivors = 1100,\
                      leftover_parents=30,Wmin=-15,Wmax=15,\
                      individual_mutation_chance=.5,gene_mutation_chance=.4,\
                      discrete=False )
    
    for i in range(0,100):
        abathur.disp_best_specimens()
        print abathur.fitness(abathur.prime_specimen(),discrete=True)
        print abathur.prime_specimen().Whx
        print abathur.prime_specimen().Wyh
        abathur.evolve()
    
    abathur.disp_best_specimens()
    net = abathur.prime_specimen()
    print net.Whx
    print net.Wyh
    
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
    return

if __name__ == '__main__':
    #learn_xor()
    #learn_vowels()
    #learn_all()
    
    #learn_vowels_by_evolution()
    learn_all_by_evolution()
    
# pairing mating:
    # vowels: 100gen discrete: 1
    # vowels: 30gen continuous: 2
    # voiwels: 100gen cont: 1
    # voiwels: 100gen combi: 1
    # all: 100gen cont: 15
    # all: 100gen disc: 19
    # qllÉ 100gen combi: 19
# 
    
    