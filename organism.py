import numpy as np
import random
from numba import njit

#MODIFY ALSO THIS IF YOU MODIFY THE SIZE OF THE WORLD IN SIMULATION
#------------------------------##
l=105                          ##
h=105                          ##
#------------------------------##

#-------PLANTS COSTANTS--------------

cost_plant_energy_needed=1
cost_plant_get_energy=1e-1
cost_plant_reproduction_chance=0.33
cost_plant_maturity=0.25 #<1

#-------PLANTS CLASS-----------------

class Plant:
    def __init__(self,i,j,height,roots,leaves,lifespan):
        #position
        self.x=j
        self.y=i
        
        #base stats, between 0 and 1, except lifespan
        self.base_height=height
        self.base_roots=roots
        self.base_leaves=leaves
        self.lifespan=lifespan
        
        #true stats
        self.age=0
        self.height=0
        self.roots=0
        self.leaves=0
        
        #needs
        self.energy_needed=0
        
        #stock
        self.energy=0
        
    def show_stats(self):
        '''Prints the plant's stats in a readable format.'''
        print("---------------------------")
        print("POSITION: (", self.x, ",", self.y, ")")
        print("HEIGHT:", self.height, "/", self.base_height)
        print("ROOTS:", self.roots, "/", self.base_roots)
        print("LEAVES:", self.leaves, "/", self.base_leaves)
        print("AGE:", self.age, "/", int(self.lifespan))
        print("ENERGY:", self.energy, "/", self.energy_needed)
        print("---------------------------")
    
    def ageing(self):
        '''Changes the age and all the stats of a plant simulating growth.'''
        self.age+=1
        self.height=self.age*self.base_height/self.lifespan
        self.roots=(self.age/self.lifespan)**(1/3)
        self.leaves=(self.age/self.lifespan)**3
        self.energy-=self.energy_needed
        self.energy_needed=cost_plant_energy_needed*self.height*self.roots*self.leaves
        
    def get_energy(self,nutrients,water,light):
        '''Simulates the energy gain for plants.'''
        if nutrients>0 and water>0 and light>0:
            hypothetical_gain=cost_plant_get_energy*((nutrients+water)*self.roots**2+self.height*self.leaves*light)
            if hypothetical_gain>3*self.energy_needed:
                self.energy+=3*self.energy_needed
                return 1.5*self.energy_needed,1.5*self.energy_needed
            else:
                self.energy+=hypothetical_gain
                return cost_plant_get_energy*nutrients*self.roots**2, cost_plant_get_energy*water*self.roots**2 #for simulation purpose
        return 0,0
        
    def try_reproduction(self,height_map):
        '''Simulates plant reproduction and genetic inheritance to the offspring.'''
        if self.energy>2*self.energy_needed and self.age>cost_plant_maturity*self.lifespan:
            self.energy-=self.energy_needed
            r=random.random()
            if r<cost_plant_reproduction_chance:
                i,j=TAC_matrix(matrix_for_plant_reproduction(height_map,self.y,self.x))
                while i<0 or i>len(height_map)-1 or j<0 or j>len(height_map[0])-1:
                    i,j=TAC_matrix(matrix_for_plant_reproduction(height_map,self.y,self.x))
                height=random.gauss(self.base_height,1e-2)
                while height<0 or height>1:
                    height=random.gauss(self.base_height,1e-2)
                roots=random.gauss(self.base_roots,1e-2)
                while roots<0 or roots>1:
                    roots=random.gauss(self.base_roots,1e-2)
                leaves=random.gauss(self.base_leaves,1e-2)
                while leaves<0 or leaves>1:
                    leaves=random.gauss(self.base_leaves,1e-2)
                lifespan=random.gauss(self.lifespan,1)
                while lifespan<10 or lifespan>50:
                    lifespan=random.gauss(self.lifespan,1)
                return Plant(i,j,height,roots,leaves,lifespan)
            return None    
                                
def mean_stats_plant(plants):
    '''Returns the mean of the stats for a list of plants. If the list is empty it returns 0 for each stat.'''
    if len(plants)>0:
        heights=[]
        rootss=[]
        leavess=[]
        lifespans=[]
        for plant in plants:
            heights.append(plant.base_height)
            rootss.append(plant.base_roots)
            leavess.append(plant.base_leaves)
            lifespans.append(plant.lifespan)
        return np.mean(heights),np.mean(rootss),np.mean(leavess),np.mean(lifespans)
    else:
        return 0,0,0,0

#-------ANIMALS----------------------

Male=1
Female=0        
cost_energy_needed=5e-2
cost_maturity=0.25 #<1

#-------ANIMALS CLASS----------------

class Animal:
    def __init__(self,i,j,height,largeness,speed,lifespan,gender):
        #position
        self.x=j
        self.y=i
        
        #base stats, between 0 and 1, except ideal temperature, lifespan and gender
        self.base_height=height
        self.base_largeness=largeness
        self.speed=speed
        self.lifespan=lifespan
        self.gender=gender
        
        #true stats
        self.age=0
        self.height=self.base_height/2
        self.largeness=self.base_largeness/2
        
        #needs
        self.energy_needed=cost_energy_needed*self.largeness*(self.largeness*self.height+self.speed**2)
        
        #stock
        self.energy=self.energy_needed
        
    def show_stats(self):
        '''Prints the animal's stats in a readable format.'''
        print("---------------------------")
        print("POSITION: (", self.x, ",", self.y, ")")
        print("HEIGHT:", self.height, "/", self.base_height)
        print("LARGENESS:", self.largeness, "/", self.base_largeness)
        print("SPEED:", self.speed)
        if self.gender==1:
            print("GENDER: MALE")
        else:
            print("GENDER: FEMALE")
        print("AGE:", self.age, "/", int(self.lifespan))
        print("ENERGY:", self.energy, "/", self.energy_needed)
        print("---------------------------")
    
    def ageing(self):
        '''Changes the age and all the stats of an animal simulating growth.'''
        self.age+=1
        self.height=self.base_height/(1+np.exp(-4*self.age/self.lifespan))
        self.largeness=self.base_largeness/(1+np.exp(-4*self.age/self.lifespan))
        self.energy-=self.energy_needed
        self.energy_needed=cost_energy_needed*self.largeness*(self.largeness*self.height+self.speed**2)

def mean_stats_animal(animals):
    '''Returns the mean of the stats for a list of animals, except gender. If the list is empty it returns 0 for each stat.'''
    if len(animals)>0:
        heights=[]
        largenesss=[]
        speeds=[]
        lifespans=[]
        for animal in animals:
            heights.append(animal.base_height)
            largenesss.append(animal.base_largeness)
            speeds.append(animal.speed)
            lifespans.append(animal.lifespan)
        return np.mean(heights),np.mean(largenesss),np.mean(speeds),np.mean(lifespans)
    else:
        return 0,0,0,0
        
#-------HERBIVORES COSTANTS----------

cost_herb_get_energy=0.5 #prima 1.5
cost_herb_reproduction_chance=0.35

#-------HERBIVORES CLASS-------------

class Herbivore(Animal):
    
    def get_energy(self,plants):
        '''Simulates the energy gain for herbivores.'''
        for plant in plants:
            if plant.height<self.height+0.1: #CHECK IF IT CAN REACH THE LEAVES
                self.energy+=cost_herb_get_energy*plant.leaves
                plant.leaves-=cost_herb_get_energy*plant.leaves
                if self.energy>3*self.energy_needed:
                    break        
   
    def try_move_or_reproduction(self,herbivores,height_map,plant_count):
        '''Simulates the movement or the reproduction for herbivores and the passing of genes to the kid. Returns the new herbivore if successful.'''
        if self.energy<2*self.energy_needed: #THE HERBIVORE STRUGGLED DURING THE LAST TURN SO SEARCH FOR A BETTER PLACE
            matrix_for_choice=help_movement(plant_count,self.y,self.x)
            y,x=TAC_matrix(matrix_for_choice) #CHOOSE WHERE TO GO
            new_y=self.y
            new_x=self.x
            if y<self.y: 
                new_y=self.y-1
            elif y>self.y:
                new_y=self.y+1
            if x<self.x: 
                new_x=self.x-1
            elif x>self.x:
                new_x=self.x+1
            if height_map[new_y][new_x]<0.5:
                if height_map[self.y][new_x]<0.5:
                    if height_map[new_y][self.x]>0.5:
                        self.y=new_y
                else:    
                    self.x=new_x
            else:
                self.x=new_x
                self.y=new_y                    
        elif self.energy>2*self.energy_needed and self.age>cost_maturity*self.lifespan: #THE HERBIVORE CAN BREED
            mate=None
            n=0
            while mate==None and n<len(herbivores): #SEARCHING FOR A MATE
                if herbivores[n].gender!=self.gender and herbivores[n].energy>2*self.energy_needed and herbivores[n].age>cost_maturity*herbivores[n].lifespan:
                    mate=herbivores[n]
                    herbivores[n].energy_needed-=0.5*herbivores[n].energy_needed
                n+=1
            if mate!=None: #SIMULATING GENES (BUT NOT ALLELES)
                self.energy-=0.5*self.energy_needed
                r=random.random()
                couple=[self,mate]
                if r<cost_herb_reproduction_chance:
                    gene=random.randint(0,1)
                    height=random.gauss(couple[gene].base_height,1e-2)
                    while height<0 or height>1:
                        height=random.gauss(couple[gene].base_height,1e-2)
                    
                    gene=random.randint(0,1)
                    largeness=random.gauss(couple[gene].base_largeness,1e-2)
                    while largeness<0 or largeness>1:
                        largeness=random.gauss(couple[gene].base_largeness,1e-2)
                    
                    gene=random.randint(0,1)
                    speed=random.gauss(couple[gene].speed,1e-2)
                    while speed<0 or speed>1:
                        speed=random.gauss(couple[gene].speed,1e-2)
                        
                    gene=random.randint(0,1)
                    lifespan=random.gauss(couple[gene].lifespan,1)
                    while lifespan<10 or lifespan>50:
                        lifespan=random.gauss(couple[gene].lifespan,1)
                    gender=random.randint(0,1)
                    return Herbivore(self.y,self.x,height,largeness,speed,lifespan,gender)
            return None    

#-------CARNIVORES COSTANTS----------

cost_carb_get_energy=5
cost_carb_reproduction_chance=0.35
cost_body=1.1 #>1

#-------HERBIVORES CLASS-------------

class Carnivore(Animal):
    
    def get_energy(self,herbivores,bodies):
        '''Simulates the energy gain for carnivores.'''
        new_herbivores=[]
        n=0
        for herbivore in herbivores:
            n+=1
            if herbivore.largeness<self.largeness-0.1 or herbivore.speed<self.speed-0.1: #CHECK IF IT WINS
                self.energy+=cost_carb_get_energy*herbivore.energy_needed
                if self.energy>3*self.energy_needed:
                    break        
            else:
                new_herbivores.append(herbivore) #UPDATES THE LIST OF HERBIVORES REMAINING
        for i in range(n,len(herbivores)):
            new_herbivores.append(herbivores[i])
        if self.energy<cost_body*self.energy_needed: #IF HUNGRY IT START EATING DEAD BODIES BUT NOT TOO MUCH, DEPENDING ON cost_body
            for body in bodies:
                if body.eff_energy<cost_body*self.energy_needed-self.energy:
                    self.energy+=body.eff_energy
                    body.eff_energy=0 #simulation purpose                        
                else:
                    body.eff_energy-=cost_body*self.energy_needed-self.energy
                    self.energy=cost_body*self.energy_needed     
                if self.energy>=cost_body*self.energy_needed:
                    break
        return new_herbivores
       
    def try_move_or_reproduction(self,carnivores,height_map,herbivore_count):
        '''Simulates the movement or the reproduction for carnivores and the passing of genes to the kid. Returns the new carnivore if successful.'''
        #SAME LOGIC OF HERBIVORES
        if self.energy<2*self.energy_needed:
            matrix_for_choice=help_movement(herbivore_count,self.y,self.x)
            y,x=TAC_matrix(matrix_for_choice) #CHOOSE WHERE TO GO
            new_y=self.y
            new_x=self.x
            if y<self.y: 
                new_y=self.y-1
            elif y>self.y:
                new_y=self.y+1
            if x<self.x: 
                new_x=self.x-1
            elif x>self.x:
                new_x=self.x+1
            if height_map[new_y][new_x]<0.5:
                if height_map[self.y][new_x]<0.5:
                    if height_map[new_y][self.x]>0.5:
                        self.y=new_y
                else:    
                    self.x=new_x
            else:
                self.x=new_x
                self.y=new_y
        elif self.energy>2*self.energy_needed and self.age>cost_maturity*self.lifespan:
            mate=None
            n=0
            while mate==None and n<len(carnivores):
                if carnivores[n].gender!=self.gender and carnivores[n].energy>2*self.energy_needed and carnivores[n].age>cost_maturity*carnivores[n].lifespan:
                    mate=carnivores[n]
                    carnivores[n].energy_needed-=0.5*carnivores[n].energy_needed
                n+=1
            if mate!=None:
                self.energy-=0.5*self.energy_needed
                r=random.random()
                couple=[self,mate]
                if r<cost_carb_reproduction_chance:
                    gene=random.randint(0,1)
                    height=random.gauss(couple[gene].base_height,1e-2)
                    while height<0 or height>1:
                        height=random.gauss(couple[gene].base_height,1e-2)
                    
                    gene=random.randint(0,1)
                    largeness=random.gauss(couple[gene].base_largeness,1e-2)
                    while largeness<0 or largeness>1:
                        largeness=random.gauss(couple[gene].base_largeness,1e-2)
                    
                    gene=random.randint(0,1)
                    speed=random.gauss(couple[gene].speed,1e-2)
                    while speed<0 or speed>1:
                        speed=random.gauss(couple[gene].speed,1e-2)
                        
                    gene=random.randint(0,1)
                    lifespan=random.gauss(couple[gene].lifespan,1)
                    while lifespan<10 or lifespan>50:
                        lifespan=random.gauss(couple[gene].lifespan,1)
                    gender=random.randint(0,1)
                    return Carnivore(self.y,self.x,height,largeness,speed,lifespan,gender)
            return None    

#------BODIES CLASS------------------

class Body:
    def __init__(self,i,j,energy):
        self.x=j
        self.y=i
        self.energy=energy
        self.eff_energy=energy

    def decomposing(self,nutrients_map):
        '''Simulates decomposition of a body (loss of energy stored).'''
        if self.eff_energy>(1/5)*self.energy:
            nutrients_map[self.y][self.x]+=(1/5)*self.energy
            self.eff_energy-=(1/5)*self.energy
            return True
        elif self.eff_energy>0:
            nutrients_map[self.y][self.x]+=self.eff_energy
            self.eff_energy=0
            return True
        else:
            return False

#------USEFUL FUNCTIONS USED BEFORE OR IN simulation.py
            
def count_organism(organisms,lenght=l,height=h):
    '''Counts the number of organism (list) in each tile and returns the matrix containing the results.'''
    count=np.zeros((height,lenght))
    for organism in organisms:
        count[organism.y][organism.x]+=1
    return count
        
def TAC_matrix(matrix):
    '''Uses Try and Catch algorithm on a matrix. It returns a tile of the matrix. 
    Works only with positive values.'''
    y=random.randint(0,len(matrix)-1)
    x=random.randint(0,len(matrix[0])-1)
    z=random.uniform(0,np.max(matrix))
    while matrix[y][x]<z:
        y=random.randint(0,len(matrix)-1)
        x=random.randint(0,len(matrix[0])-1)
        z=random.uniform(0,np.max(matrix))
    return y,x

@njit
def help_movement(count,y,x,sigma=2):
    '''Applies a gaussian filter to the count matrix simulating perception of animal (the closer the better).'''
    matrix=np.zeros((len(count),len(count[0])))
    for i in range(0,len(count)):
        for j in range(0,len(count[0])):
            matrix[i][j]=count[i][j]*np.exp(-(((i-y)/sigma)**2+((j-x)/sigma)**2))
    return matrix

help_movement(np.zeros((1,1)),0,0)

@njit
def matrix_for_plant_reproduction(height_map,y,x,sigma=3):
    '''Returns a probability matrix to choose the position for a newborn plant.'''
    matrix=np.zeros((len(height_map),len(height_map[0])))
    for i in range(0,len(height_map)):
        for j in range(0,len(height_map[0])):
            if height_map[i][j]>0.5:
                matrix[i][j]=np.exp(-(((y-i)/sigma)**2+((x-j)/sigma)**2))
    return matrix

matrix_for_plant_reproduction(np.zeros((1,1)),0,0)    