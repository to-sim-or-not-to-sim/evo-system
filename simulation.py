import noise
import organism
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import random
import matplotlib.animation as animation

#--SETTING CONSTANTS FOR WORLD-##
#------------------------------##
l=105                          ##
h=105                          ##
steps_h=[7,15,21,35,21,15,7]   ##
weights_h=[1,2,3,4,3,2,1]      ##
steps_t=[35,21,15,7]           ##
weights_t=[2,2,1,1]            ##
steps_hum=[15,21,35]           ##
weights_hum=[1,1,1]            ##
steps_nut=[7,15,21]            ##
weights_nut=[1,1,1]            ##
steps_c=[7,15,15,21,21,35]     ##
weights_c=[1,2,2,3,3,4]        ##
step_evo_c=15                  ##
#------------------------------##
num_p=900000                   ##
num_herb=50000                 ##
num_carn=1500                  ##
#------------------------------##
#IF YOU CHANG l AND h YOU HAVE TO MODIFY THEM ALSO IN organism.py


#--------WORKING ON MATRIX----------

@njit
def helper_plants_generator(height_matrix,sigma=(l+h)/(10)):
    '''Helps with the generation of plants in the world creating a probability matrix using height map and a gaussian.'''
    helper_matrix=np.zeros((len(height_matrix),len(height_matrix[0])))
    for i in range(0,len(height_matrix)):
        for j in range(0,len(height_matrix[0])):
            if height_matrix[i][j]>0.5:
                helper_matrix[i][j]=np.exp(-(((i-(len(height_matrix)/2))/sigma)**2+((j-(len(height_matrix[0])/2))/sigma)**2))
    return helper_matrix

helper=helper_plants_generator(np.zeros((1,1)))

def random_plants(N,height_map):
    '''Generates N random plants.'''
    plants=[]
    matrix=helper_plants_generator(height_map)
    print("GENERATING PLANTS...")
    print("")
    for k in range(0,N):
        i,j=organism.TAC_matrix(matrix)
        height=random.random()
        roots=random.random()
        leaves=random.random()
        lifespan=random.randint(10,50)
        plants.append(organism.Plant(i,j,height,roots,leaves,lifespan))
        print("\033[1A",k+1,"/",N)
    return plants

def random_herbivores(N,plants_count):
    '''Generates N random herbivores based on the plants.'''
    herbivores=[]
    print("GENERATING HERBIVORES...")
    print("")
    for i in range(0,N):
        y,x=organism.TAC_matrix(plants_count)
        height=random.random()
        largeness=random.random()
        speed=random.random()
        lifespan=random.randint(10,50)
        gender=random.randint(0,1)
        herbivores.append(organism.Herbivore(y,x,height,largeness,speed,lifespan,gender))
        print("\033[1A",i+1,"/",N)
    return herbivores

def random_carnivores(N,herbs_count):
    '''Generates N random carnivores based on the herbivores.'''
    carnivores=[]
    print("GENERATING CARNIVORES...")
    print("")
    for i in range(0,N):
        y,x=organism.TAC_matrix(herbs_count)
        height=random.random()
        largeness=random.random()
        speed=random.random()
        lifespan=random.randint(10,50)
        gender=random.randint(0,1)
        carnivores.append(organism.Carnivore(y,x,height,largeness,speed,lifespan,gender))
        print("\033[1A",i+1,"/",N)
    return carnivores

def fill_grid(grid,plants,herbivores,carnivores,bodies):
    '''Fills the biomap in the World class starting from lists of plants, herbivores, carnivores and bodies.'''
    for plant in plants:
        grid[plant.y][plant.x][0].append(plant)
    for herbivore in herbivores:
        grid[herbivore.y][herbivore.x][1].append(herbivore)
    for carnivore in carnivores:
        grid[carnivore.y][carnivore.x][2].append(carnivore)
    for body in bodies:
        grid[body.y][body.x][3].append(body)

#--------WORLD COSTANTS-------------

rain_cost=5e-1
decomposition_cost=5e-1

#--------WORLD CLASS----------------

class World:
    def __init__(self, lenght=l, height=h,
                 steps_height=steps_h, weights_height=weights_h,
                 steps_temperature=steps_t, weights_temperature=weights_t,
                 steps_humidity=steps_hum, weights_humidity=weights_hum,
                 steps_nutrients=steps_nut, weights_nutrients=weights_nut,
                 steps_clouds=steps_c, weights_clouds=weights_c):
        '''Initialize the world generating all the maps using noise.py.'''
        print("GENERATING WORLD...")
        
        self.height_map=noise.generate_height_map(lenght,height,steps_height,weights_height)
        self.temperature_map=noise.generate_temp_map(self.height_map,steps_t,weights_t)
        self.humidity_map=noise.generate_hum_map(self.height_map,self.temperature_map,steps_humidity,weights_humidity)
        self.nutrients_map=noise.generate_nut_map(self.height_map,self.temperature_map,self.humidity_map,steps_nutrients,weights_nutrients)
        
        #changes due to other effects for example rain, wind ecc ecc... 
        self.eff_temperature_map=np.zeros((len(self.height_map),len(self.height_map[0])))
        self.eff_humidity_map=np.zeros((len(self.height_map),len(self.height_map[0])))
        self.eff_nutrients_map=np.zeros((len(self.height_map),len(self.height_map[0])))
        
        self.cloud_map=noise.layermap(len(self.height_map[0]),len(self.height_map),steps_c,weights_c,False)
        
        self.eff_temperature_map=self.temperature_map.copy()
        self.eff_humidity_map=self.humidity_map.copy()
        self.eff_nutrients_map=self.nutrients_map.copy()
        
        self.bio_map=np.empty((len(self.height_map),len(self.height_map[0])),dtype=object) 
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                self.bio_map[i][j]=[ [], [], [], [] ] #0=plants - 1=herbivores - 2=carnivores - 3=bodies
        
    def reset(self):
        '''Resets all the "eff" map of the world.'''
        self.eff_temperature_map=self.temperature_map.copy()
        self.eff_humidity_map=self.humidity_map.copy()
        self.eff_nutrients_map=self.nutrients_map.copy()
    
    def show_maps(self,which=None):
        '''Shows the maps of the world and saves theme. Modifing "which" you can choose which map you want to show and save:\n
        1- which="height": height map\n
        2- which="temperature": temperature map\n
        3- which="humidity": humidity map\n
        4- which="nutrients": nutrients map\n
        5- which=anything else: all the maps.'''
        if which=="height":
            noise.print_height_map(self.height_map)
        elif which=="temperature":
            noise.print_temp_map(self.temperature_map)
        elif which=="humidity":
            noise.print_hum_map(self.humidity_map)
        elif which=="nutrients":
            noise.print_nut_map(self.nutrients_map)
        else:
            noise.print_height_map(self.height_map)
            noise.print_temp_map(self.temperature_map)
            noise.print_hum_map(self.humidity_map)
            noise.print_nut_map(self.nutrients_map)
        
    def update_clime(self,step=step_evo_c):
        '''Updates the eff temperature and humidity map simulating wind and rain modifing the cloud map adding a noise map over the precedent one.'''
        self.cloud_map+=noise.noise(len(self.height_map[0]),len(self.height_map),step)
        self.cloud_map/=2
        self.eff_temperature_map=0.5*self.temperature_map*(2-self.cloud_map)
        self.eff_humidity_map+=rain_cost*self.cloud_map
        
    def update_plants(self,plants):
        '''Simulates the energy gained by a list of plants and the water and nutrients lost by the tiles of the world.'''
        for plant in plants:
            nutrients_taken,water_taken=plant.get_energy(self.eff_nutrients_map[plant.y][plant.x],self.eff_humidity_map[plant.y][plant.x],1-self.cloud_map[plant.y][plant.x])
            self.eff_nutrients_map[plant.y][plant.x]-=nutrients_taken
            self.eff_humidity_map[plant.y][plant.x]-=water_taken
        
    def update_herbivores(self,herbivores,plants):
        '''Simulates the energy gained by a list of herbivores.'''
        for herbivore in herbivores:
            herbivore.get_energy(plants)
        
    def update_carnivores(self,carnivores,herbivores,bodies):
        '''Simulates the energy gained by a list of carnivores. Returns the list of herbivores that survived.'''
        for carnivore in carnivores:
            herbivores=carnivore.get_energy(herbivores,bodies)
        return herbivores

    def ageing_plant(self,plants):
        '''Simulates ageing for plants adding nutrients of dead plants to the soil.'''
        new_plants=[]
        for plant in plants:
            plant.ageing()
            if plant.energy<0 or plant.age>plant.lifespan:
                self.eff_nutrients_map+=decomposition_cost*plant.height*plant.roots*plant.leaves
            else:
                new_plants.append(plant)
        return new_plants    

    def ageing_animal(self,animals,bodies): 
        '''Simulates ageing for animals adding dead animals to the list of bodies.'''
        new_animals=[]
        for animal in animals:
            animal.ageing()
            if animal.energy<0 or animal.age>animal.lifespan:
                bodies.append(organism.Body(animal.y,animal.x,animal.energy_needed))
            else:
                new_animals.append(animal)
        return new_animals
                
    def update_bodies(self,bodies):
        '''Simulates the decomposition of bodies.'''
        new_bodies=[]
        for body in bodies:
            remain=body.decomposing(self.eff_nutrients_map)
            if remain==True:
                new_bodies.append(body)
        return new_bodies

    def refill_nutrients(self):
        '''Simulates natural refilling of nutrients.'''
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                if self.eff_nutrients_map[i][j]<(4/5)*self.nutrients_map[i][j]:
                    self.eff_nutrients_map[i][j]+=(1/5)*self.nutrients_map[i][j]
    
    def refill_water(self):
        '''Simulates natural refilling of water.'''
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                if self.eff_humidity_map[i][j]<(4/5)*self.humidity_map[i][j]:
                    self.eff_humidity_map[i][j]+=(1/5)*self.humidity_map[i][j]
    
    def organism_count(self):
        '''Returns the matrixes with the number of plants, herbivores and carnivores in each tile.'''
        plant_count=np.zeros((len(self.height_map),len(self.height_map[0])))
        herbivore_count=np.zeros((len(self.height_map),len(self.height_map[0])))
        carnivore_count=np.zeros((len(self.height_map),len(self.height_map[0])))
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                plant_count[i][j]=len(self.bio_map[i][j][0])
                herbivore_count[i][j]=len(self.bio_map[i][j][1])
                carnivore_count[i][j]=len(self.bio_map[i][j][2])
        return plant_count,herbivore_count,carnivore_count
                
    def update_world(self,plant_count,herbivore_count,carnivore_count):
        '''Updates all the organisms in the world ageing them and simulating eneergy gain and death.'''
        new_bio_map=np.empty((len(self.height_map),len(self.height_map[0])),dtype=object)
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                new_bio_map[i][j]=[ [], [], [], [] ] #0=plants - 1=herbivores - 2=carnivores - 3=bodies
        print("0%")
        tot=len(self.height_map)*len(self.height_map[0])
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                plants,herbivores,carnivores,bodies=self.bio_map[i][j]
                #--IT TAKES LONGER BUT IT'S MORE REALISTIC----------
                plants=self.ageing_plant(plants)
                herbivores=self.ageing_animal(herbivores,bodies)
                carnivores=self.ageing_animal(carnivores,bodies)        
                #---------------------------------------------------
                #----CARNIVORES--------------
                herbivores=self.update_carnivores(carnivores,herbivores,bodies)
                carnivore_len=len(carnivores)
                for n in range(0,carnivore_len):
                    new_carnivore=carnivores[n].try_move_or_reproduction(carnivores,self.height_map,herbivore_count+0.5*carnivore_count)
                    if new_carnivore!=None:
                        carnivores.append(new_carnivore)
                #----HERBIVORES--------------
                self.update_herbivores(herbivores,plants)
                herbivore_len=len(herbivores)
                for n in range(0,herbivore_len):
                    new_herbivore=herbivores[n].try_move_or_reproduction(herbivores,self.height_map,plant_count+0.5*herbivore_count)
                    if new_herbivore!=None:
                        herbivores.append(new_herbivore)
                #----PLANTS------------------
                self.update_plants(plants)
                plant_len=len(plants)
                for n in range(0,plant_len):
                    new_plant=plants[n].try_reproduction(self.height_map)
                    if new_plant!=None:
                        plants.append(new_plant)
                #----BODIES-------------------
                if len(bodies)!=0:
                    bodies=self.update_bodies(bodies)   
                fill_grid(new_bio_map,plants,herbivores,carnivores,bodies)
                print("\033[1A",int(((i*(len(self.height_map[0]))+(j+1))/tot)*100),"%")
        self.bio_map=new_bio_map
    
    def unpack_organisms(self):
        '''Returns the list of plants, herbivores and carnivores.'''
        final_plants=[]
        final_herbivores=[]
        final_carnivores=[]
        for i in range(0,len(self.height_map)):
            for j in range(0,len(self.height_map[0])):
                plants,herbivores,carnivores,bodies=self.bio_map[i][j]
                for plant in plants:
                    final_plants.append(plant)
                for herbivore in herbivores:
                    final_herbivores.append(herbivore)
                for carnivore in carnivores:
                    final_carnivores.append(carnivore)
        return final_plants,final_herbivores,final_carnivores
    
    def show_bio_map(self,plants_counts,herbivores_counts,carnivores_counts):
        '''Function that starting from the count matrixes of all the organisms generates an animation of the simulation.'''
        fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
        ax[0].set_title("PLANTS")
        ax[1].set_title("HERBIVORES")
        ax[2].set_title("CARNIVORES")
    
        plant_image=ax[0].imshow(plants_counts[0],cmap='Greens',interpolation='bilinear')
        herbivore_image=ax[1].imshow(herbivores_counts[0],cmap='Blues',interpolation='bilinear')
        carnivore_image=ax[2].imshow(carnivores_counts[0],cmap='Reds',interpolation='bilinear')
        
        def update(frame):
            plant_image.set_data(plants_counts[frame])
            herbivore_image.set_data(herbivores_counts[frame])
            carnivore_image.set_data(carnivores_counts[frame])
            return plant_image,herbivore_image,carnivore_image
            
        ani=animation.FuncAnimation(fig,update,frames=range(0,len(plants_counts)),interval=50,blit=True)
        ani.save("biomap.gif", writer="pillow",fps=10,dpi=200)
        plt.show()
                
    def initialize_simulation(self,number_of_plants=num_p,number_of_herbivore=num_herb,number_of_carnivore=num_carn):
        '''Generates the organism all at once and returns the lists containing them in this order: plants, herbivores, carnivores. It also shows the world height map.'''
        print("PREPARING TO SHOW WORLD...")
        self.show_maps("height")
        self.update_clime()
        #create the organisms
        print("GENERATING ORGANISMS...")
        plants=random_plants(number_of_plants,self.height_map)
        plants_count=organism.count_organism(plants)
        print("PLANTS GENERATED")
        herbivores=random_herbivores(number_of_herbivore,plants_count)
        herb_count=organism.count_organism(herbivores)
        print("HERBIVORES GENERATED")
        carnivores=random_carnivores(number_of_carnivore,herb_count)
        print("CARNIVORES GENERATED")
        return plants,herbivores,carnivores

    def simulation(self,days=100,saving=True):
        '''This kind of simulation starts using "initialize-simulation" as start. 
        Needs a larger number of organisms at the beginning due to adaptability problems.
        At the end shows many plots of the stats and number of the organisms and an animation of the populations. It also saves these.
        Using saving=True it's possible to save the world and organisms datas to start a new simulation from these datas using "simulation_from_data".'''
        plants,herbivores,carnivores=self.initialize_simulation()
        bodies=[]
        
        #LIST TO SAVE DATA--------------------##
        #TO SAVE DAYS PASSED------------------##
        d=[]                                  ##
        #TO SAVE THE NUMBER OF ORGANISMS------##
        plants_number=[]                      ##
        herbivores_number=[]                  ##
        carnivores_number=[]                  ##
        female_herbivores_number=[]           ##
        female_carnivores_number=[]           ##
        male_herbivores_number=[]             ##
        male_carnivores_number=[]             ##
        #TO SAVE THE STATS OF PLANTS----------##
        plants_height=[]                      ##
        plants_roots=[]                       ##
        plants_leaves=[]                      ##
        plants_lifespan=[]                    ##
        #TO SAVE THE STATS OF HERBIVORES------##
        herbivores_height=[]                  ##
        herbivores_largeness=[]               ##
        herbivores_speed=[]                   ##
        herbivores_lifespan=[]                ##
        #TO SAVE THE STATS OF CARNIVORES------##
        carnivores_height=[]                  ##
        carnivores_largeness=[]               ##
        carnivores_speed=[]                   ##
        carnivores_lifespan=[]                ##
        #FOR ANIMATION------------------------##
        plants_counts=[]                      ##
        herbivores_counts=[]                  ##
        carnivores_counts=[]                  ##
        #-------------------------------------##
        
        print("INITIALIZING...")
        #---INITIALIZING ALL ORGANISMS------------------
        fill_grid(self.bio_map,plants,herbivores,carnivores,bodies)
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        self.update_world(plant_count,herbivore_count,carnivore_count)
        plants,herbivores,carnivores=self.unpack_organisms()
        #---LET'S START THE TRUE SIMULATION-------------
        print("READY!")
        bodies=[] #delete al bodies of the first turn
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        fill_grid(self.bio_map,plants,herbivores,carnivores,bodies)
                
        for dd in range(0,days):
            print("DAY:", dd+1)
            print("UPDATING CLIME...")
            self.update_clime()
            self.refill_nutrients()
            self.refill_water()
            print("DONE")
            
            #---------------------------------------------------
            print("UPDATING ORGANISMS...")
            self.update_world(plant_count,herbivore_count,carnivore_count)
            plant_count,herbivore_count,carnivore_count=self.organism_count()
            plants,herbivores,carnivores=self.unpack_organisms()
            print("DONE")
            #---------------------------------------------------
            if len(plants)==0:
                print("All plants are dead.")
                break
            if len(herbivores)==0:
                print("All herbivores are dead.")
                break
            if len(carnivores)==0:
                print("All carnivores are dead.")
                break            
            
            #DATA COLLECTION----------------------------------------------##
            #-------------------------------------------------------------##            
            d.append(dd+1)                                                ##
            #NUMBER-------------------------------------------------------##
            plants_number.append(len(plants))                             ##
            f=0                                                           ##
            for herbivore in herbivores:                                  ##
                if herbivore.gender==0:                                   ##
                    f+=1                                                  ##
            female_herbivores_number.append(f)                            ##
            male_herbivores_number.append(len(herbivores)-f)              ##
            herbivores_number.append(len(herbivores))                     ##
            f=0                                                           ##
            for carnivore in carnivores:                                  ##
                if carnivore.gender==0:                                   ##
                    f+=1                                                  ##
            female_carnivores_number.append(f)                            ##
            male_carnivores_number.append(len(carnivores)-f)              ##
            carnivores_number.append(len(carnivores))                     ##
            #PLANTS-------------------------------------------------------##
            means=organism.mean_stats_plant(plants)                       ##
            plants_height.append(means[0])                                ##
            plants_roots.append(means[1])                                 ##
            plants_leaves.append(means[2])                                ##
            plants_lifespan.append(means[3])                              ##
            #HERBIVORES---------------------------------------------------##
            means=organism.mean_stats_animal(herbivores)                  ##
            herbivores_height.append(means[0])                            ##
            herbivores_largeness.append(means[1])                         ##
            herbivores_speed.append(means[2])                             ##
            herbivores_lifespan.append(means[3])                          ##
            #CARNIVORES---------------------------------------------------##
            means=organism.mean_stats_animal(carnivores)                  ##
            carnivores_height.append(means[0])                            ##
            carnivores_largeness.append(means[1])                         ##
            carnivores_speed.append(means[2])                             ##
            carnivores_lifespan.append(means[3])                          ##
            #-------------------------------------------------------------##
            plants_counts.append(plant_count)                             ##
            herbivores_counts.append(herbivore_count)                     ##
            carnivores_counts.append(carnivore_count)                     ##
            #-------------------------------------------------------------##
            
            print("PLANTS:", len(plants), "HERBIVORES:", len(herbivores), "CARNIVORES:", len(carnivores))
        
        if len(d)>0:    
            print("PREPARING TO SHOW RESULTS...")
            #ANIMATION--------------------------------------------------------------------##
            self.show_bio_map(plants_counts,herbivores_counts,carnivores_counts)          ##          
            #PLOTS------------------------------------------------------------------------##
            #NUMBER OF ORGANISMS----------------------------------------------------------##
            plt.plot(d,plants_number,color="forestgreen",label="Number of plants")        ##
            plt.plot(d,herbivores_number,color="cyan",label="Number of herbivores")       ##
            plt.plot(d,carnivores_number,color="crimson",label="Number or carnivores")    ##
            plt.legend()                                                                  ##
            plt.savefig("0_organisms_number.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #HERBIVORES-------------------------------------------------------------------##
            plt.title("HERBIVORES")                                                       ##
            plt.plot(d,herbivores_number,color="cyan",label="Total number")               ##
            plt.plot(d,female_herbivores_number,color="pink",label="Number of females")   ##
            plt.plot(d,male_herbivores_number,color="blue",label="Number or males")       ##
            plt.legend()                                                                  ##
            plt.savefig("0_herbivores_number.png",dpi=500)                                ##
            plt.show()                                                                    ##
            #CARNIVORES-------------------------------------------------------------------##
            plt.title("CARNIVORES")                                                       ##
            plt.plot(d,carnivores_number,color="crimson",label="Total number")            ##
            plt.plot(d,female_carnivores_number,color="pink",label="Number of females")   ##
            plt.plot(d,male_carnivores_number,color="blue",label="Number or males")       ##
            plt.legend()                                                                  ##
            plt.savefig("0_carnivores_number.png",dpi=500)                                ##
            plt.show()                                                                    ##
            #PLANTS STATS-----------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,plants_height,color="forestgreen",label="HEIGHT")             ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,plants_roots,color="forestgreen",label="ROOTS")               ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,plants_leaves,color="forestgreen",label="LEAVES")             ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,plants_lifespan,color="forestgreen",label="LIFESPAN")         ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_plants_stats.png",dpi=500)                                     ##
            plt.show()                                                                    ##
            #HERBIVORES STATS-------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,herbivores_height,color="cyan",label="HEIGHT")                ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,herbivores_largeness,color="cyan",label="LARGENESS")          ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,herbivores_speed,color="cyan",label="SPEED")                  ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,herbivores_lifespan,color="cyan",label="LIFESPAN")            ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_herbivores_stats.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #CARNIVORES STATS-------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,carnivores_height,color="crimson",label="HEIGHT")             ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,carnivores_largeness,color="crimson",label="LARGENESS")       ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,carnivores_speed,color="crimson",label="SPEED")               ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,carnivores_lifespan,color="crimson",label="LIFESPAN")         ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_carnivores_stats.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #-----------------------------------------------------------------------------##
            print("DONE")
        else:
            print("RESULTS NOT VALID")    
        if saving==True:
            self.save_data()

    def save_data(self):
        '''It saves the world and organisms datas in .txt files.'''
        np.savetxt("height_map.txt",self.height_map,fmt="%0.5f")
        np.savetxt("temperature_map.txt",self.temperature_map,fmt="%0.3f")
        np.savetxt("humidity_map.txt",self.humidity_map,fmt="%0.5f")
        np.savetxt("nutrients_map.txt",self.nutrients_map,fmt="%0.5f")
        plants,herbivores,carnivores=self.unpack_organisms()
        with open("plants.txt","w") as file_plants:
            for plant in plants:
                file_plants.write(f"{plant.x} {plant.y} {plant.base_height} {plant.base_roots} {plant.base_leaves} {plant.lifespan} {plant.age} {plant.energy_needed} {plant.energy}\n")
        with open("herbivores.txt","w") as file_herbivores:
            for herbivore in herbivores:
                file_herbivores.write(f"{herbivore.x} {herbivore.y} {herbivore.base_height} {herbivore.base_largeness} {herbivore.speed} {herbivore.lifespan} {herbivore.gender} {herbivore.age} {herbivore.energy_needed} {herbivore.energy}\n")
        with open("carnivores.txt","w") as file_carnivores:
            for carnivore in carnivores:
                file_carnivores.write(f"{carnivore.x} {carnivore.y} {carnivore.base_height} {carnivore.base_largeness} {carnivore.speed} {carnivore.lifespan} {carnivore.gender} {carnivore.age} {carnivore.energy_needed} {carnivore.energy}\n")

    def get_data(self):
        '''It gets data from .txt files generated by "save_data".'''
        print("OBTAINING DATA...")
        old_height_map=np.loadtxt("height_map.txt")
        old_temperature_map=np.loadtxt("temperature_map.txt")
        old_humidity_map=np.loadtxt("humidity_map.txt")
        old_nutrients_map=np.loadtxt("nutrients_map.txt")    
        plants_x,plants_y,plants_height,plants_roots,plants_leaves,plants_lifespan,plants_age,plants_energy_needed,plants_energy=np.loadtxt("plants.txt",unpack=True)
        herbs_x,herbs_y,herbs_height,herbs_largeness,herbs_speed,herbs_lifespan,herbs_gender,herbs_age,herbs_energy_needed,herbs_energy=np.loadtxt("herbivores.txt",unpack=True)
        carns_x,carns_y,carns_height,carns_largeness,carns_speed,carns_lifespan,carns_gender,carns_age,carns_energy_needed,carns_energy=np.loadtxt("carnivores.txt",unpack=True)
        plants=[]
        herbivores=[]
        carnivores=[]
        print("DONE")
        print("OVERWRITING WORLD...")
        self.height_map=old_height_map
        self.temperature_map=old_temperature_map
        self.humidity_map=old_humidity_map
        self.nutrients_map=old_nutrients_map
        self.reset()
        print("DONE")
        print("OVERWRITING ORGANISMS...")
        print("PLANTS")
        for i in range(0,len(plants_x)):
            plants.append(organism.Plant(int(plants_y[i]),int(plants_x[i]),plants_height[i],plants_roots[i],plants_leaves[i],plants_lifespan[i]))
            plants[i].age=plants_age[i]
            plants[i].energy_needed=plants_energy_needed[i]
            plants[i].energy=plants_energy[i]
        print("HERBIVORES")
        for i in range(0,len(herbs_x)):  
            herbivores.append(organism.Herbivore(int(herbs_y[i]),int(herbs_x[i]),herbs_height[i],herbs_largeness[i],herbs_speed[i],herbs_lifespan[i],herbs_gender[i]))
            herbivores[i].age=herbs_age[i]
            herbivores[i].energy_needed=herbs_energy_needed[i]
            herbivores[i].energy=herbs_energy[i]
        print("CARNIVORES")
        for i in range(0,len(carns_x)):   
            carnivores.append(organism.Carnivore(int(carns_y[i]),int(carns_x[i]),carns_height[i],carns_largeness[i],carns_speed[i],carns_lifespan[i],carns_gender[i]))
            carnivores[i].age=carns_age[i]
            carnivores[i].energy_needed=carns_energy_needed[i]
            carnivores[i].energy=carns_energy[i]
        print("DONE")
        print("POSITIONING ORGANISMS...")
        fill_grid(self.bio_map,plants,herbivores,carnivores,[])
        print("DONE")
        
    def simulation_from_data(self,days=200,saving=True): #restart with the organisms alive in the previous simulation but with age 0 
        '''Starting from datas of a previous simulation it continues the simulation, but shows only the new datas. 
        It's possible to save the datas obtain in the end using saving=True.'''
        self.get_data()
        self.show_maps("height")
        #LIST TO SAVE DATA--------------------##
        #TO SAVE DAYS PASSED------------------##
        d=[]                                  ##
        #TO SAVE THE NUMBER OF ORGANISMS------##
        plants_number=[]                      ##
        herbivores_number=[]                  ##
        carnivores_number=[]                  ##
        female_herbivores_number=[]           ##
        female_carnivores_number=[]           ##
        male_herbivores_number=[]             ##
        male_carnivores_number=[]             ##
        #TO SAVE THE STATS OF PLANTS----------##
        plants_height=[]                      ##
        plants_roots=[]                       ##
        plants_leaves=[]                      ##
        plants_lifespan=[]                    ##
        #TO SAVE THE STATS OF HERBIVORES------##
        herbivores_height=[]                  ##
        herbivores_largeness=[]               ##
        herbivores_speed=[]                   ##
        herbivores_lifespan=[]                ##
        #TO SAVE THE STATS OF CARNIVORES------##
        carnivores_height=[]                  ##
        carnivores_largeness=[]               ##
        carnivores_speed=[]                   ##
        carnivores_lifespan=[]                ##
        #FOR ANIMATION------------------------##
        plants_counts=[]                      ##
        herbivores_counts=[]                  ##
        carnivores_counts=[]                  ##
        #-------------------------------------##
        print("INITIALIZING...")
        #---INITIALIZING ALL ORGANISMS------------------
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        self.update_world(plant_count,herbivore_count,carnivore_count)
        #---LET'S START THE TRUE SIMULATION-------------
        print("READY!")
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        
        for dd in range(0,days):
            print("DAY:", dd+1)
            print("UPDATING CLIME...")
            self.update_clime()
            self.refill_nutrients()
            self.refill_water()
            print("DONE")
            
            #---------------------------------------------------
            print("UPDATING ORGANISMS...")
            self.update_world(plant_count,herbivore_count,carnivore_count)
            plant_count,herbivore_count,carnivore_count=self.organism_count()
            plants,herbivores,carnivores=self.unpack_organisms()
            print("DONE")
            #---------------------------------------------------
            if len(plants)==0:
                print("All plants are dead.")
                break
            if len(herbivores)==0:
                print("All herbivores are dead.")
                break
            if len(carnivores)==0:
                print("All carnivores are dead.")
                break            
            
            #DATA COLLECTION----------------------------------------------##
            #-------------------------------------------------------------##            
            d.append(dd+1)                                                ##
            #NUMBER-------------------------------------------------------##
            plants_number.append(len(plants))                             ##
            f=0                                                           ##
            for herbivore in herbivores:                                  ##
                if herbivore.gender==0:                                   ##
                    f+=1                                                  ##
            female_herbivores_number.append(f)                            ##
            male_herbivores_number.append(len(herbivores)-f)              ##
            herbivores_number.append(len(herbivores))                     ##
            f=0                                                           ##
            for carnivore in carnivores:                                  ##
                if carnivore.gender==0:                                   ##
                    f+=1                                                  ##
            female_carnivores_number.append(f)                            ##
            male_carnivores_number.append(len(carnivores)-f)              ##
            carnivores_number.append(len(carnivores))                     ##
            #PLANTS-------------------------------------------------------##
            means=organism.mean_stats_plant(plants)                       ##
            plants_height.append(means[0])                                ##
            plants_roots.append(means[1])                                 ##
            plants_leaves.append(means[2])                                ##
            plants_lifespan.append(means[3])                              ##
            #HERBIVORES---------------------------------------------------##
            means=organism.mean_stats_animal(herbivores)                  ##
            herbivores_height.append(means[0])                            ##
            herbivores_largeness.append(means[1])                         ##
            herbivores_speed.append(means[2])                             ##
            herbivores_lifespan.append(means[3])                          ##
            #CARNIVORES---------------------------------------------------##
            means=organism.mean_stats_animal(carnivores)                  ##
            carnivores_height.append(means[0])                            ##
            carnivores_largeness.append(means[1])                         ##
            carnivores_speed.append(means[2])                             ##
            carnivores_lifespan.append(means[3])                          ##
            #-------------------------------------------------------------##
            plants_counts.append(plant_count)                             ##
            herbivores_counts.append(herbivore_count)                     ##
            carnivores_counts.append(carnivore_count)                     ##
            #-------------------------------------------------------------##
            
            print("PLANTS:", len(plants), "HERBIVORES:", len(herbivores), "CARNIVORES:", len(carnivores))
        
        if len(d)>0:    
            print("PREPARING TO SHOW RESULTS...")
            #ANIMATION--------------------------------------------------------------------##
            self.show_bio_map(plants_counts,herbivores_counts,carnivores_counts)          ##          
            #PLOTS------------------------------------------------------------------------##
            #NUMBER OF ORGANISMS----------------------------------------------------------##
            plt.plot(d,plants_number,color="forestgreen",label="Number of plants")        ##
            plt.plot(d,herbivores_number,color="cyan",label="Number of herbivores")       ##
            plt.plot(d,carnivores_number,color="crimson",label="Number or carnivores")    ##
            plt.legend()                                                                  ##
            plt.savefig("0_organisms_number.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #HERBIVORES-------------------------------------------------------------------##
            plt.title("HERBIVORES")                                                       ##
            plt.plot(d,herbivores_number,color="cyan",label="Total number")               ##
            plt.plot(d,female_herbivores_number,color="pink",label="Number of females")   ##
            plt.plot(d,male_herbivores_number,color="blue",label="Number or males")       ##
            plt.legend()                                                                  ##
            plt.savefig("0_herbivores_number.png",dpi=500)                                ##
            plt.show()                                                                    ##
            #CARNIVORES-------------------------------------------------------------------##
            plt.title("CARNIVORES")                                                       ##
            plt.plot(d,carnivores_number,color="crimson",label="Total number")            ##
            plt.plot(d,female_carnivores_number,color="pink",label="Number of females")   ##
            plt.plot(d,male_carnivores_number,color="blue",label="Number or males")       ##
            plt.legend()                                                                  ##
            plt.savefig("0_carnivores_number.png",dpi=500)                                ##
            plt.show()                                                                    ##
            #PLANTS STATS-----------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,plants_height,color="forestgreen",label="HEIGHT")             ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,plants_roots,color="forestgreen",label="ROOTS")               ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,plants_leaves,color="forestgreen",label="LEAVES")             ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,plants_lifespan,color="forestgreen",label="LIFESPAN")         ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_plants_stats.png",dpi=500)                                     ##
            plt.show()                                                                    ##
            #HERBIVORES STATS-------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,herbivores_height,color="cyan",label="HEIGHT")                ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,herbivores_largeness,color="cyan",label="LARGENESS")          ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,herbivores_speed,color="cyan",label="SPEED")                  ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,herbivores_lifespan,color="cyan",label="LIFESPAN")            ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_herbivores_stats.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #CARNIVORES STATS-------------------------------------------------------------##
            fig,ax=plt.subplots(nrows=2,ncols=2)                                          ##
            ax[0][0].plot(d,carnivores_height,color="crimson",label="HEIGHT")             ##
            ax[0][0].legend()                                                             ##
            ax[0][1].plot(d,carnivores_largeness,color="crimson",label="LARGENESS")       ##
            ax[0][1].legend()                                                             ##
            ax[1][0].plot(d,carnivores_speed,color="crimson",label="SPEED")               ##
            ax[1][0].legend()                                                             ##
            ax[1][1].plot(d,carnivores_lifespan,color="crimson",label="LIFESPAN")         ##
            ax[1][1].legend()                                                             ##
            plt.savefig("0_carnivores_stats.png",dpi=500)                                 ##
            plt.show()                                                                    ##
            #-----------------------------------------------------------------------------##
            print("DONE")
        else:
            print("RESULTS NOT VALID")
        if saving==True:
            self.save_data()
    
    def simulation_with_steps(self,number_of_plants=20000,number_of_herbivore=5000,number_of_carnivore=2000,days_more=10):
        '''This simulation introduces organisms in stages. Firstly it introduces the plants. 
        When the population is 1.5 times the starting population it introduces the herbivores.
        When the population of herbivores is 1.5 times the starting one it itnroduces carnivores.
        Then simulates for more days defined by "days_more", fixed at 10. 
        At the end saves the results without showing so that the simulation can start using "simulation_from_data".'''
        #create the organisms
        plants=random_plants(number_of_plants,self.height_map)
        print("OBSERVING PLANTS...")
        dd=0
        len_plant=len(plants)
        fill_grid(self.bio_map,plants,[],[],[])
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        while len(plants)<1.5*len_plant:
            dd+=1
            print("DAY:", dd)
            print("UPDATING CLIME...")
            self.update_clime()
            self.refill_nutrients()
            self.refill_water()
            print("DONE")
            #---------------------------------------------------
            print("UPDATING PLANTS...")
            self.update_world(plant_count,herbivore_count,carnivore_count)
            plant_count,herbivore_count,carnivore_count=self.organism_count()
            plants,herbivores,carnivores=self.unpack_organisms()
            print("DONE")
            print("PLANTS:", len(plants))
        print("WORLD READY FOR HERBIVORES")
        herbivores=random_herbivores(number_of_herbivore,plant_count)
        print("OBSERVING...")
        len_herbivores=len(herbivores)
        fill_grid(self.bio_map,plants,herbivores,[],[])
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        while len(herbivores)<1.5*len_herbivores:
            dd+=1
            print("DAY:", dd)
            print("UPDATING CLIME...")
            self.update_clime()
            self.refill_nutrients()
            self.refill_water()
            print("DONE")
            #---------------------------------------------------
            print("UPDATING ORGANISMS...")
            self.update_world(plant_count,herbivore_count,carnivore_count)
            plant_count,herbivore_count,carnivore_count=self.organism_count()
            plants,herbivores,carnivores=self.unpack_organisms()
            print("DONE")
            print("PLANTS:", len(plants), "HERBIVORES:", len(herbivores))
        print("WORLD READY FOR CARNIVORES")
        carnivores=random_carnivores(number_of_carnivore,herbivore_count)
        len_herbivores=len(herbivores)
        fill_grid(self.bio_map,plants,herbivores,carnivores,[])
        plant_count,herbivore_count,carnivore_count=self.organism_count()
        for i in range(0,days_more):
            dd+=1
            print("DAY:", dd)
            print("UPDATING CLIME...")
            self.update_clime()
            self.refill_nutrients()
            self.refill_water()
            print("DONE")
            #---------------------------------------------------
            print("UPDATING ORGANISMS...")
            self.update_world(plant_count,herbivore_count,carnivore_count)
            plant_count,herbivore_count,carnivore_count=self.organism_count()
            plants,herbivores,carnivores=self.unpack_organisms()
            print("DONE")
            print("PLANTS:", len(plants), "HERBIVORES:", len(herbivores), "CARNIVORES:", len(carnivores))
        self.save_data()

def main():        
    '''Example of use.'''
    world=World()
    world.simulation_with_steps()
    world.simulation_from_data()
    
if __name__=="__main__":
    main()
    
