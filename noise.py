import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from numba import njit

#--------------BASICS----------------

@njit
def gaussian(x,y):
    return np.exp(-(x**2+y**2))
    
@njit
def randvec():
    theta=random.random()*2*np.pi
    return np.array([np.cos(theta),np.sin(theta)],dtype=np.float64)
    
@njit
def fade(t):
    return 6*t**5-15*t**4+10*t**3

@njit    
def lerp(a,b,t):
    return a+t*(b-a)

@njit
def sigmoid(x,k=7):
    return 1/(1+np.exp(-k*x))
    
@njit
def dot_product_2D(u,v):
    return u[0]*v[0]+u[1]*v[1]

#--------------NOISE MAP-------------

@njit
def noise(length,height,step):
    start=np.empty((height//step+1,length//step+1,2),dtype=np.float64)
    noise_map=np.zeros((height,length))
    
    for i in range(0,length//step+1):
        for j in range(0,height//step+1):
            start[j][i]=randvec()
    
    for i in range(0,length):
        for j in range(0,height):
            up_left=start[j//step][i//step]
            up_right=start[j//step][i//step+1]
            down_left=start[j//step+1][i//step]
            down_right=start[j//step+1][i//step+1]
            
            dist_x=i/step-i//step
            dist_y=j/step-j//step
            u_x=fade(dist_x)
            u_y=fade(dist_y)
            
            v00=dot_product_2D(up_left,(dist_x,dist_y))
            v01=dot_product_2D(up_right,((1-dist_x),dist_y))
            v10=dot_product_2D(down_left,(dist_x,(1-dist_y)))
            v11=dot_product_2D(down_right,((1-dist_x),(1-dist_y)))
            
            noise_map[j][i]=lerp(lerp(v00,v10,u_y),lerp(v01,v11,u_y),u_x)
    return sigmoid(noise_map)

#--------------NOISE MAP WITH LAYERS-

@njit        
def layermap(length,height,steps,weights,mask=True,weight_gaussian=2):
    layer_noise_map=np.zeros((height,length))
    for i in range(0,len(steps)):
        layer_noise_map+=noise(length,height,steps[i])*weights[i]
    if mask==True:
        gaussian_map=np.zeros((height,length))
        for i in range(0,length):
            for j in range(0,height):
                gaussian_map[j][i]=gaussian(3*(j-height/2)/height,3*(i-length/2)/length)
        return (layer_noise_map+weight_gaussian*gaussian_map)/(sum(weights)+weight_gaussian)
    return layer_noise_map/sum(weights)


#--------------HEIGHT MAP------------

def generate_height_map(length=210,height=210,steps=[6,10,14,35,21,15,7],weights=[1,2,3,4,3,2,1]):
    if len(steps)!=len(weights):
        raise ValueError("Steps and weights must have same length.")
    return layermap(length,height,steps,weights)

def print_height_map(noise_map):
    colors=["black","blue","lightseagreen","cyan","green","yellowgreen","saddlebrown","white"]
    tm=color.LinearSegmentedColormap.from_list("my_terrain",colors,N=500)
    plt.imshow(noise_map,cmap=tm,vmin=0,vmax=1,interpolation='bilinear')
    plt.colorbar()
    plt.savefig("map.png",dpi=500)
    plt.show()

#--------------TEMPERATURE MAP-------

def generate_temp_map(height_map,steps=[35,21,15,14],weights=[2,2,1,1]):
    temp_height=np.zeros((len(height_map),len(height_map[0])))
    for i in range(0,len(height_map)):
        for j in range(0,(len(height_map[0]))):
            if height_map[i][j]<0.5:
                temp_height[i][j]=20*(np.exp(-0.5+height_map[i][j])-1)
            else:
                temp_height[i][j]=-30*(height_map[i][j]-0.5)**1.5
    return 20+temp_height+(layermap(len(height_map[0]),len(height_map),steps,weights)-0.5)*20
    
def print_temp_map(noise_map):
    plt.imshow(noise_map,cmap='coolwarm',interpolation='bilinear')
    plt.colorbar()
    plt.savefig("map_temperature.png",dpi=500)
    plt.show()

#--------------HUMIDITY MAP----------

def generate_hum_map(height_map,temp_map,steps=[15,21,35],weights=[1,1,1]):
    if len(steps)!=len(weights):
        raise ValueError("Steps and weights must have same length.")
    hum_map=np.zeros((len(height_map),len(height_map[0])))
    for i in range(0,len(height_map)):
        for j in range(0,len(height_map[0])):
            if height_map[i][j]<0.5 or temp_map[i][j]<0:
                hum_map[i][j]=1
            else:
                hum_map[i][j]=0.5*(np.exp(0.5-height_map[i][j])+np.exp(-temp_map[i][j]/20))
    hum_map+=layermap(len(height_map[0]),len(height_map),steps,weights,mask=False)
    return hum_map/2
    
def print_hum_map(noise_map):
    plt.imshow(noise_map,cmap='gray',interpolation='bilinear')
    plt.colorbar()
    plt.savefig("map_humidity.png",dpi=500)
    plt.show()

#--------------NUTRIENTS MAP---------

def generate_nut_map(height_map,temp_map,hum_map,steps=[7,10,14],weights=[1,1,1]):
    if len(steps)!=len(weights):
        raise ValueError("Steps and weights must have same length.")
    nut_map=np.zeros((len(height_map),len(height_map[0])))
    for i in range(0,len(height_map)):
        for j in range(0,len(height_map[0])):
            nut_map[i][j]=hum_map[i][j]*(np.exp(-(temp_map[i][j]-20)**2/144)+np.exp(-(height_map[i][j]-0.5)**2/1.6))/2
    nut_map+=layermap(len(height_map[0]),len(height_map),steps,weights,mask=False)
    return nut_map/2
    
def print_nut_map(noise_map):
    plt.imshow(noise_map,cmap='copper',interpolation='bilinear')
    plt.colorbar()
    plt.savefig("map_nutrients.png",dpi=500)
    plt.show()

#--------------COMPILING-------------
gaussian(0,0)
randvec()
fade(0)
lerp(0,0,0)
sigmoid(0)
dot_product_2D(np.array([0,0]),np.array([0,0]))
noise(2,2,1)
layermap(2,2,[1],[1])

#--------------TRY-------------------
    
def main():
    height_map=generate_height_map()
    print_height_map(height_map)
    temp_map=generate_temp_map(height_map)
    print_temp_map(temp_map)
    hum_map=generate_hum_map(height_map,temp_map)
    print_hum_map(hum_map)
    nut_map=generate_nut_map(height_map,temp_map,hum_map)
    print_nut_map(nut_map)
    
if __name__=="__main__":
    main()