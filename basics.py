# J -> JIT compilation (Just In Time) 
# A -> Automatic Differenciation 
# X -> XLA (Accelrated Linear Algebra) 


# JAX as NumPy

import jax 
import jax.numpy as jnp 
#a = jnp.array([1,2,3]) 
#b = jnp.array([4,5,6]) 
#we can use all jnp as numpy. all functionality of numpy can be done here
#print("Array a = ", a ) 
#print("Array b = ", b ) 
#print("sum or arrays : ", (a+b))
#print("square root of elemes of a:  ", jnp.sqrt(a))
#print("mean of array a: ", jnp.mean(a))
#print("reshaping array in to (-1,1): " ,a.reshape(-1,1))

#JAX arrays are immuatable and do not support item assign as numpy




# JIT Compilation 
import time 

#@jax.jit  #This allows fast compilation 
#def myfunction(x) : 
#    return jnp.where(x%2 ==0 , x/2, 3*x + 1) #Collatz 

#arr = jnp.arange(10) 

#_ = myfunction(arr) # warm up (if function is already called then recalling it will take very less time 

#start = time.perf_counter()
#myfunction(arr).block_until_ready()             #result is not evaluted until necessary
#myfunction(arr)
#end = time.perf_counter()
#print(end-start)


#print(jax.make_jaxpr(myfunction)(arr) )  #show intermediate representation of jax uses for compilation 


#we can't use @jax.jit on any function 




# Automatic Differenciation 

#allows the derivative of function easily and automatically 

#def square(x): 
#    return x**2 

#f(x) = x^2 
#f'(x) = 2x 
#f''(x) = 2 
#f'''(x) = 0 

#value = 10.0
#print(square(value))
#print(jax.grad(square)(value))
#print(jax.grad(jax.grad(square))(value))
#print(jax.grad(jax.grad(jax.grad(square)))(value))


#jax also allows partial derivatives

#def f(x,y,z) : 
#    return  (x**2 + 2*y**2 + 3*z**2)

def f(arr): 
    return arr[0] ** 2 + 2*arr[1]**2 + 3*arr[2]**2

x, y, z = 2.0, 2.0, 2.0


# x^2 + 2y^2 + 3z^2
# df/dx = 2x = 4 
# df/dy = 4y = 8
# df/dz = 6z = 12

#print(f([x,y,z]))
#print(jax.grad(f,argnums=0 )([x,y,z]))
#print(jax.grad(f,argnums=1 )([x,y,z]))
#print(jax.grad(f,argnums=2 )([x,y,z]))

print(jax.grad(f)([x,y,z]))



#Automatic Vectorization
key  =  jax.random.key(42) #42 is seed 

W = jax.random.normal(key, (150,100)) # 100 values per input sample, 150 neurons in next layer
X = jax.random.normal(key, (10,100)) 

def calculte_output(x): 
    return jnp.dot(W,x)

def batched_calculation_loop(X):
    return jnp.stack([calculte_output(x) for x in X ])

def batched_calculation_manual(X): 
    return jnp.dot(X, W.T)

batched_calculation_vmap = jax.vmap(calculte_output)

start = time.perf_counter()
batched_calculation_loop(X)
end = time.perf_counter()
print(end-start)



start = time.perf_counter()
batched_calculation_manual(X)
end = time.perf_counter()
print(end-start)



start = time.perf_counter()
batched_calculation_vmap(X) 
end = time.perf_counter()
print(end-start)





# Randomness 

# work with some kind of keys not seeds for randomnss

key = jax.random.key(42) #42 is seed once key is used it is never used again

jax.random.normal(key)

#key1, key2 = jax.random.split(key)  #generate two different keys using key
#key3, key4 = jax.random.split(key1)

keys = jax.random.split(key, 10) #gives 10 keys 

print(keys)























