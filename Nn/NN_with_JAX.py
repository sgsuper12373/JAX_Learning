import jax 
import jax.numpy as jnp 
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = load_iris()
X = data.data
y = data.target.reshape(-1,1)


encoder = OneHotEncoder(sparse_output = False) 

y = encoder.fit_transform(y) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test) 

#print(X_test) 
#print(y)

#input -> hidden layer1 -> hidden layer 2 -> output
def init_params(input_dim, hidden_dim1 , hidden_dim2, output_dim, random_key):
    random_keys = jax.random.split(random_key, 3) 

    W1 = jax.random.normal(random_keys[0], (input_dim,hidden_dim1) )
    b1 = jnp.zeros((hidden_dim1)) 

    W2 = jax.random.normal(random_keys[0], (hidden_dim1, hidden_dim2) )
    b2 = jnp.zeros((hidden_dim2)) 

    W3 = jax.random.normal(random_keys[0], (hidden_dim2, output_dim) )
    b3 = jnp.zeros((output_dim)) 

    return W1,b1, W2,b2,W3,b3 

def forward(params, X): 
    W1, b1, W2, b2, W3, b3 = params

    h1 = jax.nn.relu(jnp.dot(X,W1) + b1) #relu is function such that if x<0 then output is 0 else output=x 
    #relu(x): x<0? 0 : x 

    h2 = jax.nn.relu(jnp.dot(h1,W2) +b2) 
    logits = jnp.dot(h2,W3) + b3    #we don't relu here because this are logits which are going to passed to softmax function 

    return logits   



def loss_fn(params, X,y, l2_reg = 0.0001): 
    logits = forward(params, X)
    probs = jax.nn.softmax(logits) 
    l2_loss = l2_reg *sum ([jnp.sum(w**2) for w in params[::2]]) 

    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-8), axis=1))

@jax.jit
def train_step(params, X, y, lr): 
    grads = jax.grad(loss_fn)(params, X, y)
    return [(param - lr*grad) for param,  grad in zip(params,grads)]

def accuracy(params, X, y) :
    preds = jnp.argmax(forward(params, X), axis = 1) 
    targets = jnp.argmax(y,axis =1) 
    return jnp.mean(preds==targets)

def data_loader(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

random_key = jax.random.key(int(time.time()) )
input_dim = X.shape[1]
hidden_dim1 = 16 
hidden_dim2 = 8 
output_dim = y.shape[1]
learning_rate = 0.01
batch_size = 16
epochs = 1000

params = init_params(input_dim, hidden_dim1, hidden_dim2, output_dim, random_key)

for epoch in range(epochs):
    for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
        params = train_step(params, X_batch, y_batch, learning_rate)
    
    if epoch % 10 == 0:
        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)
        # train_loss = loss_fn(params, X_train, y_train)
        # test_loss = loss_fn(params, X_test, y_test)
        print(f"Epoch {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
print(f"Final Test acc : {accuracy(params, X_test, y_test):.4f}")

