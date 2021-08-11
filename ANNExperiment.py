# Setup
import ctypes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from keras import backend as K
from keras.models import Model
from keras.engine.input_layer import Input
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop
from keras.models import load_model

def random_batch(X_train, y_train, batch_size):
    index_set = np.random.randint(0, X_train.shape[0], batch_size)
    X_batch = X_train[index_set]
    y_batch = y_train[index_set]
    return X_batch, y_batch

# Start with 16, 32, 64, 128, 256, (512?) key,message,cipher sizes

# Symmetric (secret-key) encryption
## Model `crypto1` (Google)

## This model was build according to the specifications from Google's paper *Learning to protect communications with adversarial neural cryptography*.

model_name = 'crypto1'

# Set up the crypto parameters: message, key, and ciphertext bit lengths
m_bits = 32
k_bits = 32
c_bits = 32
pad = 'same' #cnn param

# Compute the size of the message space
m_train = 2**(m_bits) #+ k_bits)
# maybe make this fixed to 100,000

alice_file = './' + model_name + '-alice'
bob_file = './' + model_name + '-bob'
eve_file = './' + model_name + '-eve'

### Network arch

K.clear_session()

##### Alice network #####
#
ainput0 = Input(shape=(m_bits,)) #message
ainput1 = Input(shape=(k_bits,)) #key
ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + k_bits))(ainput)
adense1a = Activation('tanh')(adense1)
areshape = Reshape((m_bits + k_bits, 1,))(adense1a)

aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(areshape)
aconv1a = Activation('tanh')(aconv1)
aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(aconv1a)
aconv2a = Activation('tanh')(aconv2)
aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(aconv2a)
aconv3a = Activation('tanh')(aconv3)
aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(aconv3a)
aconv4a = Activation('sigmoid')(aconv4)

aoutput = Flatten()(aconv4a)

alice = Model([ainput0, ainput1], aoutput, name='alice')
# alice.summary()


##### Bob network #####
#
binput0 = Input(shape=(c_bits,)) #ciphertext
binput1 = Input(shape=(k_bits,)) #key
binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=(c_bits + k_bits))(binput)
bdense1a = Activation('tanh')(bdense1)

breshape = Reshape((c_bits + k_bits, 1,))(bdense1a)

bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(breshape)
bconv1a = Activation('tanh')(bconv1)
bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(bconv1a)
bconv2a = Activation('tanh')(bconv2)
bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(bconv2a)
bconv3a = Activation('tanh')(bconv3)
bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(bconv3a)
bconv4a = Activation('sigmoid')(bconv4)

boutput = Flatten()(bconv4a)

bob = Model([binput0, binput1], boutput, name='bob')
# bob.summary()


# Eve network
#
einput = Input(shape=(c_bits,)) #ciphertext only

edense1 = Dense(units=(c_bits + k_bits))(einput)
edense1a = Activation('tanh')(edense1)

edense2 = Dense(units=(c_bits + k_bits))(edense1a)
edense2a = Activation('tanh')(edense2)

ereshape = Reshape((c_bits + k_bits, 1,))(edense2a)

econv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(ereshape)
econv1a = Activation('tanh')(econv1)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(econv1a)
econv2a = Activation('tanh')(econv2)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(econv2a)
econv3a = Activation('tanh')(econv3)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(econv3a)
econv4a = Activation('sigmoid')(econv4)

eoutput = Flatten()(econv4a)# Eve's attempt at code guessing

eve = Model(einput, eoutput, name='eve')
# eve.summary()

alice.compile(loss='mse', optimizer='sgd') #adam, or rmsprop optimizers
bob.compile(loss='mse', optimizer='sgd')
eve.compile(loss='mse', optimizer='sgd')

if False:
    alice.summary()
    bob.summary()
    eve.summary()

### Loss + Optimizer

# Establish the communication channels by linking inputs to outputs
#
aliceout = alice([ainput0, ainput1])
bobout = bob( [aliceout, binput1] )# bob sees ciphertext AND key
eveout = eve( aliceout )# eve doesn't see the key, only the cipher

# Loss for Eve is just L1 distance between ainput0 and eoutput. The sum
# is taken over all the bits in the message. The quantity inside the K.mean()
# is per-example loss. We take the average across the entire mini-batch
#
eveloss = K.mean(  K.sum(K.abs(ainput0 - eveout), axis=-1)  )

# Loss for Alice-Bob communication depends on Bob's reconstruction, but
# also on Eve's ability to decrypt the message. Eve should do no better
# than random guessing, so on average she will guess half the bits right.
#
bobloss = K.mean(  K.sum(K.abs(ainput0 - bobout), axis=-1)  )
abeloss = bobloss + K.square(m_bits/2 - eveloss)/( (m_bits//2)**2 )  #careful

# Optimizer and compilation
#
abeoptim = RMSprop(lr=0.001)
eveoptim = RMSprop(lr=0.001) #default 0.001


# Build and compile the ABE model, used for training Alice-Bob networks
#
abemodel = Model([ainput0, ainput1, binput1], bobout, name='abemodel')
abemodel.add_loss(abeloss)
abemodel.compile(optimizer=abeoptim)


# Build and compile the EVE model, used for training Eve net (with Alice frozen)
#
alice.trainable = False
evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
evemodel.add_loss(eveloss)
evemodel.compile(optimizer=eveoptim)

### Train / save / restore

abelosses = []
boblosses = []
evelosses = []

n_epochs = 20 #can be optimized
batch_size = 512  #can be optimized
n_batches = m_train // batch_size

abecycles = 1
evecycles = 2 # ideally one

epoch = 0
print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

while epoch < n_epochs:
    abelosses0 = []
    boblosses0 = []
    evelosses0 = []
    for iteration in range(n_batches):

        # Train the A-B+E network
        #
        alice.trainable = True
        for cycle in range(abecycles):
            # Select a random batch of messages, and a random batch of keys
            #
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = abemodel.train_on_batch([m_batch, k_batch, k_batch], None)

        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        loss = np.mean(  np.sum( np.abs(m_batch - m_dec), axis=-1)  )
        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        #
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = evemodel.train_on_batch([m_batch, k_batch], None)

        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    print()
    epoch += 1

print('Training finished.')

steps = -1

plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

#plt.savefig("images/" + model_name + ".png", transparent=True) #dpi=100
plt.show()

if True: #Don't overwrite by accident
    alice.save(alice_file + '.h5', overwrite=True)
    bob.save(bob_file + '.h5', overwrite=True)
    eve.save(eve_file + '.h5', overwrite=True)

alice = load_model(alice_file + '.h5')
bob = load_model(bob_file + '.h5')
eve = load_model(eve_file + '.h5')

n_examples = 10000

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

m_enc = alice.predict([m_batch, k_batch])
m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
m_att = (eve.predict(m_enc) > 0.5).astype(int)

bdiff = np.abs(m_batch - m_dec)
bsum = np.sum(bdiff, axis=-1)
ediff = np.abs(m_batch - m_att)
esum = np.sum(ediff, axis=-1)

print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')

### Evaluate
n_examples = 10000

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

m_enc = alice.predict([m_batch, k_batch])
m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
m_att = (eve.predict(m_enc) > 0.5).astype(int)

bdiff = np.abs(m_batch - m_dec)
bsum = np.sum(bdiff, axis=-1)
ediff = np.abs(m_batch - m_att)
esum = np.sum(ediff, axis=-1)

print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')

### Freeze Alice-Bob
if False:
    alice = load_model(alice_file + '.h5')
    bob = load_model(bob_file + '.h5')
    eve = load_model(eve_file + '.h5')

aliceout = alice([ainput0, ainput1])
bobout = bob( [aliceout, binput1] )# bob sees ciphertext AND key
eveout = eve( aliceout )# eve doesn't see the key, only the cipher

eveloss = K.mean(  K.sum(K.abs(ainput0 - eveout), axis=-1)  )
bobloss = K.mean(  K.sum(K.abs(ainput0 - bobout), axis=-1)  )
abeloss = bobloss + K.square(m_bits/2 - eveloss)/( (m_bits//2)**2 )

abeoptim = RMSprop(lr=0.001)
eveoptim = Adam()#RMSprop(lr=0.001) #default 0.001

abemodel = Model([ainput0, ainput1, binput1], bobout, name='abemodel')
abemodel.add_loss(abeloss)
abemodel.compile(optimizer=abeoptim)

alice.trainable = False
evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
evemodel.add_loss(eveloss)
evemodel.compile(optimizer=eveoptim)

abelosses = []
boblosses = []
evelosses = []

n_epochs = 20
batch_size = 512
n_batches = m_train // batch_size

epoch = 0
print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

while epoch < n_epochs:
    abelosses0 = []
    boblosses0 = []
    evelosses0 = []
    for iteration in range(n_batches):
        # Train Eve model only
        #
        alice.trainable = False
        m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
        k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
        eveloss = evemodel.train_on_batch([m_batch, k_batch], None)
        evelosses0.append(eveloss)
        evelosses.append(eveloss)
        eveavg = np.mean(evelosses0)

        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        bobloss = np.mean(  np.sum( np.abs(m_batch - m_dec), axis=-1)  )
        boblosses0.append(bobloss)
        boblosses.append(bobloss)
        bobavg = np.mean(boblosses0)

        # Evaluate the ABE loss
        abeloss = bobloss + ((m_bits/2 - eveloss)**2) / ( (m_bits//2)**2 )
        abelosses0.append(abeloss)
        abelosses.append(abeloss)
        abeavg = np.mean(abelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    print()
    epoch += 1

print('Training finished.')

steps = -1

plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

#plt.savefig("images/" + model_name + "-eve1.png", transparent=True) #dpi=100
plt.show()

n_examples = 10000

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

m_enc = alice.predict([m_batch, k_batch])
m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
m_att = (eve.predict(m_enc) > 0.5).astype(int)

bdiff = np.abs(m_batch - m_dec)
bsum = np.sum(bdiff, axis=-1)
ediff = np.abs(m_batch - m_att)
esum = np.sum(ediff, axis=-1)

print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')

### Encoding distribution
# Let's plot a few of the encoded vectors' coodinates:

n_examples = 10000
showAll = True

coord_indeces = np.array([
    [ 0, 1, 2, 4],
    [ 5, 6, 7, 14]
])

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
m_enc = alice.predict([m_batch, k_batch])

if showAll:
    n_cols = 4
    n_rows = m_enc.shape[1] // n_cols
else:
    n_cols = coord_indeces.shape[1]
    n_rows = coord_indeces.shape[0]

plt.figure(figsize=(8, int(8.0/n_cols * n_rows)))
for row in range(n_rows):
    for col in range(n_cols):
        i = row * n_cols + col
        plt.subplot(n_rows, n_cols, i + 1)
        if showAll:
            plt.title("Coord " + str(i), fontsize=14)
            plt.hist(m_enc[:, i], bins=20, density=True)
        else:
            plt.title("Coord " + str(coord_indeces[row, col]), fontsize=12)
            plt.hist(m_enc[:, coord_indeces[row, col]], bins=20, density=True)
plt.tight_layout()
#plt.savefig("images/" + model_name + "-encall.png", transparent=True) #dpi=100
plt.show()

# Let's examine various correlations, if any
data_arr = np.c_[m_batch, k_batch, m_enc]

columns = [
    'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15',
    'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15',
    'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
]

data = pd.DataFrame(data=data_arr, index=range(10000), columns=columns)
data.head()

datac = data[['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15']]
datac.head()

corr = data.corr()
corrc = datac.corr()
corrc['c2'].sort_values(ascending=False)
pd.plotting.scatter_matrix(datac, alpha=0.2, figsize=(12,12))
plt.show()

## Model `crypto2`

# Add dense, and allow tanh for codings.

K.floatx()
model_name = 'crypto2'

# Set up the crypto parameters: message, key, and ciphertext bit lengths
m_bits = 16
k_bits = 16
c_bits = 16
pad = 'same'

# Compute the size of the message space
m_train = 2**(m_bits) # + k_bits)

alice_file = './' + model_name + '-alice'
bob_file = './' + model_name + '-bob'
eve_file = './' + model_name + '-eve'

### Network arch

K.clear_session()
kersize = 4

##### Alice network #####
#
ainput0 = Input(shape=(m_bits,)) #message
ainput1 = Input(shape=(k_bits,)) #key
ainput = concatenate([ainput0, ainput1], axis=1)

adense1 = Dense(units=(m_bits + k_bits))(ainput)
adense1a = Activation('tanh')(adense1)

areshape = Reshape((m_bits + k_bits, 1,))(adense1a)

aconv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(areshape)
aconv1a = Activation('tanh')(aconv1)
aconv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv1a)
aconv2a = Activation('tanh')(aconv2)
aconv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv2a)
aconv3a = Activation('tanh')(aconv3)
aconv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv3a)
aconv4a = Activation('tanh')(aconv4)

aflat = Flatten()(aconv4a)
aoutput = Dense(units=c_bits, activation='tanh')(aflat) #ciphertext

alice = Model([ainput0, ainput1], aoutput, name='alice')
#alice.summary()


##### Bob network #####
#
binput0 = Input(shape=(c_bits,)) #ciphertext
binput1 = Input(shape=(k_bits,)) #key
binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=(c_bits + k_bits))(binput)
bdense1a = Activation('tanh')(bdense1)

breshape = Reshape((c_bits + k_bits, 1,))(bdense1a)

bconv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(breshape)
bconv1a = Activation('tanh')(bconv1)
bconv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv1a)
bconv2a = Activation('tanh')(bconv2)
bconv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv2a)
bconv3a = Activation('tanh')(bconv3)
bconv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv3a)
bconv4a = Activation('tanh')(bconv4)

bflat = Flatten()(bconv4a)
boutput = Dense(units=m_bits, activation='sigmoid')(bflat) #decrypted message

bob = Model([binput0, binput1], boutput, name='bob')
#bob.summary()


# Eve network
#
einput = Input(shape=(c_bits,)) #ciphertext only

edense1 = Dense(units=(c_bits + k_bits))(einput)
edense1a = Activation('tanh')(edense1)
edense2 = Dense(units=(m_bits + k_bits))(edense1a)
edense2a = Activation('tanh')(edense2)

ereshape = Reshape((m_bits + k_bits, 1,))(edense2a)

econv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(ereshape)
econv1a = Activation('tanh')(econv1)
econv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv1a)
econv2a = Activation('tanh')(econv2)
econv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv2a)
econv3a = Activation('tanh')(econv3)
econv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv3a)
econv4a = Activation('tanh')(econv4)

eflat = Flatten()(econv4a)
eoutput = Dense(units=m_bits, activation='sigmoid')(eflat) #code break attempt

eve = Model(einput, eoutput, name='eve')
#eve.summary()

alice.compile(loss='mse', optimizer='sgd')
bob.compile(loss='mse', optimizer='sgd')
eve.compile(loss='mse', optimizer='sgd')

if False:
    alice.summary()
    bob.summary()
    eve.summary()

### Loss + Optimizer

# Establish the communication channels by linking inputs to outputs
#
aliceout = alice([ainput0, ainput1])
bobout = bob( [aliceout, binput1] )# bob sees ciphertext AND key
eveout = eve( aliceout )# eve doesn't see the key, only the cipher

# Loss for Eve is just L1 distance between ainput0 and eoutput. The sum
# is taken over all the bits in the message. The quantity inside the K.mean()
# is per-example loss. We take the average across the entire mini-batch
#
eveloss = K.mean(  K.sum(K.abs(ainput0 - eveout), axis=-1)  )

# Loss for Alice-Bob communication depends on Bob's reconstruction, but
# also on Eve's ability to decrypt the message. Eve should do no better
# than random guessing, so on average she will guess half the bits right.
#
bobloss = K.mean(  K.sum(K.abs(ainput0 - bobout), axis=-1)  )
abeloss = bobloss + K.square(m_bits/2 - eveloss)/( (m_bits//2)**2 )

# Optimizer and compilation
#
abeoptim = Adam()#RMSprop(lr=0.0015)
eveoptim = Adam()#RMSprop(lr=0.0015) #default 0.001


# Build and compile the ABE model, used for training Alice-Bob networks
#
abemodel = Model([ainput0, ainput1, binput1], bobout, name='abemodel')
abemodel.add_loss(abeloss)
abemodel.compile(optimizer=abeoptim)


# Build and compile the EVE model, used for training Eve net (with Alice frozen)
#
alice.trainable = False
evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
evemodel.add_loss(eveloss)
evemodel.compile(optimizer=eveoptim)

### Train / save / restore

# Keep track of loss at every iteration for the final graph
abelosses = []
boblosses = []
evelosses = []

n_epochs = 30
batch_size = 256
n_batches = m_train // batch_size

abecycles = 1
evecycles = 2

epoch = 0
print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

while epoch < n_epochs:
    abelosses0 = [] #epoch-bound losses for text display during training
    boblosses0 = []
    evelosses0 = []
    for iteration in range(n_batches):

        # Train the A-B+E network
        #
        alice.trainable = True
        for cycle in range(abecycles):
            # Select a random batch of messages, and a random batch of keys
            #
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = abemodel.train_on_batch([m_batch, k_batch, k_batch], None)

        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        loss = np.mean(  np.sum( np.abs(m_batch - m_dec), axis=-1)  )
        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        #
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = evemodel.train_on_batch([m_batch, k_batch], None)

        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    print()
    epoch += 1

print('Training finished.')

steps = -1

plt.figure(figsize=(10, 6))
plt.plot(abelosses[:steps], label='$L_{Alice-Bob}$', alpha=0.85, color="#648fff")
plt.plot(evelosses[:steps], label='$L_{Eve}$', alpha=0.85, color="#dc267f")
plt.plot(boblosses[:steps], label='$L_{Bob}$', alpha=0.85, color="#785ef0")
plt.xlabel("Training Steps", fontsize=13)
plt.ylabel("Losses", fontsize=13)
plt.legend(fontsize=13, loc='upper right')

plt.savefig("./" + model_name + "-all.png", bbox_inches='tight', dpi=350)
plt.show()

alice.save(alice_file + '.h5', overwrite=True)
bob.save(bob_file + '.h5', overwrite=True)
eve.save(eve_file + '.h5', overwrite=True)

alice = load_model(alice_file + '.h5')
bob = load_model(bob_file + '.h5')
eve = load_model(eve_file + '.h5')

### Evaluate

n_examples = 10000

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

m_enc = alice.predict([m_batch, k_batch])
#m_enc = np.round(m_enc, 3)
m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
m_att = (eve.predict(m_enc) > 0.5).astype(int)

bdiff = np.abs(m_batch - m_dec)
bsum = np.sum(bdiff, axis=-1)
ediff = np.abs(m_batch - m_att)
esum = np.sum(ediff, axis=-1)

print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')

#New Experiment - PR

abelosses_pr = {}
evelosses_pr = {}
boblosses_pr = {}
bob_correct = []
eve_correct = []
best_model = 200.0

for pr in range(30):

  K.floatx()

  model_name = 'crypto2'

  # Set up the crypto parameters: message, key, and ciphertext bit lengths
  m_bits = 512
  k_bits = 512
  c_bits = 512
  pad = 'same'

  # Compute the size of the message space
  m_train = 50000 #2**(m_bits) # + k_bits)

  alice_file = './' + model_name + '-alice'
  bob_file = './' + model_name + '-bob'
  eve_file = './' + model_name + '-eve'

  K.clear_session()
  kersize = 4

  ##### Alice network #####
  #
  ainput0 = Input(shape=(m_bits,)) #message
  ainput1 = Input(shape=(k_bits,)) #key
  ainput = concatenate([ainput0, ainput1], axis=1)

  adense1 = Dense(units=(m_bits + k_bits))(ainput)
  adense1a = Activation('tanh')(adense1)

  areshape = Reshape((m_bits + k_bits, 1,))(adense1a)

  aconv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(areshape)
  aconv1a = Activation('tanh')(aconv1)
  aconv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv1a)
  aconv2a = Activation('tanh')(aconv2)
  aconv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv2a)
  aconv3a = Activation('tanh')(aconv3)
  aconv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(aconv3a)
  aconv4a = Activation('tanh')(aconv4)

  aflat = Flatten()(aconv4a)
  aoutput = Dense(units=c_bits, activation='tanh')(aflat) #ciphertext

  alice = Model([ainput0, ainput1], aoutput, name='alice')
  #alice.summary()


  ##### Bob network #####
  #
  binput0 = Input(shape=(c_bits,)) #ciphertext
  binput1 = Input(shape=(k_bits,)) #key
  binput = concatenate([binput0, binput1], axis=1)

  bdense1 = Dense(units=(c_bits + k_bits))(binput)
  bdense1a = Activation('tanh')(bdense1)

  breshape = Reshape((c_bits + k_bits, 1,))(bdense1a)

  bconv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(breshape)
  bconv1a = Activation('tanh')(bconv1)
  bconv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv1a)
  bconv2a = Activation('tanh')(bconv2)
  bconv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv2a)
  bconv3a = Activation('tanh')(bconv3)
  bconv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(bconv3a)
  bconv4a = Activation('tanh')(bconv4)

  bflat = Flatten()(bconv4a)
  boutput = Dense(units=m_bits, activation='sigmoid')(bflat) #decrypted message

  bob = Model([binput0, binput1], boutput, name='bob')
  #bob.summary()


  # Eve network
  #
  einput = Input(shape=(c_bits,)) #ciphertext only

  edense1 = Dense(units=(c_bits + k_bits))(einput)
  edense1a = Activation('tanh')(edense1)
  edense2 = Dense(units=(m_bits + k_bits))(edense1a)
  edense2a = Activation('tanh')(edense2)

  ereshape = Reshape((m_bits + k_bits, 1,))(edense2a)

  econv1 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(ereshape)
  econv1a = Activation('tanh')(econv1)
  econv2 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv1a)
  econv2a = Activation('tanh')(econv2)
  econv3 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv2a)
  econv3a = Activation('tanh')(econv3)
  econv4 = Conv1D(filters=4, kernel_size=kersize, strides=1, padding=pad)(econv3a)
  econv4a = Activation('tanh')(econv4)

  eflat = Flatten()(econv4a)
  eoutput = Dense(units=m_bits, activation='sigmoid')(eflat) #code break attempt

  eve = Model(einput, eoutput, name='eve')
  #eve.summary()

  alice.compile(loss='mse', optimizer='sgd')
  bob.compile(loss='mse', optimizer='sgd')
  eve.compile(loss='mse', optimizer='sgd')

  # Establish the communication channels by linking inputs to outputs
  #
  aliceout = alice([ainput0, ainput1])
  bobout = bob( [aliceout, binput1] )# bob sees ciphertext AND key
  eveout = eve( aliceout )# eve doesn't see the key, only the cipher

  # Loss for Eve is just L1 distance between ainput0 and eoutput. The sum
  # is taken over all the bits in the message. The quantity inside the K.mean()
  # is per-example loss. We take the average across the entire mini-batch
  #
  eveloss = K.mean(  K.sum(K.abs(ainput0 - eveout), axis=-1)  )

  # Loss for Alice-Bob communication depends on Bob's reconstruction, but
  # also on Eve's ability to decrypt the message. Eve should do no better
  # than random guessing, so on average she will guess half the bits right.
  #
  bobloss = K.mean(  K.sum(K.abs(ainput0 - bobout), axis=-1)  )
  abeloss = bobloss + K.square(m_bits/2 - eveloss)/( (m_bits//2)**2 )

  # Optimizer and compilation
  #
  abeoptim = Adam()#RMSprop(lr=0.0015)
  eveoptim = Adam()#RMSprop(lr=0.0015) #default 0.001


  # Build and compile the ABE model, used for training Alice-Bob networks
  #
  abemodel = Model([ainput0, ainput1, binput1], bobout, name='abemodel')
  abemodel.add_loss(abeloss)
  abemodel.compile(optimizer=abeoptim)


  # Build and compile the EVE model, used for training Eve net (with Alice frozen)
  #
  alice.trainable = False
  evemodel = Model([ainput0, ainput1], eveout, name='evemodel')
  evemodel.add_loss(eveloss)
  evemodel.compile(optimizer=eveoptim)

  # Keep track of loss at every iteration for the final graph
  abelosses = []
  boblosses = []
  evelosses = []

  n_epochs = 30
  batch_size = 250
  n_batches = m_train // batch_size

  abecycles = 1
  evecycles = 2

  epoch = 0
  print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

  while epoch < n_epochs:
      abelosses0 = [] #epoch-bound losses for text display during training
      boblosses0 = []
      evelosses0 = []
      for iteration in range(n_batches):

          # Train the A-B+E network
          #
          alice.trainable = True
          for cycle in range(abecycles):
              # Select a random batch of messages, and a random batch of keys
              #
              m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
              k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
              loss = abemodel.train_on_batch([m_batch, k_batch, k_batch], None)

          abelosses0.append(loss)
          abelosses.append(loss)
          abeavg = np.mean(abelosses0)

          # Evaluate Bob's ability to decrypt a message
          m_enc = alice.predict([m_batch, k_batch])
          m_dec = bob.predict([m_enc, k_batch])
          loss = np.mean(  np.sum( np.abs(m_batch - m_dec), axis=-1)  )
          boblosses0.append(loss)
          boblosses.append(loss)
          bobavg = np.mean(boblosses0)

          # Train the EVE network
          #
          alice.trainable = False
          for cycle in range(evecycles):
              m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
              k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
              loss = evemodel.train_on_batch([m_batch, k_batch], None)

          evelosses0.append(loss)
          evelosses.append(loss)
          eveavg = np.mean(evelosses0)

          if iteration % max(1, (n_batches // 100)) == 0:
              print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                  epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
              sys.stdout.flush()

      print()
      epoch += 1

  print('Training finished.')

  abelosses_pr[pr] = abelosses
  evelosses_pr[pr] = evelosses
  boblosses_pr[pr] = boblosses
  # steps = -1

  # plt.figure(figsize=(10, 6))
  # plt.plot(abelosses[:steps], label='$L_{Alice-Bob}$', alpha=0.85, color="#648fff")
  # plt.plot(evelosses[:steps], label='$L_{Eve}$', alpha=0.85, color="#dc267f")
  # plt.plot(boblosses[:steps], label='$L_{Bob}$', alpha=0.85, color="#785ef0")
  # plt.xlabel("Training Steps", fontsize=13)
  # plt.ylabel("Losses", fontsize=13)
  # plt.legend(fontsize=13, loc='upper right')

  # plt.savefig("./" + model_name + "-all.png", bbox_inches='tight', dpi=350)
  # plt.show()

  alice.save(alice_file + '.h5', overwrite=True)
  bob.save(bob_file + '.h5', overwrite=True)
  eve.save(eve_file + '.h5', overwrite=True)

  alice = load_model(alice_file + '.h5')
  bob = load_model(bob_file + '.h5')
  eve = load_model(eve_file + '.h5')

  n_examples = 10000

  m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
  k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

  m_enc = alice.predict([m_batch, k_batch])
  #m_enc = np.round(m_enc, 3)
  m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
  m_att = (eve.predict(m_enc) > 0.5).astype(int)

  bdiff = np.abs(m_batch - m_dec)
  bsum = np.sum(bdiff, axis=-1)
  ediff = np.abs(m_batch - m_att)
  esum = np.sum(ediff, axis=-1)

  print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
  print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')

  bob_correct.append(100.0*np.sum(bsum == 0) / n_examples)
  eve_correct.append(100.0*np.sum(esum == 0) / n_examples)

  if best_model > ((100.0 - bob_correct[-1]) + eve_correct[-1]):
    best_model = (100.0 - bob_correct[-1]) + eve_correct[-1]
    print("Saving best: ", best_model)
    alice.save(alice_file + '-best.h5', overwrite=True)
    bob.save(bob_file + '-best.h5', overwrite=True)
    eve.save(eve_file + '-best.h5', overwrite=True)
