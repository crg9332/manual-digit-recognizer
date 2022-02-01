#feed forward
from numpy import dot, exp, random, array, argmax
import tensorflow as tf # Import tensorflow library
import matplotlib.pyplot as plt # Import matplotlib library

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

random.seed(1)

iun = x_train
i = iun.reshape((60000, 1, 784))
#bias
hb = random.random((1,512))
sb = random.random((1,256))
tb = random.random((1,128))
b = random.random((1,10))
#weights
hw = 2 * random.random((784,512)) - 1
sw = 2 * random.random((512,256)) - 1
tw = 2 * random.random((256,128)) - 1
w = 2 * random.random((128,10)) - 1
lr = 0.0001

numbers = array([[1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1]])

def sig(x):
    return 1 / (1 + exp(-(x)))

def dsig(x):
    return sig(x) * (1 - sig(x))

# dcost/hw = dcost/a * da/sum * dsum/ha * dha/hsum * dhsum/dhw = (a - y) * dsig(sum) * w * dsig(hsum) * i

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r',
    ('#'*filled_progbar + '-'*(full_progbar-filled_progbar)),
    ('[{:>7.2%}]'.format(frac)),
    '[Epoch:', tot + 1,']',
    '[Iteration:', curr,']',
    end='')

for tot in range(5):
    count = 0
    for epoch in i[range(59999)]:
        progbar(count, 59999, 20)
        yun = y_train
        y = numbers[yun[count]]
        hsum = dot(epoch, hw) + hb #1x512
        ha = sig(hsum) #1x512
        ssum = dot(ha, sw) + sb #1x256
        sa = sig(ssum) #1x256
        tsum = dot(sa, tw) + tb #1x128
        ta = sig(tsum) #1x128
        sum = dot(ta, w) + b #1x10
        a = sig(sum) #1x10
#        dcost_a = (a - y)

        dhsum_hw = i.T[:, :,count] #784x1
        dha_hsum = dsig(hsum) #1x512
        dssum_ha = sw.T #256x512
        dsa_ssum = dsig(ssum) #1x256
        dtsum_sa = tw.T #128x256
        dta_tsum = dsig(tsum) #1x128
        dsum_ta = w.T #10x128
        da_sum = dsig(sum) #1x10
        dcost_a = (a - y) #1x10

        hw -= lr * dot(dhsum_hw, (dha_hsum * dot((dsa_ssum * dot((dta_tsum * dot((da_sum * dcost_a), dsum_ta)), dtsum_sa)), dssum_ha))) #784x512
        hb -= dha_hsum * dot((dsa_ssum * dot((dta_tsum * dot((da_sum * dcost_a), dsum_ta)), dtsum_sa)), dssum_ha) #1x512

        dssum_sw = ha.T #512x1
        #dsa_ssum = dsig(ssum) #1x256

        sw -= lr * dot(dssum_sw, (dsa_ssum * dot((dta_tsum * dot((da_sum * dcost_a), dsum_ta)), dtsum_sa))) #512x256
        sb -= lr * (dsa_ssum * dot((dta_tsum * dot((da_sum * dcost_a), dsum_ta)), dtsum_sa))

        dtsum_tw = sa.T #256x1

        tw -= lr * (dot(dtsum_tw, (dta_tsum * dot((da_sum * dcost_a), dsum_ta)))) #256x128
        tb -= lr * (dta_tsum * dot((da_sum * dcost_a), dsum_ta)) #1x128

        dsum_w = ta.T #128x1

        w -= lr * (dot(dsum_w, (da_sum * dcost_a))) #128x10
        b -= lr * (da_sum * dcost_a)
        count += 1
    print()


#Testing
it = x_test
yt = y_test
newit = it.reshape((10000, 1, 784))

#print(newit[0])
#print(newit.shape)
while True:
    index = int(input("index: "))
    hsum = dot(newit[index], hw) + hb
    ha = sig(hsum)
    ssum = dot(ha, sw) + sb
    sa = sig(ssum)
    tsum = dot(sa, tw) + tb
    ta = sig(tsum)
    sum = dot(ta, w) + b
    a = sig(sum)
    print(argmax(a))
    print(yt[index])
