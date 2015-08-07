import os, json, pickle, signal, sys
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

n=16000

cuda.init()

model=None
if os.path.exists('model'):
    print 'Loading model'
    f=open('model')
    model=pickle.load(f)
    f.close()
    print 'Model Loaded'
else:
    model= FunctionSet(
        x_to_h = F.Linear(8, n),
        h_to_h = F.Linear(n, n),
        h_to_y = F.Linear(n, 8)
    )

optimizer=optimizers.MomentumSGD()
optimizer.setup(model.collect_parameters())

def forward_one_step(h, x):
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    return h, y

def forward(input_string, output_string, volatile=False):
    h=Variable(np.zeros((1,n),dtype=np.float32), volatile=volatile)
    for c in input_string:
        bits=Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32), volatile=volatile) #8 bits, never all 0 for ascii
        h, _ = forward_one_step(h, bits)

    optimizer.zero_grads()
    y=''
    nullEnd=False
    loss=0

    for c in output_string:
        h, yc = forward_one_step(h, Variable(np.array([[0]*8], dtype=np.float32), volatile=volatile))
        loss+=F.mean_squared_error(yc, Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32)))
        if not any(yc.data[0]):
            nullEnd=True
        if not nullEnd:
            y+=chr(sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(yc.data[0])]))
    _, yc = forward_one_step(h, Variable(np.array([[0]*8], dtype=np.float32),  volatile=volatile))
    loss+=F.mean_squared_error(yc, Variable(np.array([[0]*8], dtype=np.float32)))

    loss.backward()
    optimizer.update()
    return y, nullEnd

def trainTree(tree): #DFS training
    if 'children' in tree:
        for child in tree['children']:
            trainTree(child)
            prompt=tree['body']
            trainResponse=child['body']
            if prompt!='[deleted]' and trainResponse!='[deleted]' and prompt and trainResponse:
                optimizer.zero_grads()
                givenResponse, nullEnd=forward(prompt, trainResponse)
                print '<--prompt--'+str(len(prompt))+'chars-->\n', prompt, '\n<--trainResponse--'+str(len(trainResponse))+'chars-->\n', trainResponse, '\n<--givenResponse--'+str(len(givenResponse))+'chars'+('' if nullEnd else ', truncated')+'-->\n', givenResponse, '\n\n'

def sig_exit(_1, _2):
    print 'Stopped computation, saving model. Please wait...'
    f=open('model','w')
    pickle.dump(model, f)
    f.close()
    print 'Saved model'
    exit()
signal.signal(signal.SIGINT, sig_exit)

for fileName in os.listdir('trees/'):
    if '.' not in fileName:
        with open('trees/'+fileName) as f:
            for treeText in f:
                if treeText.strip():
                    #print fileName, treeText
                    tree=json.loads(treeText.strip())
                    if 'children' in tree:
                        print 'training '+fileName+'\n'
                        trainTree(tree)
