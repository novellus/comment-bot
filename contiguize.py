import os, json, random, unicodedata, sys
sys.setrecursionlimit(100000)

maxLength=500
trees=[]

allPairs=[]
standardize=lambda x: x.strip().replace('\r\n', '\r').replace('\n', '\r').strip()

def allPairsFromTree(tree):
    a=standardize(tree['body'])
    for child in tree['children']:
        b=standardize(child['body'])
        if a and b and a!='[deleted]' and b!='[deleted]' and len(a)<maxLength and len(b)<maxLength:
            allPairs.append([a,b])
        if 'children' in child:
            allPairsFromTree(child)

f_output=open('contiguous.txt','w')

def dumpOutput():
    global trees
    global allPairs
    print 'dumping output'
    for i_tree, tree in enumerate(trees):
        if not i_tree%1000:
            print 'processing tree', i_tree, '/', len(trees)
        allPairsFromTree(tree)
    trees=[]
    random.shuffle(allPairs)
    for i_pair, pair in enumerate(allPairs):
        if not i_pair%1000:
            print 'processing pair', i_pair, '/', len(allPairs)
        f_output.write(unicodedata.normalize('NFKD', pair[0]+'\n'+pair[1]+'\n\n').encode('ascii', 'backslashreplace'))
    allPairs=[]

print 'Gathering tree list...'
allTreeFiles=list(os.walk(os.getcwd()+'/trees'))[0][2]
for i, treeFile in enumerate(allTreeFiles):
    if not i%10000:
        dumpOutput()
    if not i%1000:
        print 'processing file', i, '/', len(allTreeFiles)
    if '.' not in treeFile:
        f=open('trees/'+treeFile)
        for line in f:
            if line.strip():
                file_trees=json.loads(line.strip())
                for tree in file_trees:
                    if 'children' in tree:
                        trees.append(tree)

dumpOutput()
f_output.close()
