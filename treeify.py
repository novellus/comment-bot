import json, os, pickle

allNames={} #every node name indexed by file name
#any node in a tree can have a new child, but only roots can have new parents.
#files are named as the parent id of the root node

if os.path.exists('allNames'):
    with open('allNames') as nameStash:
        allNames=json.load(nameStash)

#DFS insert
#returns false if could not insert
def insert(root, node):
    if root['name']==node['parent_id']:
        if 'children' not in root:
            root['children']=[]
        root['children'].append(node)
        return True
    elif 'children' in root:
        for child in root['children']:
            if insert(child, node):
                return True #not just back propagation of success, also to exit quickly once child is inserted
    return False #node does not belong in the root or any of its children
    
f_name='Uncompressed/RC_2007-10'
#f_name='Uncompressed/RC_2009-08'
with open(f_name) as f:
    print('processing '+f_name+', approx '+str(int(os.stat(f_name).st_size*2/1000))+' lines')
    for i_line, line in enumerate(f):
        if not i_line%10000:
            print(i_line) #progress printing, mostly debugging
        if line.strip():
            comment=json.loads(line)
            #Is comment the parent of anything?
            isParent=False
            for root in os.listdir('trees'):
                if comment['name']==root:
                    isParent=True
                    #add children
                    if 'children' not in comment:
                        comment['children']=[]
                    with open('trees/'+root) as f_child:
                        for line_child in f_child:
                            if line_child.strip():
                                child=json.loads(line_child)
                                comment['children'].append(child)
                    os.remove('trees/'+root) #delete unused child tree(s) file
                    #update allNames dictionary
                    allNames[comment['parent_id']]=allNames[root]
                    del allNames[root]
                    allNames[comment['parent_id']].append(comment['name'])
                    break #guaranteed to have only one file of this name
            #Is comment the child of anything?
            isChild=False
            for file_name in allNames:
                for name in allNames[file_name]:
                    if name==comment['parent_id']:
                        isChild=True
                        #traverse the tree to insert
                        newNodeList=[] #files can have multiple trees in them if they never end up with a parent (some comments are missing)
                        with open('trees/'+file_name) as f_parent:
                            for line_parent in f_parent: #I do not expect to be parsing more than one node in a file often, since the root node should appear first chronologically. However the occasional comment is missing so there may be a few cases of this.
                                if line_parent.strip():
                                    parent=json.loads(line_parent)
                                    insert(parent, comment) #try insert on every tree in this file, nothing happens if it doesn't belong
                                    newNodeList.append(json.dumps(parent))
                                #Don't break if insert returns true, since we will still need to finish adding all nodes form this file back into the file... (its a good thing this shouldn't happen very often)
                        #save new parent file
                        f_parent=open('trees/'+file_name,'w')
                        f_parent.write('\n'.join(newNodeList))
                        f_parent.close()
                        #update allNames
                        if isParent:
                            allNames[file_name]+=allNames[comment['parent_id']]
                            del allNames[comment['parent_id']]
                        else:
                            allNames[file_name].append(comment['name'])
                        break #guaranteed to have only 1 parent
                if isChild:
                    break #break this loop if we already found comment's parent
            #if not a child of anything, either join existing file of children looking for same parent (aka cult), or create new file if we are the first child looking for this parent
            if not isChild:
                outputDir='trees/'+comment['parent_id']
                #update allNames
                if not isParent:
                    if os.path.exists(outputDir): #join a cult
                        allNames[comment['parent_id']].append(comment['name'])
                    else:
                        allNames[comment['parent_id']]=[comment['name']]
                #no else b/c if we are a parent allNames is already up to date
                #save file
                f_output=open(outputDir, 'a' if os.path.exists(outputDir) else 'w')
                f_output.write('\n'+json.dumps(comment))
                f_output.close()

with open('allNames','w') as nameStash:
    json.dump(allNames, nameStash)
