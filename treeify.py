import json, os, pickle, time, mmap
initialTime=time.time()

def reverse_readline(filename, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        total_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(total_size, offset + buf_size)
            fh.seek(-offset, os.SEEK_END)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first 
                if buffer[-1] is not '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if len(lines[index]):
                    yield lines[index]
        yield segment

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
    
Files={} #replacement for file system, keeps all in memory
processingTimeEstimate=lambda bytes: 0.00000000000004056*bytes*bytes-0.00000011*bytes+2

f_name='Uncompressed/RC_2007-10'
totalBytes=int(os.stat(f_name).st_size)
totalTimeEstimate=processingTimeEstimate(totalBytes)
bytesProcessed=0
print 'processing '+f_name+', '+str(totalBytes)+' bytes'
for i_line, line in enumerate(reverse_readline(f_name)):
    bytesProcessed+=len(line)
    if not i_line%1000 and i_line!=0:
        timeLeftEstimate=totalTimeEstimate-processingTimeEstimate(bytesProcessed)
        print bytesProcessed, ':', bytesProcessed*100.0/totalBytes, '% ;', time.time()-initialTime, 'secs passed, ~', timeLeftEstimate, 'remaining (', totalTimeEstimate-(timeLeftEstimate+time.time()-initialTime), ' error)'  #progress printing, mostly debugging
    if line.strip():
        comment=json.loads(line)
        #Is comment the parent of anything?
        for root in Files:
            if comment['name']==root:
                #add children
                if 'children' not in comment:
                    comment['children']=[]
                for child in Files[root]:
                    comment['children'].append(child)
                del Files[root]
                break #guaranteed to have only one file of this name
        if comment['parent_id'] not in Files:
            Files[comment['parent_id']]=[]
        Files[comment['parent_id']].append(comment)
            
totalFiles=len(Files)
print 'Saving '+str(totalFiles)+' files'
for (i, (f_name, f_content)) in enumerate(Files.iteritems()):
    with open('trees/'+f_name, 'w') as f:
        json.dump(f_content, f)
    if not i%1000:
        print i, '/', totalFiles

