import json, os, pickle, time, sys
sys.setrecursionlimit(100000)
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

treeDir='/media/novellus/160cdbe0-e0b5-4838-b650-5b1f8ee6fbb4/novellus/trees/'
filesLastAccessed={}
Files={} #replacement for file system, keeps all in memory
oldFiles=set() #cache these to disk to free up memory.

f_name='Uncompressed/RC_2015-04'
#f_name='Uncompressed/RC_2009-08'
totalBytes=float(os.stat(f_name).st_size)
bytesProcessed=0
print 'processing '+f_name+', '+str(totalBytes)+' bytes'
for i_line, line in enumerate(reverse_readline(f_name)):
    bytesProcessed+=len(line)
    if not i_line%10000:
        elapsedTime=time.time()-initialTime
        propBytes=bytesProcessed/totalBytes
        print bytesProcessed, ':', propBytes*100, '% ;', time.time()-initialTime, 'secs passed, ~', elapsedTime/propBytes*(1-propBytes), 'remaining'
        #Cache oldest files to disk
        lastAccessedSorted=sorted(filesLastAccessed.keys(), key=lambda x: filesLastAccessed[x])
        print 'caching', len(lastAccessedSorted)/2, 'files...',
        for i_file in range(len(lastAccessedSorted)/2):
            with open(treeDir+lastAccessedSorted[i_file], 'w') as f:
                json.dump(Files[lastAccessedSorted[i_file]], f)
            del Files[lastAccessedSorted[i_file]]
            oldFiles.add(lastAccessedSorted[i_file])
            del filesLastAccessed[lastAccessedSorted[i_file]]
        print 'done'
    if line.strip():
        comment=json.loads(line)
        #Is comment the parent of anything?
        if comment['name'] in Files:
            #add children
            if 'children' not in comment:
                comment['children']=[]
            for child in Files[comment['name']]:
                comment['children'].append(child)
            del Files[comment['name']]
            del filesLastAccessed[comment['name']]
        elif comment['name'] in oldFiles:
            if 'children' not in comment:
                comment['children']=[]
            with open(treeDir+comment['name']) as f:
                for child in json.load(f):
                    comment['children'].append(child)
            os.remove(treeDir+comment['name'])
            oldFiles.remove(comment['name'])
        if comment['parent_id'] not in Files:
            Files[comment['parent_id']]=[]
        Files[comment['parent_id']].append(comment)
        filesLastAccessed[comment['parent_id']]=i_line
            
totalFiles=len(Files)
print 'Saving '+str(totalFiles)+' files'
for (i, (f_name, f_content)) in enumerate(Files.iteritems()):
    with open(treeDir+f_name, 'w') as f:
        json.dump(f_content, f)
    if not i%1000:
        print i, '/', totalFiles

