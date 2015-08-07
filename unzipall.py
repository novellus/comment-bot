import os, re
#print([x for x in os.walk(os.getcwd())][1][0])

for directory, folders, files in os.walk(os.getcwd()):
    for file in filter(lambda x: re.search('\.bz2$', x), files):
        if file[:-4] not in os.listdir('Uncompressed'):
            f=open('a.sh','w')
            fullPath=('"'+directory+'\\'+file+'"').replace('\\','/')
            f.write('bunzip2 -k '+fullPath+'\ncp '+fullPath[:-5]+'" Uncompressed/.\nrm '+fullPath[:-5]+'"')
            f.close()
            os.system('a.sh')
        else:
            print('skipped '+file)
#bunzip2 -k 2007/RC_2007-10.bz2

