import os

path = "/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_DOTA/test/VOC/test/VOCdevkit/VOC2007/JPEGImages/"



# l=os.listdir(path)
# li=[x.split('.')[0]for x in l]
# print(li)

f = open("/home/mousumi/PycharmProjects/Joint_Optimization_Fine_Tune_DOTA/test/VOC/test/VOCdevkit/VOC2007/ImageSets/Main/test.txt", "w")
for filename in sorted(os.listdir(path)):
  filepath = os.path.join(path, filename)
  fd = open(filepath, 'r')
  annotfile = os.path.splitext(filename)[0]
        #annotfile=annotfile.replace("shadow", "shadow1")
  print(annotfile)
        # annotfile1 = os.path.splitext(filename)[0] + "_ir"
  f.write(annotfile+'\n')
        #f.write(annotfile1 + '\n')
 # process(fd, annotfile)
f.close()
