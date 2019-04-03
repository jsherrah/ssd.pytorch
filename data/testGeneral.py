# Test out the general data class.
import general
import sys

data = general.GeneralDetection(sys.argv[1], sys.argv[2], sys.argv[3])


print('Data set length = {}'.format(len(data)))

      
for i in range(len(data)):
    img, target = data[i]
    print('sample {:8d} = img:{}, tgt:{}'.format(i, img.shape, target.shape))
    
