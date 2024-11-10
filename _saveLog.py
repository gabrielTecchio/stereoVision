import os
import time

def saveLog(folder, mtx):
    timestamp = time.strftime("%H%M%S")
    file_object = open(os.path.join(folder, 'log_' + timestamp + '.txt'), 'w')
    for line in mtx:
        file_object.write(str(line)+'\n')
    file_object.close()