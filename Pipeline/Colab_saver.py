import time
import datetime
import os
open('save_process.log', 'w').close()
while True:
    os.system('cp -av "/content/drive/MyDrive/LOB/Temp/callbacks" "/content/drive/MyDrive/LOB/Save"')
    with open('save_process.log','a')as file:
        file.write(f'Copied at {datetime.datetime.now().strftime("%H:%M:%S %d.%m")}\n')
    time.sleep(120)
