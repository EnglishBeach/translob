import time
import datetime
import os
open('save_process.log', 'w').close()
while True:
    os.system('cp -av "/content/drive/MyDrive/LOB/Temp/callbacks" "/content/drive/MyDrive/LOB/Save"')
    with open('save_process.log','a')as file:
        time_now = time_now = datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=3)
        file.write(f'Copied at {time_now.strftime("%H:%M:%S %d.%m")}\n')
    time.sleep(120)
