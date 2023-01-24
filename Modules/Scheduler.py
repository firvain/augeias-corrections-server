import logging
import time


from apscheduler.triggers.cron import CronTrigger

from colorama import init
from apscheduler.schedulers.background import BackgroundScheduler

logname = 'scheduler.log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logging.getLogger('apscheduler').setLevel(logging.ERROR)

init(autoreset=True)

def my_schedule(job1):
    scheduler = BackgroundScheduler({'apscheduler.timezone': 'Europe/Athens'})

    trigger_job1 = CronTrigger(
        year="*", month="*", day="*", hour="*/12", minute="5", second="0", timezone="Europe/Athens"
    )

    scheduler.add_job(job1, trigger=trigger_job1, name="corrections")

    scheduler.start()

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()