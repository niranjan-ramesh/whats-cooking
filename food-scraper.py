# import time
import requests
from multiprocessing import Pool
# from bs4 import BeautifulSoup
# from selenium import webdriver
import os

if not os.path.exists('source'):
    os.mkdir('source')

import logging
from threading import Thread
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(threadName)-2s) %(message)s',
                    )

def worker(name):
    URL = 'https://www.food.com/recipe?pn={}'.format(name)
    response = requests.get(URL)
    source_file = open('source/{}.html'.format(name), 'w')
    source_file.write(response.text)
    source_file.close()

for i in range(52381):
    if((i%100) == 0):
        time.sleep(5)
    t = Thread(target=worker, args=(i, ))
    t.start()
