# import time
import requests
from multiprocessing import Pool
# from bs4 import BeautifulSoup
# from selenium import webdriver
import os

if not os.path.exists('source'):
    os.mkdir('source')

import logging
import random
import threading
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(threadName)-2s) %(message)s',
                    )

class ActivePool(object):
    def __init__(self):
        super(ActivePool, self).__init__()
        self.active = []
        self.lock = threading.Lock()
    def makeActive(self, name):
        with self.lock:
            self.active.append(name)
            logging.debug('Running: %s', self.active)
    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)
            logging.debug('Running: %s', self.active)

def worker(s, pool):
    logging.debug('Waiting to join the pool')
    with s:
        name = threading.currentThread().getName()
        pool.makeActive(name)
        URL = 'https://www.food.com/recipe?pn={}'.format(name)
        response = requests.get(URL)
        source_file = open('source/{}.html'.format(name), 'w')
        source_file.write(response.text)
        source_file.close()
        pool.makeInactive(name)


pool = ActivePool()
s = threading.Semaphore(50)
for i in range(52381):
    t = threading.Thread(target=worker, name=str(i), args=(s, pool))
    t.start()

# op = webdriver.ChromeOptions()
# op.add_argument('headless')

# driver = webdriver.Chrome(executable_path=r"/Users/ninja/Downloads/chromedriver", options=op)
# driver.get("https://www.food.com/recipe")
# time.sleep(2)  
# driver.find_element_by_css_selector('.gk-aa-load-more').click()
# scroll_pause_time = 0.5
# screen_height = driver.execute_script("return window.screen.height;")   
# i = 1

# while True:
    
#     driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
#     i += 1
#     time.sleep(scroll_pause_time)
    
#     scroll_height = driver.execute_script("return document.body.scrollHeight;")  
    
#     if (screen_height) * i > scroll_height:
#         break

# soup = BeautifulSoup(driver.page_source, "html.parser")
# with open("output1.html", "w") as file:
#     file.write(str(soup))