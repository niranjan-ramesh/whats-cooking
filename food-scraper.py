import time
from bs4 import BeautifulSoup
from selenium import webdriver

# for i in range(52379 + 1):
#     url = 'https://www.food.com/recipe?pn={}'.format(i)

op = webdriver.ChromeOptions()
op.add_argument('headless')

driver = webdriver.Chrome(executable_path=r"/Users/ninja/Downloads/chromedriver", options=op)
driver.get("https://www.food.com/recipe")
time.sleep(2)  
driver.find_element_by_css_selector('.gk-aa-load-more').click()
scroll_pause_time = 0.5
screen_height = driver.execute_script("return window.screen.height;")   
i = 1

while True:
    
    driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
    i += 1
    time.sleep(scroll_pause_time)
    
    scroll_height = driver.execute_script("return document.body.scrollHeight;")  
    
    if (screen_height) * i > scroll_height:
        break

soup = BeautifulSoup(driver.page_source, "html.parser")
with open("output1.html", "w") as file:
    file.write(str(soup))