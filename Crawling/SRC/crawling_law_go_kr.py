#%%
from requests import options

from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException

from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import re
import time
import datetime
import os



class Web_Browser:
    
    def __init__(self):
        
        self.UserAgent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '            
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
        )
        
        self.DownPath = (
            r"C:\Users\Equalia\Desktop\Python_VScode"
            r"\EdgeAI_Crawling\EdgeAI_Crawling\no_git\download_pdf"
        )
        
        self.url = 'https://www.law.go.kr/precSc.do?menuId=7&subMenuId=47&tabMenuId=213'
        
        self.driver = None
        
        self.XPATH = None
        
        self.DataFrame = pd.DataFrame(columns=['Category', 
                                               'Title', 
                                               'Detail'])
        
        self.Category = None
        self.Title = None
        self.Detail = None
                
        self.DataNum = 0
    
    
    
    # 웹 브라우저를 엶.
    def OpenBrowser(self):

        # Chrome 옵션 설정
        chrome_options = ChromeOptions()

        # 다운로드 디렉토리 지정.
        # 다운로드 팝업 비활성화
        # 
        # PDF를 브라우저가 아닌 디스크로 직접 다운로드
        prefs = {
            "download.default_directory": self.DownPath,  
            "download.prompt_for_download": False,       
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True   
        }
        
        chrome_options.add_experimental_option("prefs", prefs)

        chrome_options.add_argument('user_agent=' + self.UserAgent)
        chrome_options.add_argument('long=ko_KR')

        # ChromeDriver 설정
        service = ChromeService(executable_path=ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    
    
    # URL로 이동.
    def OpenURL(self):
        self.driver.get(self.url)

        # time.sleep(3)
        # # 새로고침 한번 (안하면 막힘.)
        # self.driver.refresh()
        time.sleep(1)
        
    
    
    # 카테고리로 판례 필터링.
    def Surch_Category(self):
        btn_XPATH = '//*[@id="dtlSch"]/a[1]'
        btn = self.driver.find_element(By.XPATH, btn_XPATH)
        btn.click()
        
        time.sleep(0.5)
        
        cbox = self.driver.find_element(By.XPATH, self.XPATH)
        cbox.click()
        
        btn_XPATH ='//*[@id="precDetailCtDiv"]/div[3]/div[2]/a'
        btn = self.driver.find_element(By.XPATH, btn_XPATH)
        btn.click()
        time.sleep(0.5) 
    
       
    # 제목 내용 가져오기.
    def Get_Title(self):
        self.Title = self.driver.find_element(By.XPATH, self.XPATH).text
        print(self.Title) #디버깅용
    
       
    # 판례 내의 요약글 가져오기.
    def Get_Detail(self):
        self.Detail = self.driver.find_element(By.XPATH, self.XPATH).text
        print(self.Detail) #디버깅용
       
          
    # 하단 페이지 중 하나 선택
    def Click_Page(self):        
        list_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, self.XPATH))
        )
        list_btn.click()
    
    
    # 하단 페이지 이동
    def Click_Next_Page(self):
        next_page_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, self.XPATH))
        )
        next_page_btn.click()
    
    
    # 데이터 프레임에 데이터 추가
    def Append_DataFrame(self):        
        list_buff = []
        
        list_buff.append(self.Category)
        list_buff.append(self.Title)
        list_buff.append(self.Detail)
        
        self.DataFrame.loc[len(self.DataFrame)] = list_buff
        
        self.DataNum = self.DataNum + 1
          


Search_btn_XPATH = '//*[@id="dtlSch"]/a[1]'

Cvil_cbx_XPATH = '//*[@id="evtKnd1"]'
Criminal_cbx_XPATH = '//*[@id="evtKnd2"]'
Family_cbx_XPATH = '//*[@id="evtKnd3"]'

Patent_cbx_XPATH = '//*[@id="evtKnd6"]'
Administ_cbx_XPATH = '//*[@id="evtKnd7"]'
Tax_cbx_XPATH = '//*[@id="evtKnd8"]'

#li
UnderPageIndex_btn = '//*[@id="WideListDIV"]/div/div[6]/ol/li[1]'




#===================================================================================================   
#%%
wb = Web_Browser()

wb.OpenBrowser()
wb.OpenURL()

wb.Category = '민사'

wb.XPATH = Cvil_cbx_XPATH
wb.Surch_Category()

time.sleep(0.5)

#



df = pd.DataFrame(columns=['Ctegory', 'Title', 'Detail'])

# 50 * 5 = 250
# 50 * 5 * 5 = 1250

for k in range(0,5):
    for j in range(1, 6):
        for i in range(1,101, 2):
            test_list = []
            
            wb.XPATH = f'//*[@id="viewHeightDiv"]/table/tbody/tr[{i}]/td[2]'               
            wb.Get_Title()
            
            wb.XPATH = f'//*[@id="viewHeightDiv"]/table/tbody/tr[{i + 1}]/td'
            wb.Get_Detail()
            
            wb.Append_DataFrame()
            test_list.append(wb.Category)
            test_list.append(wb.Title)
            test_list.append(wb.Detail)
            df.loc[len(df)] = test_list
            print(i)

        if j < 5 :
            wb.XPATH = f'//*[@id="WideListDIV"]/div/div[6]/ol/li[{j + 1}]/a'
            wb.Click_Page()
            time.sleep(2)
        else :
            pass
    
    if k < 4:
        wb.XPATH = f'//*[@id="WideListDIV"]/div/div[6]/a[3]'
        wb.Click_Next_Page()
        time.sleep(2)
    else:
        pass



df.to_csv(f'./Crawling/Data/row_{wb.Category}.csv', index=False)