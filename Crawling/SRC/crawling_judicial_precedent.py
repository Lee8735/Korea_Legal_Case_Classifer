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


# 다운로드가 완료되었는지 확인하는 함수
def is_download_finished(download_path):
    for _ in range(30):  # 최대 30초 동안 대기
        # 다운로드 중인 파일 확인
        downloading_files = [
            f for f in os.listdir(download_path)
            if f.endswith('.crdownload') or f.endswith('.part')
        ]
        if not downloading_files:
            return True  # 다운로드 완료
        time.sleep(1)  # 1초 대기
    return False  # 다운로드 미완료



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
        
        self.url = 'https://www.scourt.go.kr/portal/dcboard/DcNewsListAction.work?gubun=44'
        
        self.driver = None
        
        self.XPATH = None
        
        self.DataFrame = pd.DataFrame(columns=['Category', 
                                               'Orign Category', 
                                               'Clean Title', 
                                               'Detail', 
                                               'PDF File'])
        
        self.Category = None
        self.Origin_Category = None
        self.Clean_Title = None
        self.Detail = None
        self.PDF_File = None
        
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

        time.sleep(3)
        # 새로고침 한번 (안하면 막힘.)
        self.driver.refresh()
        time.sleep(1)
        
    
    
    # 카테고리로 판례들 검색.
    def Surch_Category(self, str):
        serch_field_XPATH = '//*[@id="searchWord"]'
        
        serch_field = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, serch_field_XPATH))
        )
        
        serch_field.send_keys(str)
        
        serch_btn_XPATH = '//*[@id="content"]/div[2]/form/div[2]/div[2]/a'
        serch_btn = self.driver.find_element(By.XPATH, serch_btn_XPATH)
        serch_btn.click()
        
        time.sleep(1)
    
    
    
    # 목록에서 제목 클릭해서 내용으로 이동.
    def Select_title(self):
        button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH))
        )
        button.click()

        time.sleep(2)
    
    
    
    # 제목 내용 가져오기.
    def Get_Title(self):
        title_text = self.driver.find_element(By.XPATH, self.XPATH).text
        # print(title_text) #디버깅용

        # 정규표현식으로 [분류]만 남김.
        self.Origin_Category = re.findall(r"\[.*?\]", title_text)
        # print(category) #디버깅용

        # 정규표현식으로 [분류]와 (사건 번호) 삭제.
        self.Clean_Title = re.sub(r"\[.*?\]|\(.*?\)", "", title_text).strip()
        # print(cleaned_title) #디버깅용
    
    
    
    # 판례 내의 요약글 가져오기.
    def Get_Detail(self):
        self.Detail = self.driver.find_element(By.XPATH, self.XPATH).text
        # print(text) #디버깅용
    
    
    
    # PDF 링크 찾기 및 PDF 다운로드 클릭
    def Download_PDF(self):
           
        Download_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH))
        )
        Download_button.click()

        time.sleep(0.5)
        
        # 다운로드 종료 대기
        if not is_download_finished(self.DownPath):
            raise TimeoutError("파일 다운로드가 완료되지 않았습니다.")

        # 파일 이름 변경
        downloaded_file = max(
            [os.path.join(self.DownPath, f) 
                      for f in os.listdir(self.DownPath)],
            key=os.path.getctime
        )
        
         # 파일 이름 변경
        downloaded_file = max(
            [os.path.join(self.DownPath, f) for f in os.listdir(self.DownPath)],
            key=os.path.getctime
        )

        
        self.PDF_File = f'{self.Category}_{self.DataNum + 1}'
        new_file_name = os.path.join(self.DownPath, f"{self.PDF_File}.pdf")
        os.rename(downloaded_file, new_file_name)
        
    
    # 하단 페이지 중 하나 선택
    def Click_Page(self):
        list_btn = self.driver.find_element(By.XPATH, self.XPATH)
        list_btn.click()
    
    
    # 하단 페이지 이동
    def Click_Next_Page(self):
        next_page_btn = self.driver.find_element(By.XPATH, self.XPATH)
        next_page_btn.click()
    
    
    # 데이터 프레임에 데이터 추가
    def Append_DataFrame(self):        
        list_buff = []
        
        list_buff.append(self.Category)
        list_buff.append(self.Origin_Category)
        list_buff.append(self.Clean_Title)
        list_buff.append(self.Detail)
        list_buff.append(self.PDF_File)
        
        
        self.DataFrame.loc[len(self.DataFrame)] = list_buff
        
        self.DataNum = self.DataNum + 1
        
        
        
     
    
#%%
wb = Web_Browser()

wb.OpenBrowser()
wb.OpenURL()

wb.Surch_Category('[행정]')
wb.Category = '행정'

wb.XPATH = f'//*[@id="content"]/div[2]/div/a[8]'

for k in range(0, 48):
    for i in range(3, 8):
        wb.XPATH = f'//*[@id="content"]/div[2]/div/a[{i}]'
        wb.Click_Page()
        
        for j in range(1, 11):
            wb.XPATH = f'//*[@id="content"]/div[2]/table/tbody/tr[{j}]/td[2]/a'
            wb.Select_title()
            
            wb.XPATH = '//*[@id="content"]/form/div/table[1]/tbody/tr[1]/td'
            wb.Get_Title()
            
            wb.XPATH = '//*[@id="content"]/form/div/table[1]/tbody/tr[5]/td'
            wb.Get_Detail()
            
            wb.XPATH = '//*[@id="content"]/form/div/table[1]/tbody/tr[4]/td/a'
            wb.Download_PDF()
            
            wb.Append_DataFrame()
            print(wb.DataNum)
            
            wb.driver.back()
            time.sleep(0.5)
    
    if k < 47:
        wb.XPATH = f'//*[@id="content"]/div[2]/div/a[8]'
        wb.Click_Next_Page()
    
    
wb.DataFrame.to_csv(f'./no_git/Data/{wb.Category}.csv')





#파일 없는경우 예외처리