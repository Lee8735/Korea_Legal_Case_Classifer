<img src="https://img.shields.io/badge/Python v3.10.11-3776AB?style=for-the-badge&logo=Python&logoColor=white"><br/>
<img src="https://img.shields.io/badge/TensorFlow v2.18.0-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white">
<img src="https://img.shields.io/badge/Keras v3.7.0-D00000?style=for-the-badge&logo=Keras&logoColor=white"><br/>
<img src="https://img.shields.io/badge/scikit learn v1.6.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Pandas v2.2.3-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/numpy v2.0.2-013243?style=for-the-badge&logo=numpy&logoColor=white">

모델 학습 데이터 출저: https://www.scourt.go.kr/portal/dcboard/DcNewsListAction.work?gubun=44
<br/><br/><br/>


# 법원 판례기반 사건 유형 분류기
- 목표: 사건 내용 입력시 사건 유형을 분류해 주는 프로그램 만들기.

- 개요: 대한민국 법원의 판례 데이터를 학습시킨 모델을 이용하여, 사건 내용을 텍스트로 입력시 민사, 형사, 가정, 행정, 특허 중 어떤 사건인지 분류하여 가능성이 높은 유형 두 개를 보여주는 프로그램.
<br/>

![Screenshot 2024-12-26 193723](https://github.com/user-attachments/assets/c2a43d85-4129-4e42-8323-8b795e1cc3e7)
UI 디자인. **CHOITAEK TAEKEUN** (https://github.com/CHOITAEK)

<br/><br/><br/>

# 데이터 전처리 과정

 ┣ 📂Preprocessing  
초기 데이터에서 한글만 남긴 후 연속 공백을 제거한 뒤 하나로 규합.
<br/>
- 초기 규합 뒤 데이터 컬럼
- Category: 분류(형사, 민사, 행정, 가사, 특허, 디자인, 상표)
- Origin Category: 제목상의 분류
- Clean title: 제목내용
- Detail: 본문내용
- ~~PDF File~~: 사용 안함.(삭제)

<br/>

위의 데이터에서 제목 내용도 학습에 사용 가능할거라 판단해 **제목내용**과 **본문내용**을 각각 하나의 데이터로 보기로 함.  

**분류** 와 **제목상의 분류**를 비교하여 다를 경우 **Category** 컬럼을 **제목상의 분류**로 대체한 뒤, **Origin Category**컬럼 제거.(다른 경우 없었음)  

**제목내용**이 **본문내용**에 포함되어 있거나 같은 경우, 해당 **Clean Title** 컬럼 NaN값으로 변경.  

**Clean Title**, **Detail**컬럼을 **Category**는 유지한 채 **Data**컬럼으로 규합.  

이 후 NaN 값 제거, 중복 제거, 공백포함 텍스트 길이 15이하인 데이터 삭제.  

불용어의 경우 띄어쓰기를 기준으로 토큰화 한 뒤 임의로 제거.

이 후 **kolnPy** 의 **Okt**를 사용하여 형태소 분리.

<br/>

 ┣ 📂Make_Learning_Data  
상표, 디자인의 경우 특허와 같다고 보고 특허로 변경.  

**죄종 학습 데이터**  
카테고리: 민사, 형사, 가정, 행정, 특허 (5개)  
학습 데이터 수: 각 카테고리 별 1000개, 총합 5000개  
검증 데이터 수: 가정(79 개)를 제외한 나머지 카테고리 각각 1000개, 촣합 4079개  

<br/><br/><br/>

# 프로젝트 폴더 구조
📦Korea_Legal_Case_Classifer  : 프로젝트 작업영역 root폴더.  
 ┣ 📂APP                      : 응용 프로그램을 위한 소스코드 및 .ui파일, 이미지 리소스, .qrc, background_rc.py 파일.  
 ┣ 📂Crawling                 : 크롤링 소스코드 및 크롤링 데이터.  
 ┣ 📂Make_Learning_Data       : 모델 학습, 검증을 위한 데이터를 만들기 위한 소스코드 및 데이터. (인코더 pickle 포함)  
 ┣ 📂Model_Learning           : 모델 생성 및 학습을 위한 소스코드 및 학습된 모델. (.h5)  
 ┣ 📂Model_predict            : 모델 테스트를 위한 소스코드.  
 ┣ 📂Preprocessing            : 전처리를 위한 소스코드 및 전처리 중간 데이터.  
 ┗ 📜requirements.txt         : 필요 패키지 목록.  


<br/><br/><br/>


# 모델 정보

모델 학습중의 accuracy 및 val accruacy 변화.  
![Model1_learnig_history](https://github.com/user-attachments/assets/98c46124-9ebc-4624-9c7c-c05af272a039)

<br/>

학습 후 모델 parameter.  
![model_summary](https://github.com/user-attachments/assets/ad56198d-358c-4cc8-9927-eb00719cc98d)

<br/>

최종 저장된 모델 정보
- accuracy : 0.9196
- loss : 0.2590
- val accuracy : 0.8769
- val loss : 0.4059

<br/><br/><br/>

