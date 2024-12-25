#===================================================================================================
#%%
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk

from konlpy.tag import Okt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter



# 텍스트에서 한글만 남겨서 반환하는 함수
def Remain_Korean(text):
    if pd.isna(text):
        return pd.NA
    else:
        return re.compile('[^가-힣 ]').sub(' ', text)


# 텍스트에서 연속된 공백을 줄인 텍스트를 반환하는 함수.
def Reduce_Space(text):
    if pd.isna(text):
        return pd.NA
    else:
        return re.sub(r'\s+', ' ', text).strip()


# Clean Title의 문장이 Detail에 포함되는지의 여부를 반환하는 함수.
def Str_is_Include(row):
    if pd.isna(row['Clean Title']) or pd.isna(row['Detail']):
        return False
    else:
        if row['Clean Title'] in row['Detail']:
            return True
        else:
            return False


# 단순히 단어로 토큰화 한 뒤 불용어를 제거하고 다시 문장으로 반환하는 함수.
def Tokenize_and_Remove(text, remove_words):
    tokens = word_tokenize(text)
    removed_tokens = [token for token in tokens if token not in remove_words]
    result = ' '.join(removed_tokens) 
    return result



#===================================================================================================
#%% Clean Title 및 Dtail 컬럼의 텍스트 한글만 남기고 연속공백 처리
name_list = ['가사', '디자인', '민사', '상표', '특허', '행정', '형사']


for name in name_list:
    df = pd.read_csv(f'./Crawling/Data/{name}.csv')

    # 텍스트에서 한글과 공백만 남김
    df['Clean Title'] = df['Clean Title'].apply(Remain_Korean)
    df['Detail'] = df['Detail'].apply(Remain_Korean)

    # 텍스트에서 연속된 공백을 하나로 줄임
    df['Clean Title'] = df['Clean Title'].apply(Reduce_Space)
    df['Detail'] = df['Detail'].apply(Reduce_Space)

    print(df.head(10))

    df.to_csv(f'./Preprocessing/Data/Pre_{name}.csv', index=False)



#===================================================================================================
#%% 위의 데이터 파일들을 일단 하나로 합침
name_list = ['가사', '디자인', '민사', '상표', '특허', '행정', '형사']

result = pd.DataFrame(columns=['Category', 'Origin Category', 'Clean Title', 'Detail', 'PDF File'])

for name in name_list:
    df_buff = pd.read_csv(f'./Preprocessing/Data/Pre_{name}.csv')

    result = pd.concat([result, df_buff], axis=0)

result.reset_index()

result.to_csv('./Preprocessing/Data/Pre_Total.csv', index=False)

# 중간저장
result = pd.read_csv('./Preprocessing/Data/Pre_Total.csv')
result.info()

result.head(10)



#===================================================================================================
#%% 데이터 중복 제거 및 카테고리 확인
df = pd.read_csv('./Preprocessing/Data/Pre_Total.csv')

# PDF File열 제거
df.drop(columns=['PDF File'], inplace=True)


# Clean Title 열의 내용이 Detail안에 포함되거나 동일한지 확인.
# 동일 하다면 Clean Title을 NaN값으로 변경.
df['include'] = df.apply(Str_is_Include, axis=1)
df.loc[df['include'] == True, 'Clean Title'] = pd.NA


# Detail 행과 Clean Title 행을 Data로 해서 하나의 열로 합침.
df1 = df.drop(columns=['Detail'])
df1.rename(columns={'Clean Title':'Data'}, inplace=True)
df2 = df.drop(columns=['Clean Title'])
df2.rename(columns={'Detail':'Data'}, inplace=True)

df = pd.concat([df1, df2], axis=0)


# Data행에 NaN값이 있는 경우 삭제.
df.dropna(subset=['Data'], inplace=True)

# Data행이 중복된 경우 삭제.
df.drop_duplicates(subset=['Data'], inplace=True)

# Origin Category 와 Category 비교.
df['Same'] = df.apply(lambda row: row['Category'] in row['Origin Category'], axis=1)
df[df['Same']]
# 모두 같아서 특별한 처리 안함.

df.drop(columns=['Origin Category', 'include', 'Same'], inplace=True)
df.info()

# 중간 저장.
df.to_csv('./Preprocessing/Data/Pre_Total_Category_Data.csv', index=False)



#===================================================================================================
#%% 단어 빈도수 파악 후 워드클라우드 생성
nltk.download('punkt_tab')
df = pd.read_csv('./Preprocessing/Data/Pre_Total_Category_Data.csv')
total_row = ' '.join(df['Data'])

tokens = word_tokenize(total_row)

words_count = Counter(tokens)

top_n = 20
print(words_count.most_common(top_n))

wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/malgun.ttf', 
    background_color='white'
).generate_from_frequencies(words_count)


# 시각화
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



#===================================================================================================
#%% 단순히 단어단위로 토큰화 한 후 불용어 선정 및 제거
df = pd.read_csv('./Preprocessing/Data/Pre_Total_Category_Data.csv')

Remove_Words = [
    '제', 
    '점', 
    '항', 
    '조', 
    '이',
    '제', 
    '승소',
    '사건', 
    '이에',
    '판결',
    '판시', 
    '사례', 
    '사안', 
    '선고', 
    '내지', 
    '가합', 
    '구합', 
    '고단', 
    '고합',
    '가단',
    '가소',  
    '개요', 
    '요지',                  
    '카합', 
    '가소', 
    '표장',
    '제목',
    '형사',
    '병합',
    '선고한',
    '사례임',
    '기각한', 
    '사안의',               
    '민사부', 
    '대상판결', 
    '사건번호', 
    '판결요지', 
    '판시사항', 
    '주요판결',
    '서울고등법원', 
    '대전고등법원', 
    '대구고등법원', 
    '부산고등법원',
    '광주고등법원',
    '수원고등법원',
    '특허법원',
    '서울중앙지방법원',
    '서울가정법원',
    '서울행정법원',
    '서울회생법원',
    '서울동부지방법원',
    '서울남부지방법원',
    '서울북부지방법원',
    '서울서부지방법원',
    '의정부지방법원',
    '인천지방법원',
    '인천가정법원',
    '춘천지방법원',
    '대전지방법원',
    '대전가정법원',
    '청주지방법원',
    '대구지방법원',
    '대구가정법원',
    '부산지방법원',
    '부산가정법원',
    '부산회생법원',
    '울산지방법원',
    '울산가정법원',
    '창원지방법원',
    '광주지방법원',
    '광주가정법원',
    '전주지방법원',
    '제주지방법원',
    '수원지방법원',
    '수원가정법원',
    '수원회생법원'                
]


df['Data'] = df['Data'].apply(lambda data: 
    Tokenize_and_Remove(data,Remove_Words))
df.info()

# Data길이가 15 이하인 데이터 모두 삭제
df['Data'] = df['Data'].fillna('')  # NaN 값을 빈 문자열로 처리
df = df[df['Data'].apply(len) > 15]
df.info()

df.to_csv('./Preprocessing/Data/Pre_remove_word.csv', index=False)


# 단어 삭제후 단어 빈도수 파악 후 워드클라우드 생성
nltk.download('punkt_tab')
df = pd.read_csv('./Preprocessing/Data/Pre_remove_word.csv')
total_row = ' '.join(df['Data'])

tokens = word_tokenize(total_row)

words_count = Counter(tokens)

top_n = 20
print(words_count.most_common(top_n))

wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/malgun.ttf', 
    background_color='white'
).generate_from_frequencies(words_count)


# 시각화
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
    

#===================================================================================================    
#%% Data 형태소 분리

df = pd.read_csv('./Preprocessing/Data/Pre_remove_word.csv')

okt = Okt()

df['Data'] = df['Data'].apply(lambda text: okt.morphs(text, stem=True))

df.to_csv('./Preprocessing/Data/Pre_Okt.csv', index=False)
   
df = pd.read_csv('./Preprocessing/Data/Pre_Okt.csv')

category_counts = df["Category"].value_counts()

print(category_counts)   
    
df.to_csv('./Preprocessing/Pre_End/Pre_End_1.csv', index=False)



