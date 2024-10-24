import os
import json
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from creative_summary import NewsClusteringSummarizer  

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 1. 키워드 입력 함수
def get_keyword():
    keyword = input("검색할 키워드를 입력하세요: ")
    return keyword

# 2. 데이터베이스에서 기사 검색 함수
def search_news_by_keyword(keyword, db_connection):
    query = f"""
        SELECT date, title, content, keyword
        FROM news
        WHERE keyword LIKE '%{keyword}%'
    """
    df = pd.read_sql_query(query, db_connection)
    return df

# 3. 검색 결과를 날짜별로 정렬하고, impect_point를 찾는 함수
def process_results(df):
    if df.empty:
        print("검색 결과가 없습니다.")
        return None, None

    # 날짜별로 정렬
    df['date'] = pd.to_datetime(df['date'])
    sorted_df = df.sort_values(by='date')

    # 날짜별 기사 수 세기
    date_counts = sorted_df['date'].value_counts()

    # 기사 수가 10개 이상인 날짜(impact_point)
    impact_dates = date_counts[date_counts >= 10].index

    # impact_point 날짜의 기사만 따로 추출
    impact_df = sorted_df[sorted_df['date'].isin(impact_dates)]
    
    return sorted_df, impact_df

# 4. 날짜별로 기사를 요약하는 함수
def summarize_articles_by_date(impact_df):
    # 기사들을 날짜별로 그룹화하여 JSON 형태로 변환
    articles_by_date = {}
    for date, group in impact_df.groupby(impact_df['date'].dt.date):
        articles_by_date[str(date)] = group[['title', 'content']].to_dict(orient='records')
    
    # NewsClusteringSummarizer를 사용하여 요약 생성
    summarizer = NewsClusteringSummarizer(api_key=OPENAI_API_KEY)
    summaries = {}
    
    for date, articles in articles_by_date.items():
        results = summarizer.process_articles(articles)
        summaries[date] = results
    
    return summaries

# 5. 요약된 결과를 데이터베이스에 저장하는 함수
def save_summaries_to_db(summaries, db_connection):
    summary_list = []
    
    for date, clusters in summaries.items():
        for cluster in clusters:
            summary_list.append({
                'date': date,
                'summary': cluster['summary'],
                'article_count': cluster['article_count'],
                'processed_at': cluster['processed_at'],
                'titles': json.dumps(cluster['titles'])  # 제목을 JSON 문자열로 변환하여 저장
            })

    # 요약 데이터를 pandas DataFrame으로 변환
    summary_df = pd.DataFrame(summary_list)
    
    # 데이터베이스의 'timeline_table'에 저장
    summary_df.to_sql('timeline_table', db_connection, if_exists='append', index=False)

# 전체 흐름 실행
def main():
    # 데이터베이스 연결 설정
    db_connection = sqlite3.connect('news_database.db')
    
    # 1. 키워드 입력 받기
    keyword = get_keyword()
    
    # 2. 키워드로 기사 검색
    results_df = search_news_by_keyword(keyword, db_connection)
    
    # 3. 검색 결과를 날짜별로 정렬하고, impact_point 기사 찾기
    sorted_df, impact_df = process_results(results_df)
    
    if impact_df is not None:
        # 4. impact_point 기사들을 요약
        summaries = summarize_articles_by_date(impact_df)
        
        # 5. 요약된 결과를 DB에 저장
        save_summaries_to_db(summaries, db_connection)
        print("요약된 결과가 timeline_table에 저장되었습니다.")
    
    # 데이터베이스 연결 종료
    db_connection.close()

# Python 스크립트를 실행할 때 main 함수가 호출되도록 설정
if __name__ == "__main__":
    main()