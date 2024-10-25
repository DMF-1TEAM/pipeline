from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import json
from datetime import datetime
from creative_summary import NewsClusteringSummarizer
from dotenv import load_dotenv
import os
import webbrowser

load_dotenv()
app = Flask(__name__)

class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

class NewsTimelineService:
    def __init__(self, db_path, api_key):
        self.db_path = db_path
        self.summarizer = NewsClusteringSummarizer(api_key=api_key)

    def search_news(self, keyword):
        with DatabaseConnection(self.db_path) as conn:
            # SQL Injection 방지를 위한 파라미터화된 쿼리
            query = """
                SELECT date, title, content
                FROM news
                WHERE content LIKE ?
            """
            df = pd.read_sql_query(query, conn, params=[f'%{keyword}%'])
            return df

    def process_results(self, df):
        if df.empty:
            return None, None

        df['date'] = pd.to_datetime(df['date'])
        sorted_df = df.sort_values(by='date')
        date_counts = sorted_df['d ate'].value_counts()
        impact_dates = date_counts[date_counts >= 10].index
        impact_df = sorted_df[sorted_df['date'].isin(impact_dates)]
        
        return sorted_df, impact_df

    def summarize_articles(self, impact_df):
        articles_by_date = {}
        for date, group in impact_df.groupby(impact_df['date'].dt.date):
            articles_by_date[str(date)] = group[['title', 'content']].to_dict(orient='records')
        
        summaries = {}
        for date, articles in articles_by_date.items():
            results = self.summarizer.process_articles(articles)
            if results:
                summaries[date] = results
            else:
                summaries[date] = [{
                    'title_summary': '요약 결과 없음',
                    'content_summary': '요약 결과를 생성할 수 없습니다.',
                    'article_count': len(articles),
                    'titles': [article['title'] for article in articles]
                }]
        
        return summaries

    def save_summaries(self, summaries):
        with DatabaseConnection(self.db_path) as conn:
            # 타임라인 테이블이 없는 경우 생성
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS timeline_table (
                    date TEXT,
                    title_summary TEXT,
                    content_summary TEXT,
                    article_count INTEGER,
                    processed_at TEXT,
                    titles TEXT
                )
            ''')
            conn.commit()
            
            # 중복 방지를 위해 기존 데이터 삭제
            dates = list(summaries.keys())
            cursor.execute(
                "DELETE FROM timeline_table WHERE date IN ({})".format(
                    ','.join(['?'] * len(dates))
                ), 
                dates
            )
            
            summary_list = []
            for date, clusters in summaries.items():
                for cluster in clusters:
                    summary_list.append({
                        'date': date,
                        'title_summary': cluster['title_summary'],
                        'content_summary': cluster['content_summary'],
                        'article_count': cluster['article_count'],
                        'processed_at': datetime.now().isoformat(),
                        'titles': json.dumps(cluster['titles'])
                    })

            summary_df = pd.DataFrame(summary_list)
            summary_df.to_sql('timeline_table', conn, if_exists='append', index=False)
            conn.commit()

@app.route('/api/timeline', methods=['POST'])
def create_timeline():
    try:
        data = request.get_json()
        keyword = data.get('keyword')
        
        if not keyword:
            return jsonify({'error': '키워드가 필요합니다.'}), 400

        service = NewsTimelineService(
            db_path='news_data.db',
            api_key=os.getenv('OPENAI_API_KEY')
        )

        # 뉴스 검색
        df = service.search_news(keyword)
        
        # 임팩트 포인트 처리
        _, impact_df = service.process_results(df)
        
        if impact_df is None or impact_df.empty:
            return jsonify({'message': '검색 결과가 없습니다.'}), 404

        # 요약 생성
        summaries = service.summarize_articles(impact_df)
        
        # DB 저장
        service.save_summaries(summaries)

        return jsonify({
            'message': '타임라인이 생성되었습니다.',
            'timeline': summaries
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/search', methods=['GET'])
def search_articles():
    try:
        keyword = request.args.get('keyword')
        
        if not keyword:
            return jsonify({'error': '키워드가 필요합니다.'}), 400

        service = NewsTimelineService(
            db_path='news_data.db',
            api_key=os.getenv('OPENAI_API_KEY')
        )

        # 뉴스 검색
        df = service.search_news(keyword)
        
        if df.empty:
            return jsonify({'message': '검색 결과가 없습니다.'}), 404

        # 검색 결과를 JSON으로 변환
        results = df.to_dict(orient='records')

        return jsonify({
            'message': f'"{keyword}" 키워드로 검색된 결과입니다.',
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeline', methods=['GET'])
def get_timeline():
    try:
        date = request.args.get('date')
        
        with DatabaseConnection('news_data.db') as conn:
            query = "SELECT * FROM timeline_table"
            params = []
            
            if date:
                query += " WHERE date = ?"
                params.append(date)
                
            query += " ORDER BY date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return jsonify({'message': '데이터가 없습니다.'}), 404
                
            timeline = df.to_dict(orient='records')
            return jsonify({'timeline': timeline}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)
    app.run(debug=True, port=port)
