# web/services/news_service.py
from django.db.models import Q
from ..models import News, Timeline
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from .creative_summary import NewsClusteringSummarizer
from .issue_analyzer import NewsStoryAnalyzer
import json

class NewsService:
    def __init__(self, api_key):
        self.summarizer = NewsClusteringSummarizer(api_key=api_key)
        self.analyzer = NewsStoryAnalyzer(api_key=api_key)

    def search_news(self, keyword: str) -> pd.DataFrame:
        """키워드로 뉴스 검색"""
        news_queryset = News.objects.filter(
            Q(content__icontains=keyword) | Q(title__icontains=keyword)
        ).values('date', 'title', 'content')
        
        df = pd.DataFrame.from_records(news_queryset)
        return df

 
    def process_results(self, df: pd.DataFrame) -> tuple:
        """검색 결과 처리 및 임팩트 포인트 추출"""
        if df.empty:
            return None, None

        df['date'] = pd.to_datetime(df['date'])
        sorted_df = df.sort_values(by='date')
        
        # 모든 데이터를 분석 대상으로 사용
        impact_df = sorted_df
        
        return sorted_df, impact_df

    def create_timeline(self, impact_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """타임라인 생성"""
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

    def analyze_story(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """스토리 분석 수행"""
        return self.analyzer.analyze_story(articles)

    def save_timeline(self, summaries: Dict[str, List[Dict[str, Any]]]):
        """타임라인 저장"""
        # 기존 데이터 삭제
        dates = list(summaries.keys())
        Timeline.objects.filter(date__in=dates).delete()
        
        # 새로운 데이터 저장
        timeline_objects = []
        for date, clusters in summaries.items():
            for cluster in clusters:
                timeline_objects.append(Timeline(
                    date=datetime.strptime(date, '%Y-%m-%d').date(),
                    title_summary=cluster['title_summary'],
                    content_summary=cluster['content_summary'],
                    article_count=cluster['article_count'],
                    titles=json.dumps(cluster['titles'])
                ))
        
        if timeline_objects:
            Timeline.objects.bulk_create(timeline_objects)