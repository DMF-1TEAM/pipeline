import os
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
import logging

logger = logging.getLogger(__name__)

class NewsStoryAnalyzer:
    def __init__(self, api_key: str = None, max_tokens_per_chunk: int = 4000):
        """Initialize NewsStoryAnalyzer"""
        logger.info("Initializing NewsStoryAnalyzer")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("No API key provided")
            raise ValueError("OpenAI API key must be provided either as argument or environment variable")
            
        self.client = OpenAI(api_key=api_key)
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w+\b',
            lowercase=False
        )
        logger.info("NewsStoryAnalyzer initialized successfully")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """TF-IDF를 사용한 텍스트 임베딩 생성"""
        if not texts:
            return np.array([])
        
        try:
            # 정규화된 텍스트로 변환
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # 빈 텍스트 확인
            if all(not text.strip() for text in processed_texts):
                return np.zeros((len(texts), 1))
                
            # 하나의 텍스트만 있는 경우
            if len(processed_texts) == 1:
                return np.ones((1, 1))
                
            return self.vectorizer.fit_transform(processed_texts).toarray()
        except Exception as e:
            logger.error(f"Error in get_embeddings: {str(e)}")
            return np.ones((len(texts), 1))

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return "empty"
        text = text.strip()
        text = ' '.join(text.split())
        return text or "empty"

    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """기사들을 토큰 제한에 맞게 청크로 분할"""
        try:
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for article in articles:
                article_text = f"제목: {article['title']}\n내용: {article['content']}"
                tokens = len(article_text.split())
                
                if current_tokens + tokens > self.max_tokens_per_chunk:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = [article]
                    current_tokens = tokens
                else:
                    current_chunk.append(article)
                    current_tokens += tokens
            
            if current_chunk:
                chunks.append(current_chunk)
                
            return chunks
        except Exception as e:
            logger.error(f"Error in chunk_articles: {str(e)}")
            return [[article] for article in articles]

    def analyze_chunk(self, chunk: List[Dict[str, Any]]) -> Dict[str, str]:
        """단일 청크에 대한 통합 분석 수행"""
        try:
            texts = []
            for idx, article in enumerate(chunk):
                text = f"[기사 {idx+1}]\n"
                text += f"날짜: {article.get('date', 'Unknown')}\n"
                text += f"제목: {article['title']}\n"
                text += f"내용: {article['content']}\n"
                texts.append(text)
                
            combined_text = "\n\n".join(texts)
            
            prompt = f"""다음은 하나의 이슈에 대한 시간순으로 정렬된 연속된 기사들입니다. 
                        전체 기사들을 종합적으로 분석하여 다음 세 가지 관점에서 각각 100자 내외로 요약해주세요.
                        
                        1. 배경:
                        - 이슈가 처음 등장하게 된 근본적인 원인
                        - 이슈가 발생하게 된 사회적/정치적/경제적 맥락
                        
                        2. 핵심 내용:
                        - 시간의 흐름에 따른 주요 사건들의 전개 과정
                        - 각 이해관계자들의 입장 변화와 대응
                        - 이슈의 핵심 쟁점이 어떻게 발전/변화했는지
                        
                        3. 결론 및 전망:
                        - 이슈가 현재까지 어떻게 진행/해결되었는지
                        - 각 이해관계자들의 최종 입장과 합의 사항
                        - 향후 진행 방향과 예상되는 파급 효과
                        
                        기사들:
                        {combined_text}
                        
                        다음 형식으로 답변해주세요:
                        배경: (여기에 배경 설명)
                        핵심내용: (여기에 핵심 내용 설명)
                        결론: (여기에 결론 및 전망 설명)"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 여러 뉴스 기사들을 분석하여 이슈의 배경, 핵심 내용, 결론을 도출하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            parts = content.split('\n')
            result = {}
            
            for part in parts:
                if part.startswith('배경:'):
                    result['background'] = part[3:].strip()
                elif part.startswith('핵심내용:'):
                    result['core_content'] = part[5:].strip()
                elif part.startswith('결론:'):
                    result['conclusion'] = part[3:].strip()
                    
            return result
        except Exception as e:
            logger.error(f"Error in analyze_chunk: {str(e)}")
            return {
                'background': f"분석 중 오류 발생: {str(e)}",
                'core_content': "",
                'conclusion': ""
            }

    def merge_analyses(self, analyses: List[Dict[str, str]]) -> Dict[str, str]:
        """여러 청크의 분석 결과를 통합"""
        if not analyses:
            return {'background': "", 'core_content': "", 'conclusion': ""}
            
        all_texts = {
            'background': [a['background'] for a in analyses if a.get('background')],
            'core_content': [a['core_content'] for a in analyses if a.get('core_content')],
            'conclusion': [a['conclusion'] for a in analyses if a.get('conclusion')]
        }
        
        try:
            prompt = """다음은 여러 시기별로 분석된 내용들입니다. 이를 통합하여 각각 100자 내외로 최종 요약해주세요.

            배경:
            {}

            핵심내용:
            {}

            결론:
            {}

            다음 형식으로 답변해주세요:
            배경: (통합된 배경 설명)
            핵심내용: (통합된 핵심 내용 설명)
            결론: (통합된 결론 설명)""".format(
                "\n".join(all_texts['background']),
                "\n".join(all_texts['core_content']),
                "\n".join(all_texts['conclusion'])
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 여러 분석 결과들을 통합하여 최종 요약을 만드는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            parts = content.split('\n')
            result = {}
            
            for part in parts:
                if part.startswith('배경:'):
                    result['background'] = part[3:].strip()
                elif part.startswith('핵심내용:'):
                    result['core_content'] = part[5:].strip()
                elif part.startswith('결론:'):
                    result['conclusion'] = part[3:].strip()
                    
            return result
        except Exception as e:
            logger.error(f"Error in merge_analyses: {str(e)}")
            return {
                'background': "분석 결과 통합 중 오류 발생",
                'core_content': "",
                'conclusion': ""
            }

    def analyze_story(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 분석 프로세스 실행"""
        try:
            logger.info(f"Starting analysis with {len(articles)} articles")
            
            # 날짜 범위 계산
            dates = [article.get('date', '') for article in articles if article.get('date')]
            start_date = min(dates) if dates else datetime.now().isoformat()
            end_date = max(dates) if dates else datetime.now().isoformat()
            
            # 전처리
            processed_articles = self.preprocess_articles(articles)
            logger.info(f"Preprocessed articles: {len(processed_articles)}")
            
            if not processed_articles:
                logger.warning("No articles after preprocessing")
                return self._create_empty_analysis()
            
            # 청크 분할
            chunks = self.chunk_articles(processed_articles)
            logger.info(f"Created {len(chunks)} chunks")
            
            # 청크별 분석
            chunk_analyses = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
                analysis = self.analyze_chunk(chunk)
                chunk_analyses.append(analysis)
            
            # 분석 결과 통합
            final_analysis = self.merge_analyses(chunk_analyses)
            
            return {
                **final_analysis,
                'article_count': len(articles),
                'processed_article_count': len(processed_articles),
                'chunk_count': len(chunks),
                'processed_at': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_story: {str(e)}")
            return self._create_empty_analysis()

    def _create_empty_analysis(self) -> Dict[str, Any]:
        """빈 분석 결과 생성"""
        current_time = datetime.now().isoformat()
        return {
            'background': "",
            'core_content': "",
            'conclusion': "",
            'article_count': 0,
            'processed_article_count': 0,
            'chunk_count': 0,
            'processed_at': current_time,
            'date_range': {
                'start': current_time,
                'end': current_time
            }
        }

    def preprocess_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """기사들을 전처리하고 중요도에 따라 필터링"""
        if not articles:
            return []
            
        try:
            # 시간순 정렬
            sorted_articles = sorted(articles, key=lambda x: x.get('date', ''))
            grouped_articles = []
            
            # 날짜별로 그룹화
            for date, group in groupby(sorted_articles, key=lambda x: x.get('date', '')[:10]):
                group_list = list(group)
                
                if len(group_list) > 3:
                    try:
                        titles = [article['title'] for article in group_list if article.get('title')]
                        if len(titles) > 1:
                            embeddings = self.get_embeddings(titles)
                            if embeddings.size > 0:
                                centrality_scores = np.mean(cosine_similarity(embeddings), axis=1)
                                top_indices = np.argsort(centrality_scores)[-3:]
                                group_list = [group_list[i] for i in top_indices]
                    except Exception as e:
                        logger.error(f"Error in group processing: {str(e)}")
                        group_list = group_list[:3]
                
                grouped_articles.extend(group_list)
            
            return grouped_articles
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return articles[:min(len(articles), 10)]  # 오류 발생시 최대 10개만 반환