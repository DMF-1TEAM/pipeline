import os
import torch
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
from operator import itemgetter

class NewsStoryAnalyzer:
    def __init__(self, api_key=None, max_tokens_per_chunk=4000):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.bert_model = AutoModel.from_pretrained("klue/bert-base")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def preprocess_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """기사들을 전처리하고 중요도에 따라 필터링"""
        # 1. 시간순 정렬
        sorted_articles = sorted(articles, key=lambda x: x.get('date', ''))
        
        # 2. 날짜별로 그룹화
        grouped_articles = []
        for date, group in groupby(sorted_articles, key=lambda x: x.get('date', '')[:10]):
            group_list = list(group)
            
            # 3. 각 그룹 내에서 임베딩 기반으로 대표 기사 선정
            if len(group_list) > 3:  # 하루에 3개 이상의 기사가 있는 경우
                titles = [article['title'] for article in group_list]
                embeddings = self.get_embeddings(titles)
                
                # 중심성이 높은 상위 3개 기사 선택
                centrality_scores = np.mean(cosine_similarity(embeddings), axis=1)
                top_indices = np.argsort(centrality_scores)[-3:]
                group_list = [group_list[i] for i in top_indices]
            
            grouped_articles.extend(group_list)
        
        return grouped_articles

    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """기사들을 토큰 제한에 맞게 청크로 분할"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for article in articles:
            # 기사 텍스트 구성
            article_text = f"제목: {article['title']}\n내용: {article['content']}"
            tokens = len(self.tokenizer.encode(article_text))
            
            if current_tokens + tokens > self.max_tokens_per_chunk:
                if current_chunk:  # 현재 청크가 있으면 저장
                    chunks.append(current_chunk)
                current_chunk = [article]
                current_tokens = tokens
            else:
                current_chunk.append(article)
                current_tokens += tokens
        
        if current_chunk:  # 마지막 청크 저장
            chunks.append(current_chunk)
            
        return chunks

    def analyze_chunk(self, chunk: List[Dict[str, Any]]) -> Dict[str, str]:
        """단일 청크에 대한 통합 분석 수행"""
        # 청크 내 기사들을 텍스트로 변환
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

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # 정확한 모델명으로 수정
                messages=[
                    {"role": "system", "content": "당신은 여러 뉴스 기사들을 분석하여 이슈의 배경, 핵심 내용, 결론을 도출하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # 응답 파싱
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
            return {
                'background': f"분석 중 오류 발생: {str(e)}",
                'core_content': "",
                'conclusion': ""
            }

    def merge_analyses(self, analyses: List[Dict[str, str]]) -> Dict[str, str]:
        """여러 청크의 분석 결과를 통합"""
        if not analyses:
            return {
                'background': "",
                'core_content': "",
                'conclusion': ""
            }
            
        # 각 섹션별로 모든 텍스트 결합
        all_texts = {
            'background': [a['background'] for a in analyses],
            'core_content': [a['core_content'] for a in analyses],
            'conclusion': [a['conclusion'] for a in analyses]
        }
        
        # 최종 통합 분석을 위한 프롬프트
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

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 여러 분석 결과들을 통합하여 최종 요약을 만드는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # 응답 파싱
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
            return {
                'background': f"최종 통합 중 오류 발생: {str(e)}",
                'core_content': "",
                'conclusion': ""
            }

    def analyze_story(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 분석 프로세스 실행"""
        # 1. 전처리 및 필터링
        processed_articles = self.preprocess_articles(articles)
        
        # 2. 청크 분할
        chunks = self.chunk_articles(processed_articles)
        
        # 3. 각 청크별 분석
        chunk_analyses = []
        for chunk in chunks:
            analysis = self.analyze_chunk(chunk)
            chunk_analyses.append(analysis)
        
        # 4. 분석 결과 통합
        final_analysis = self.merge_analyses(chunk_analyses)
        
        # 5. 메타데이터 추가
        return {
            **final_analysis,
            'article_count': len(articles),
            'processed_article_count': len(processed_articles),
            'chunk_count': len(chunks),
            'processed_at': datetime.now().isoformat(),
            'article_titles': [article['title'] for article in processed_articles],
            'date_range': {
                'start': min(article.get('date', 'Unknown') for article in articles),
                'end': max(article.get('date', 'Unknown') for article in articles)
            }
        }