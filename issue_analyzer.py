import os
import torch
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class NewsStoryAnalyzer:
    def __init__(self, api_key=None):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.bert_model = AutoModel.from_pretrained("klue/bert-base")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """문장들을 BERT 임베딩으로 변환"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
            
        return np.array(embeddings)

    def get_background(self, articles: List[Dict[str, Any]]) -> str:
        """사건의 배경 정보 추출"""
        # 시간순으로 정렬
        sorted_articles = sorted(articles, key=lambda x: x.get('date', ''))
        
        # 전체 컨텍스트 구성
        texts = []
        for idx, article in enumerate(sorted_articles):
            text = f"[기사 {idx+1}]\n"
            text += f"날짜: {article.get('date', 'Unknown')}\n"
            text += f"제목: {article['title']}\n"
            text += f"내용: {article['content']}\n"
            texts.append(text)
        
        combined_text = "\n\n".join(texts)
        
        prompt = f"""다음은 하나의 이슈에 대한 시간순으로 정렬된 연속된 기사들입니다. 
                    전체 기사들을 종합적으로 분석하여 이 이슈의 발생 배경을 100자 내외로 요약해주세요.
                    
                    분석 포인트:
                    - 이 이슈가 처음 등장하게 된 근본적인 원인
                    - 이슈가 발생하게 된 사회적/정치적/경제적 맥락
                    
                    기사들:
                    {combined_text}

                    종합적인 배경 설명:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 여러 뉴스 기사들을 시간 순서대로 분석하여 이슈의 전체적인 배경을 객관적으로 설명하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"배경 분석 중 오류 발생: {str(e)}"

    def get_core_content(self, articles: List[Dict[str, Any]]) -> str:
        """사건의 핵심 내용 추출"""
        # 시간순으로 정렬
        sorted_articles = sorted(articles, key=lambda x: x.get('date', ''))
        
        # 전체 컨텍스트 구성
        texts = []
        for idx, article in enumerate(sorted_articles):
            text = f"[기사 {idx+1}]\n"
            text += f"날짜: {article.get('date', 'Unknown')}\n"
            text += f"제목: {article['title']}\n"
            text += f"내용: {article['content']}\n"
            texts.append(text)
            
        combined_text = "\n\n".join(texts)
        
        prompt = f"""다음은 하나의 이슈에 대한 시간순으로 정렬된 연속된 기사들입니다. 
                    전체 기사들을 종합적으로 분석하여 이 이슈의 핵심 내용을 100자 내외로 설명해주세요.
                    
                    분석 포인트:
                    - 시간의 흐름에 따른 주요 사건들의 전개 과정
                    - 각 이해관계자들의 입장 변화와 대응
                    - 이슈의 핵심 쟁점이 어떻게 발전/변화했는지
                    
                    기사들:
                    {combined_text}

                    종합적인 핵심 내용:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 여러 뉴스 기사들을 시간 순서대로 분석하여 이슈의 전체적인 흐름을 명확하게 전달하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"핵심 내용 분석 중 오류 발생: {str(e)}"

    def get_conclusion(self, articles: List[Dict[str, Any]]) -> str:
        """사건의 결론 및 향후 전망 추출"""
        # 시간순으로 정렬
        sorted_articles = sorted(articles, key=lambda x: x.get('date', ''))
        
        # 전체 컨텍스트 구성
        texts = []
        for idx, article in enumerate(sorted_articles):
            text = f"[기사 {idx+1}]\n"
            text += f"날짜: {article.get('date', 'Unknown')}\n"
            text += f"제목: {article['title']}\n"
            text += f"내용: {article['content']}\n"
            texts.append(text)
            
        combined_text = "\n\n".join(texts)
        
        prompt = f"""다음은 하나의 이슈에 대한 시간순으로 정렬된 연속된 기사들입니다. 
                    전체 기사들을 종합적으로 분석하여 이 이슈의 결과를 얘기해 주듯 현재상황을 100자 내외로 설명해주세요.
                    
                    분석 포인트:
                    - 이슈가 현재까지 어떻게 진행/해결되었는지
                    - 각 이해관계자들의 최종 입장과 합의 사항
                    - 이슈의 향후 진행 방향과 예상되는 파급 효과
                    - 유사한 과거 사례들과 비교했을 때의 전망
                    
                    기사들:
                    {combined_text}

                    종합적인 결론 및 전망:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 여러 뉴스 기사들을 시간 순서대로 분석하여 이슈의 결론을 도출하고 향후 전망을 예측하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"결론 분석 중 오류 발생: {str(e)}"

    def analyze_story(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """전체 분석 프로세스 실행"""
        return {
            'background': self.get_background(articles),
            'core_content': self.get_core_content(articles),
            'conclusion': self.get_conclusion(articles),
            'article_count': len(articles),
            'processed_at': datetime.now().isoformat(),
            'article_titles': [article['title'] for article in articles],
            'date_range': {
                'start': min(article.get('date', 'Unknown') for article in articles),
                'end': max(article.get('date', 'Unknown') for article in articles)
            }
        }