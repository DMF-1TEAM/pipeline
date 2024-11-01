import os
import torch
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class NewsClusteringSummarizer:
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

    def find_clusters(self, similarity_matrix: np.ndarray, threshold: float = 0.5) -> List[List[int]]:
        """유사도 기반으로 기사 클러스터링"""
        n = len(similarity_matrix)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
                    
            clusters.append(cluster)
            
        return clusters

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """기사들을 유사도 기반으로 군집화"""
        if len(articles) < 2:
            return [articles]

        # 제목만으로 클러스터링
        titles = [article['title'] for article in articles]
        embeddings = self.get_embeddings(titles)
        
        similarity_matrix = cosine_similarity(embeddings)
        clusters_indices = self.find_clusters(similarity_matrix, threshold=0.7)
        
        clusters = []
        for cluster_indices in clusters_indices:
            cluster = [articles[idx] for idx in cluster_indices]
            clusters.append(cluster)
            
        return clusters

    def summarize_titles(self, cluster: List[Dict[str, Any]]) -> str:
        """제목만 사용하여 요약"""
        titles = [article['title'] for article in cluster]
        combined_titles = "\n".join(titles)
        
        prompt = f"""다음은 같은 사건을 다룬 여러 기사의 제목들입니다. 
                    이 제목들의 핵심 내용을 30자 이내로 요약해주세요.

                    제목들:
                    {combined_titles}

                    요약:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 뉴스 제목을 간단명료하게 요약하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"제목 요약 중 오류 발생: {str(e)}"

    def summarize_contents(self, cluster: List[Dict[str, Any]]) -> str:
        """전체 내용을 사용하여 요약"""
        texts = [f"제목: {article['title']}\n내용: {article['content']}" for article in cluster]
        combined_text = "\n\n".join(texts)
        
        prompt = f"""다음은 같은 사건을 다룬 여러 기사들입니다. 
                    이 기사들의 핵심 내용을 100자 이내로 상세히 요약해주세요.

                    기사들:
                    {combined_text}

                    요약:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 뉴스 기사를 정확하고 상세하게 요약하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"내용 요약 중 오류 발생: {str(e)}"

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """전체 처리 프로세스 실행"""
        clusters = self.cluster_articles(articles)
        
        results = []
        for cluster in clusters:
            results.append({
                'title_summary': self.summarize_titles(cluster),
                'content_summary': self.summarize_contents(cluster),
                'article_count': len(cluster),
                'processed_at': datetime.now().isoformat(),
                'titles': [article['title'] for article in cluster]
            })
            
        return results