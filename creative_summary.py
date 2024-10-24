import os
import torch
import numpy as np
import pandas as pd
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class NewsClusteringSummarizer:
    def __init__(self, api_key=None):
        # BERT 모델 초기화 (문장 임베딩용)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.bert_model = AutoModel.from_pretrained("klue/bert-base")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """문장들을 BERT 임베딩으로 변환"""
        embeddings = []
        
        for text in texts:
            # 제목과 내용을 함께 사용하여 임베딩 생성
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
                
            # 현재 기사와 유사한 기사들을 찾아 클러스터 형성
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

        # 제목과 내용을 합쳐서 임베딩 생성
        texts = [f"{article['title']} {article['content']}" for article in articles]
        embeddings = self.get_embeddings(texts)
        
        # 코사인 유사도 행렬 계산
        similarity_matrix = cosine_similarity(embeddings)
        
        # 클러스터 찾기
        clusters_indices = self.find_clusters(similarity_matrix, threshold=0.8)
        
        # 인덱스를 실제 기사로 변환
        clusters = []
        for cluster_indices in clusters_indices:
            cluster = [articles[idx] for idx in cluster_indices]
            clusters.append(cluster)
            
        return clusters

    def summarize_cluster(self, cluster: List[Dict[str, Any]]) -> str:
        """군집 내 기사들을 GPT를 사용하여 요약"""
        # 기사 내용 결합
        texts = [f"제목: {article['title']}\n내용: {article['content']}" for article in cluster]
        combined_text = "\n\n".join(texts)
        
        prompt = f"""다음은 같은 사건을 다룬 여러 기사들입니다. 
                    이 기사들의 핵심 내용을 50자 이내로 요약해주세요.

                    기사들:
                    {combined_text}

                    요약:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 뉴스 기사를 정확하고 간단히 요약하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"요약 생성 중 오류 발생: {str(e)}"

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """전체 처리 프로세스 실행"""
        clusters = self.cluster_articles(articles)
        
        results = []
        for cluster in clusters:
            summary = self.summarize_cluster(cluster)
            results.append({
                'articles': cluster,
                'summary': summary,
                'article_count': len(cluster),
                'processed_at': datetime.now().isoformat(),
                'titles': [article['title'] for article in cluster]  # 클러스터에 포함된 기사 제목들
            })
            
        return results

def main():
    # API 키 설정
    api_key = OPENAI_API_KEY  # 실제 API 키로 교체 필요
    
    try:
        # JSON 파일에서 기사 데이터 로드
        with open('extracted_articles.json', 'r', encoding='utf-8') as file:
            articles = json.load(file)
        
        # GPT 요약기 초기화
        summarizer = NewsClusteringSummarizer(api_key=api_key)
        results = summarizer.process_articles(articles)
        
        # 결과 출력
        for i, result in enumerate(results, 1):
            print(f"\n클러스터 {i}:")
            print(f"포함된 기사 수: {result['article_count']}")
            print(f"포함된 기사 제목: {', '.join(result['titles'])}")
            print(f"요약: {result['summary']}")
            print(f"처리 시간: {result['processed_at']}")
        
        # 결과를 JSON 파일로 저장
        with open('summarized_articles.json', 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
        
        print("\n요약 결과가 summarized_articles.json 파일에 저장되었습니다.")
    
    except FileNotFoundError:
        print("오류: extracted_articles.json 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print("오류: JSON 파일 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()