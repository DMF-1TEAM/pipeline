## 데이터 선별 및 요약 프로세스


1. [데이터 추출](https://github.com/DMF-1TEAM/pipeline/blob/main/extraction.py)

    1. 검색된 키워드 입력 <- 보완 필요 / 현재 테스트를 위해 input을 통해 키워드 받고있음
    2. 데이터베이스에서 키워드로 기사 검색
    3. 검색 결과를 날짜별로 정렬하고, impect_point를 찾는 함수
        * impect_point : 같은날 같은 키워드로 올라온 기사수가 10개 이상인 기사들
    4. 날짜별로 기사를 요약하는 함수 <- 데이터 요약 함수와 연동
    5. 요약된 결과를 데이터베이스에 저장하는 함수

2. [기사요약](https://github.com/DMF-1TEAM/pipeline/blob/main/creative_summary.py)

    1. 문장들을 BERT 임베딩으로 변환
    2. 유사도 기반으로 기사 클러스터링
    3. 기사들을 유사도 기반으로 군집화
    4. 군집 내 기사들을 GPT를 사용하여 요약
    5. 결과를 JSON 파일로 저장