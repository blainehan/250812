# RealEstate Aggregator (Vercel + FastAPI)

ChatGPT/Actions 또는 OpenAI Function-calling에서 호출 가능한 **부동산 집계 API**의 최소 구현입니다.
현재는 **개별공시지가**(원/㎡) 조회가 동작하며, 토지대장·건축물대장은 엔드포인트 매핑 TODO로 남겨두었습니다.

## 배포
1. 이 리포를 GitHub에 푸시
2. Vercel에서 Import → Deploy
3. 환경변수
   - `PUBLICDATA_KEY` : 공공데이터포털 **URL-인코딩된** 서비스키
   - `SERVICE_API_KEY` : (선택) 우리 API 보호용 키

## 로컬/원격 테스트
```bash
curl "https://YOUR-PROJECT.vercel.app/realestate/by-pnu/1111010100100010000?stdrYear=2025"
# 보호용 헤더 사용 시
curl -H "X-API-Key: YOUR_SERVICE_API_KEY"       "https://YOUR-PROJECT.vercel.app/realestate/by-pnu/1111010100100010000?stdrYear=2025"
```

## TODO
- 토지대장/건축물대장: 공식 문서에 맞춰 엔드포인트/필드 정확 매핑
- 주소→PNU 변환(JUSO) 유틸
- Redis 캐시, 레이트리밋/재시도 개선
- 응답 `source`, `referenceDate` 추가
