# from django.shortcuts import render
# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import ContiRequestSerializer, ContiResponseSerializer
from .services import load_data, create_conti

class CreateContiView(APIView):
    def post(self, request):
        serializer = ContiRequestSerializer(data=request.data)
        if serializer.is_valid():
            user_keywords = serializer.validated_data['user_keywords']
            bible_verse_range = serializer.validated_data['bible_verse_range']

            # 데이터 로드 (데이터 파일 경로를 적절히 설정)
            songs_df = load_data('data.csv')  # 'data.csv' 파일의 경로를 정확히 지정
            if songs_df is None:
                return Response({"error": "Failed to load songs data."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            conti = create_conti(songs_df, user_keywords, bible_verse_range)

            if "error" in conti:
                return Response({"error": conti["error"]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                response_serializer = ContiResponseSerializer(conti)
                return Response(response_serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
