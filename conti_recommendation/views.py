import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import ContiRequestSerializer, ContiResponseSerializer
from .services import load_data, create_conti

logger = logging.getLogger(__name__)

class CreateContiView(APIView):
    def post(self, request):
        serializer = ContiRequestSerializer(data=request.data)
        if serializer.is_valid():
            keywords = serializer.validated_data['keywords']
            bible_verse_range = serializer.validated_data['bible_verse_range']

            try:
                songs_df = load_data('data.csv') # 추후 실제 곡 DB와 연결
                if songs_df is None:
                    logger.error("Failed to load songs data.")
                    return Response(
                        {"success": False, "error": "Internal server error."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                conti = create_conti(songs_df, keywords, bible_verse_range)

                if "error" in conti:
                    logger.error(f"Conti creation error: {conti['error']}")
                    return Response(
                        {"success": False, "error": "Failed to generate conti."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                else:
                    response_serializer = ContiResponseSerializer(conti)
                    return Response(
                        {"success": True, "data": response_serializer.data},
                        status=status.HTTP_200_OK
                    )
            except Exception as e:
                logger.exception("An unexpected error occurred during conti creation.")
                return Response(
                    {"success": False, "error": "Internal server error."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        else:
            logger.warning(f"Invalid input data: {serializer.errors}")
            return Response(
                {"success": False, "error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
