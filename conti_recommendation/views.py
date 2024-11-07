import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import ContiRequestSerializer, ContiResponseSerializer
from .services import create_conti

logger = logging.getLogger(__name__)

class CreateContiView(APIView):
    def post(self, request):
        serializer = ContiRequestSerializer(data=request.data)
        if serializer.is_valid():
            keywords = serializer.validated_data['keywords']
            bible_verse_range = serializer.validated_data['bible_verse_range']

            conti = create_conti(keywords, bible_verse_range)

            if "error" in conti:
                logger.error(f"Conti creation error: {conti['error']}")
                return Response(
                    {"success": False, "error": conti["error"]},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            else:
                response_serializer = ContiResponseSerializer(conti)
                return Response(
                    {"success": True, "data": response_serializer.data},
                    status=status.HTTP_200_OK
                )
        else:
            logger.warning(f"Invalid input data: {serializer.errors}")
            return Response(
                {"success": False, "error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )