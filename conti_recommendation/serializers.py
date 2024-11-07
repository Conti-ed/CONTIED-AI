from rest_framework import serializers

class ContiRequestSerializer(serializers.Serializer):
    keywords = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
    bible_verse_range = serializers.CharField()

class SongSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    # title = serializers.CharField()  # 필요 시 추가
    # artist = serializers.CharField()  # 필요 시 추가

class ContiResponseSerializer(serializers.Serializer):
    title = serializers.CharField()
    description = serializers.CharField()
    songs = SongSerializer(many=True)
    # 필요한 경우 필드 추가
    # createdAt = serializers.DateTimeField()
    # id = serializers.IntegerField()
