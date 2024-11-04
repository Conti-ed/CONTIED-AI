from rest_framework import serializers

class ContiRequestSerializer(serializers.Serializer):
    user_keywords = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
    bible_verse_range = serializers.CharField()

class SongSerializer(serializers.Serializer):
    title = serializers.CharField()
    artist = serializers.CharField()
    similarity = serializers.FloatField()

class ContiResponseSerializer(serializers.Serializer):
    title = serializers.CharField()
    description = serializers.CharField()
    songs = SongSerializer(many=True)
    error = serializers.CharField(required=False)
