# conti_recommendation/apps.py
from django.apps import AppConfig

class ContiRecommendationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'conti_recommendation'

    def ready(self):
        from .services import initialize_services
        initialize_services()
