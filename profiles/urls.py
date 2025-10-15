from django.urls import path

from .views import ProfileDetailView, ProfileUpdateView

app_name = 'profiles'

urlpatterns = [
    path('perfil/', ProfileDetailView.as_view(), name='profile_detail'),
    path('perfil/editar/', ProfileUpdateView.as_view(), name='profile_edit'),
]
