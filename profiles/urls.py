from django.urls import path

from .views import ProfileDetailView, ProfileListView, ProfileUpdateView

app_name = 'profiles'

urlpatterns = [
    path('', ProfileListView.as_view(), name='profile_list'),
    path('perfil/', ProfileDetailView.as_view(), name='profile_detail'),
    path('perfil/editar/', ProfileUpdateView.as_view(), name='profile_edit'),
]
