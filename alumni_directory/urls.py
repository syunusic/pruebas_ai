from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include(('profiles.urls', 'profiles'), namespace='profiles')),
    path('', RedirectView.as_view(pattern_name='profiles:profile_detail', permanent=False)),
]
