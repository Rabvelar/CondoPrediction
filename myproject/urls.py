# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app_condo.urls')),  # Main app URLs
    path('predict/', include('app_condo.urls')),  # Direct predict URL
]
