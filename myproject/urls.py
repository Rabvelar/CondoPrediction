# myproject/urls.py
from django.contrib import admin
from django.urls import path, include
from app_condo.views import prediction_form

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', prediction_form, name='home'),  # Root URL points to prediction form
    path('predict/', include('app_condo.urls')),  # Predict app URLs
]
