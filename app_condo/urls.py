from django.urls import path
from . import views

app_name = 'app_condo'

urlpatterns = [
    path('', views.predict, name='predict'),  # Root URL
    path('predict/', views.predict, name='predict'),  # Direct predict URL
    path('explore/', views.explore_view, name='explore'),
    path('loan_table/', views.loan_table_view, name='loan_table'),
    path('get-subdistricts/<str:district_name>/', views.get_subdistricts, name='get_subdistricts'),
    path('get-nearest-roads/<str:subdistrict_name>/', views.get_nearest_roads, name='get_nearest_roads'),
    path('explore/district_psm/<str:district>/', views.district_psm, name='district_psm'),
    path('explore/distance_psm/<str:distance_field>/', views.distance_psm, name='distance_psm'),
    path('explore/facilities/<str:district>/', views.facilities, name='facilities'),
]
