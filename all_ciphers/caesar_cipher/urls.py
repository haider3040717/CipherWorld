from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),

    path('caesar/', views.caesar_encrypt, name='caesar_encrypt'),
    path('decrypt_caesar/', views.decrypt_caesar, name='decrypt_caesar'),

    path('playfair/', views.playfair_encrypt, name='playfair_encrypt'),
    path('decrypt_playfair/', views.playfair_decrypt, name='playfair_decrypt'),

    path('vigenere/', views.vigenere_encrypt, name='vigenere_encrypt'),
    path('decrypt_vigenere/', views.vigenere_decrypt, name='vigenere_decrypt'),

    path('one_time_pad/', views.encrypt_otp, name='encrypt_otp'),
    path('decrypt_one_time_pad/', views.decrypt_otp, name='decrypt_otp'),

    path('row_column/', views.row_column_encrypt, name='row_column_encrypt'),
    path('decrypt_row_column/', views.row_column_decrypt, name='row_column_decrypt'),

    path('encrypt_hill/', views.hill_encrypt, name='hill_encrypt'),
    path('decrypt_hill/', views.hill_decrypt, name='hill_decrypt'),

    path('encrypt_rail_fence/', views.rail_fence_encrypt, name='rail_fence_encrypt'),
    path('decrypt_rail_fence/', views.rail_fence_decrypt, name='rail_fence_decrypt'),
]