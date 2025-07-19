from django import forms
from .models import ImageCheck


class MyModelForm(forms.ModelForm):
    class Meta:
        model = ImageCheck
        fields = ['file_name', 'file_url', 'check_result', 'check_time']
