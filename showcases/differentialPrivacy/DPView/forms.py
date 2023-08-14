from django import forms
from general.forms import AbstractForm
from django.utils.translation import gettext

class ParameterForm(AbstractForm):
    num_bucket = forms.IntegerField(label='Num_Bucket', initial=3, min_value=1, help_text = '')
    privacy_budget = forms.FloatField(label='Privacy_Budget', initial=1.0, min_value=0, help_text = '')