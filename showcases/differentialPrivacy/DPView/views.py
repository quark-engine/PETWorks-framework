from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.utils.translation import gettext
from django.urls import reverse
from django.views import View

from DPView import DP_View
from DPView.forms import ParameterForm

from general.views import AbstractMethodView
from general.views import AbstractExecuteView
from general.views import AbstractBreakProgramView

from general.views import FileView
from general.function import DataframeDetection
from general.function import Path

from .models import FileModel
import pandas as pd
import json

@login_required
def index(request):
    request_dict = {}
    request_dict['file_upload_url'] = reverse('DPView:file_upload', args=['DPView'])
    request_dict['custom_url'] = reverse('DPView:custom')
    request_dict['caller'] = 'DPView'
    return render(request, 'DPView/DPView_home.html', request_dict)
          
class CustomView(View):
    @method_decorator(login_required)
    def get(self, request, *arg, **kwargs):
        file_name = kwargs.get('csv_name')
        if not file_name:
            return redirect('home')
            
        request_dict = self.get_request_dict(request, *arg, **kwargs)
        return render(request, 'general/parameter_custom.html', request_dict)

    @method_decorator(login_required)
    def post(self, request, *arg, **kwargs):
        file_name = kwargs.get('csv_name')
        if not file_name:
            return redirect('home')
            
        title_id_pair = request.POST.get('title_id_pair', None)
        interval_dict = request.POST.get('interval_dict', None)
        structure_dict = request.POST.get('structure_dict', None)
        almost_number_is_empty_dict = request.POST.get('almost_number_is_empty_dict', None)
        request_dict = self.get_request_dict(request, *arg, **kwargs)
        request_dict['title_id_pair'] = title_id_pair
        request_dict['interval_dict'] = interval_dict
        request_dict['structure_dict'] = structure_dict
        request_dict['almost_number_is_empty_dict'] = almost_number_is_empty_dict
        return render(request, 'general/parameter_custom.html', request_dict)
    
    def get_request_dict(self, request, *arg, **kwargs):
        path = Path()
        
        file_name = kwargs.get('csv_name')        
        caller = path.get_caller(request)
        
        file_path = path.get_upload_path(request, file_name, caller=caller)
        data_frame = DataframeDetection(file_path)
        number_title_list = data_frame.get_number_title()
        almost_number_dict = data_frame.get_almost_number_element()
                
        request_dict = {}
        request_dict = self.set_url_path(request_dict, caller, file_name)
        request_dict['caller'] = caller
        request_dict['number_title_list'] = number_title_list
        request_dict['almost_number_dict'] = almost_number_dict
        request_dict['file_name'] = file_name
        request_dict['custom_mode'] = 'DPView'
        return request_dict
    
    def set_url_path(self, request_dict, caller, file_name):
        request_dict['create_json'] = reverse(caller+':create_json')
        request_dict['title_check'] = reverse(caller+':title_check')
        request_dict['advanced_settings_url'] = reverse(caller+':advanced_settings', args=[file_name])
        request_dict['base_settings_url'] = reverse(caller+':custom')+file_name+'/'
        request_dict['previous_page_url'] = reverse(caller+':home')
        request_dict['execute_url'] = reverse(caller+':execute_page', args=[file_name])
        request_dict['upload_display_url'] = reverse(caller+':display', args=['upload'])
        return request_dict
        
class AdvancedSettingsView(CustomView):
    @method_decorator(login_required)
    def get(self, request, *arg, **kwargs):
        file_name = kwargs.get('csv_name')
        if not file_name:
            return redirect('home')
            
        request_dict = self.get_request_dict(request, *arg, **kwargs)
        return render(request, 'general/parameter_custom.html', request_dict)
        
    @method_decorator(login_required)
    def post(self, request, *arg, **kwargs):
        file_name = kwargs.get('csv_name')
        if not file_name:
            return redirect('home')
            
        title_id_pair = request.POST.get('title_id_pair', None)
        interval_dict = request.POST.get('interval_dict', None)
        structure_dict = request.POST.get('structure_dict', None)
        almost_number_is_empty_dict = request.POST.get('almost_number_is_empty_dict', None)
        request_dict = self.get_request_dict(request, *arg, **kwargs)
        request_dict['title_id_pair'] = title_id_pair
        request_dict['interval_dict'] = interval_dict
        request_dict['structure_dict'] = structure_dict
        request_dict['almost_number_is_empty_dict'] = almost_number_is_empty_dict
        return render(request, 'general/parameter_custom.html', request_dict)
        
    def get_request_dict(self, request, *arg, **kwargs):
        path = Path()
        
        file_name = kwargs.get('csv_name')
        caller = path.get_caller(request)
        
        file_path = path.get_upload_path(request, file_name, caller=caller)
        data_frame = DataframeDetection(file_path)
        type_pair = data_frame.get_type_pair()
        number_title_list = data_frame.get_number_title(type_pair=type_pair)
        number_type_pair = data_frame.get_number_type_pair(number_title_list, type_pair=type_pair)
        max_value_dict, min_value_dict = data_frame.get_number_limit(number_title_list, number_type_pair=number_type_pair)
        
        request_dict = super().get_request_dict(request, *arg, **kwargs)
        request_dict['advanced_settings'] = True
        request_dict['type_pair'] = type_pair
        request_dict['number_type_pair'] = number_type_pair
        request_dict['max_value_dict'] = max_value_dict
        request_dict['min_value_dict'] = min_value_dict
        return request_dict        

class DPViewView(AbstractMethodView):
    def get_form(self, requestContent):
        return ParameterForm(requestContent)
        
    def method_run(self, request):
        DP_View.run(request)
        
    def get_method_template(self):
        return 'DPView/DPView.html'

class BreakProgramView(AbstractBreakProgramView):
    def break_program(self, file):
        DP_View.break_program(file)

class ExecuteView(AbstractExecuteView):
    def get_empty_form(self):
        return ParameterForm()