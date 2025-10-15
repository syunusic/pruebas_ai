from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import DetailView, UpdateView

from .forms import AlumniProfileForm
from .models import AlumniProfile


class ProfileDetailView(LoginRequiredMixin, DetailView):
    model = AlumniProfile
    template_name = 'profiles/profile_detail.html'

    def get_object(self, queryset=None):
        queryset = (
            AlumniProfile.objects
            .select_related('user')
            .prefetch_related('audit_logs__changed_by')
        )
        return queryset.get(user=self.request.user)


class ProfileUpdateView(LoginRequiredMixin, UpdateView):
    model = AlumniProfile
    form_class = AlumniProfileForm
    template_name = 'profiles/profile_form.html'
    success_url = reverse_lazy('profiles:profile_detail')

    def get_object(self, queryset=None):
        queryset = AlumniProfile.objects.select_related('user')
        return queryset.get(user=self.request.user)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        response = super().form_valid(form)
        full_name = self.request.POST.get('full_name', '').strip()
        if full_name and full_name != self.request.user.full_name:
            self.request.user.full_name = full_name
            self.request.user.save(update_fields=['full_name'])
        messages.success(self.request, 'Perfil actualizado correctamente.')
        return response
