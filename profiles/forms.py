from django import forms
from django.utils import timezone

from .models import AlumniProfile


class AlumniProfileForm(forms.ModelForm):
    consent = forms.BooleanField(
        label='Autorizo que mis datos se compartan dentro del curso',
        required=False,
    )

    class Meta:
        model = AlumniProfile
        fields = [
            'parents_status',
            'marital_status',
            'children_count',
            'job_title',
            'job_summary',
            'email_visible',
        ]
        widgets = {
            'job_summary': forms.Textarea(attrs={'rows': 4}),
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user')
        super().__init__(*args, **kwargs)
        self.fields['consent'].initial = self.instance.consent_given
        self.fields['children_count'].min_value = 0

    def save(self, commit=True):
        profile = super().save(commit=False)
        profile._updated_by = self.user
        if self.cleaned_data.get('consent') and not profile.consent_given:
            profile.consent_given = True
            profile.consent_timestamp = timezone.now()
        elif not self.cleaned_data.get('consent') and profile.consent_given:
            profile.consent_given = False
            profile.consent_timestamp = None
        if commit:
            profile.save()
        return profile
