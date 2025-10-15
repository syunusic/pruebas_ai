from datetime import datetime

from django import template
from django.utils import formats, timezone

from ..models import AlumniProfile

register = template.Library()

FIELD_LABELS = {
    'parents_status': 'Situación de los padres',
    'marital_status': 'Estado civil',
    'children_count': 'Número de hijos',
    'job_title': 'Cargo o rol',
    'job_summary': 'Descripción del trabajo',
    'email_visible': 'Mostrar correo al curso',
    'consent_given': 'Consentimiento otorgado',
    'consent_timestamp': 'Fecha de consentimiento',
}

@register.filter(name='display_change')
def display_change(value, field_name=''):
    if value in (None, ''):
        return '-'

    if isinstance(value, bool):
        return 'Sí' if value else 'No'

    if field_name == 'parents_status':
        return dict(AlumniProfile.ParentsStatus.choices).get(value, value)
    if field_name == 'marital_status':
        return dict(AlumniProfile.MaritalStatus.choices).get(value, value)

    if field_name == 'consent_timestamp':
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return value
        if timezone.is_naive(value):
            value = timezone.make_aware(value, timezone.get_current_timezone())
        return formats.date_format(value, 'DATETIME_FORMAT')

    return value


@register.filter(name='field_label')
def field_label(field_name):
    return FIELD_LABELS.get(field_name, field_name.replace('_', ' ').capitalize())
