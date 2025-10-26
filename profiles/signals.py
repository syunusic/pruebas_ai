from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

from accounts.models import User
from .models import AlumniProfile, ProfileAuditLog


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        AlumniProfile.objects.create(user=instance)


@receiver(pre_save, sender=AlumniProfile)
def log_profile_updates(sender, instance, **kwargs):
    if not instance.pk:
        return

    previous = AlumniProfile.objects.get(pk=instance.pk)
    changed_fields = {}

    track_fields = [
        'parents_status',
        'marital_status',
        'children_count',
        'job_title',
        'company_name',
        'company_start_year',
        'job_summary',
        'email_visible',
        'consent_given',
        'consent_timestamp',
    ]

    for field in track_fields:
        old_value = getattr(previous, field)
        new_value = getattr(instance, field)
        if old_value != new_value:
            if hasattr(old_value, 'isoformat'):
                old_serialized = old_value.isoformat() if old_value else old_value
            else:
                old_serialized = old_value
            if hasattr(new_value, 'isoformat'):
                new_serialized = new_value.isoformat() if new_value else new_value
            else:
                new_serialized = new_value
            changed_fields[field] = {'old': old_serialized, 'new': new_serialized}

    if changed_fields:
        ProfileAuditLog.objects.create(
            profile=instance,
            changed_by=getattr(instance, '_updated_by', None),
            changes=changed_fields,
            created_at=timezone.now(),
        )
