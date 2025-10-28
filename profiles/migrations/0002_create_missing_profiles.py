from django.db import migrations


def create_missing_profiles(apps, schema_editor):
    User = apps.get_model('accounts', 'User')
    AlumniProfile = apps.get_model('profiles', 'AlumniProfile')

    existing_user_ids = AlumniProfile.objects.values_list('user_id', flat=True)
    missing_users = User.objects.exclude(id__in=existing_user_ids)

    profiles_to_create = [
        AlumniProfile(user=user)
        for user in missing_users
    ]
    if profiles_to_create:
        AlumniProfile.objects.bulk_create(profiles_to_create)


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0001_initial'),
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(create_missing_profiles, migrations.RunPython.noop),
    ]
