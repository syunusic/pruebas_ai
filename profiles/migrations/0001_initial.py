from django.conf import settings
from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='AlumniProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('parents_status', models.CharField(blank=True, choices=[('both', 'Ambos vivos'), ('one', 'Uno vivo'), ('none', 'Ninguno vivo')], max_length=10, verbose_name='situación de los padres')),
                ('marital_status', models.CharField(blank=True, choices=[('single', 'Soltero/a'), ('married', 'Casado/a'), ('partnered', 'Convive'), ('separated', 'Separado/a'), ('divorced', 'Divorciado/a'), ('widowed', 'Viudo/a')], max_length=15, verbose_name='estado civil')),
                ('children_count', models.PositiveIntegerField(default=0, verbose_name='número de hijos')),
                ('job_title', models.CharField(blank=True, max_length=255, verbose_name='cargo o rol')),
                ('job_summary', models.TextField(blank=True, verbose_name='resumen del trabajo')),
                ('email_visible', models.BooleanField(default=True, help_text='Controla si los demás pueden ver el correo electrónico.', verbose_name='mostrar correo al curso')),
                ('consent_given', models.BooleanField(default=False, verbose_name='dio consentimiento')),
                ('consent_timestamp', models.DateTimeField(blank=True, null=True, verbose_name='fecha de consentimiento')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'perfil de exalumno',
                'verbose_name_plural': 'perfiles de exalumnos',
            },
        ),
        migrations.CreateModel(
            name='ProfileAuditLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('changes', models.JSONField(verbose_name='cambios realizados')),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('changed_by', models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, related_name='profile_changes', to=settings.AUTH_USER_MODEL)),
                ('profile', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='audit_logs', to='profiles.alumniprofile')),
            ],
            options={
                'verbose_name': 'registro de cambios de perfil',
                'verbose_name_plural': 'registros de cambios de perfil',
                'ordering': ('-created_at',),
            },
        ),
    ]
