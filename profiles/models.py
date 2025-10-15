from django.conf import settings
from django.db import models
from django.utils import timezone


class AlumniProfile(models.Model):
    class ParentsStatus(models.TextChoices):
        BOTH_ALIVE = 'both', 'Ambos vivos'
        ONE_ALIVE = 'one', 'Uno vivo'
        NONE_ALIVE = 'none', 'Ninguno vivo'

    class MaritalStatus(models.TextChoices):
        SINGLE = 'single', 'Soltero/a'
        MARRIED = 'married', 'Casado/a'
        PARTNERED = 'partnered', 'Convive'
        SEPARATED = 'separated', 'Separado/a'
        DIVORCED = 'divorced', 'Divorciado/a'
        WIDOWED = 'widowed', 'Viudo/a'

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    parents_status = models.CharField(
        'situación de los padres',
        max_length=10,
        choices=ParentsStatus.choices,
        blank=True,
    )
    marital_status = models.CharField(
        'estado civil',
        max_length=15,
        choices=MaritalStatus.choices,
        blank=True,
    )
    children_count = models.PositiveIntegerField('número de hijos', default=0)
    job_title = models.CharField('cargo o rol', max_length=255, blank=True)
    job_summary = models.TextField('resumen del trabajo', blank=True)
    email_visible = models.BooleanField(
        'mostrar correo al curso',
        default=True,
        help_text='Controla si los demás pueden ver el correo electrónico.',
    )
    consent_given = models.BooleanField('dio consentimiento', default=False)
    consent_timestamp = models.DateTimeField('fecha de consentimiento', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'perfil de exalumno'
        verbose_name_plural = 'perfiles de exalumnos'

    def __str__(self):
        return f'Perfil de {self.user.full_name}'


class ProfileAuditLog(models.Model):
    profile = models.ForeignKey(AlumniProfile, on_delete=models.CASCADE, related_name='audit_logs')
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='profile_changes',
    )
    changes = models.JSONField('cambios realizados')
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name = 'registro de cambios de perfil'
        verbose_name_plural = 'registros de cambios de perfil'
        ordering = ('-created_at',)

    def __str__(self):
        return f'Cambios en {self.profile} ({self.created_at:%Y-%m-%d %H:%M})'
